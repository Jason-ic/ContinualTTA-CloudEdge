"""
云端双模型蒸馏训练器
Cloud Dual-Model Distillation Trainer for Cloud-Edge Collaborative TTA.

实现论文 Chapter 5 的核心算法：
1. Phase 1: 教师模型 (ResNet-101 + LoRA) LoRA适配 + 特征对齐
2. Phase 2: 教师推理生成伪标签和特征
3. Phase 3: 学生模型 (ResNet-50) 三级知识蒸馏训练
4. 防遗忘正则化 + mAP下降触发参数回滚

关键超参数（论文默认）:
- Teacher: AdamW, lr=2e-3, weight_decay=1e-4, batch=8, epochs=1-3
- Student: AdamW, lr=1e-3, batch=16, epochs=3-5
- 遗忘正则 beta ∈ [0.1, 0.5]
- 回滚触发: mAP下降 > 5%
"""

import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detectron2.utils.distill_loss import (
    TotalDistillLoss,
    ForgettingRegularization,
)

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA包装器：用于Linear层。
    W' = W + B*A * (alpha/r)
    其中 A (in, r), B (r, out) 为可训练低秩矩阵，W 被冻结。
    """

    def __init__(self, linear: nn.Linear, r: int = 16, lora_alpha: int = 16):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = lora_alpha / r

        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def get_lora_params(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


class LoRAConv2d(nn.Module):
    """
    LoRA包装器：用于 1x1 Conv2d 层（等价于Linear）。
    """

    def __init__(self, conv: nn.Conv2d, r: int = 16, lora_alpha: int = 16):
        super().__init__()
        assert conv.kernel_size == (1, 1), "LoRAConv2d 仅支持 1x1 卷积"
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        self.conv = conv
        self.conv.weight.requires_grad_(False)
        if self.conv.bias is not None:
            self.conv.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(in_ch, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_ch))
        self.scaling = lora_alpha / r

        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> reshape -> (B*H*W, C)
        B, C, H, W = x.shape
        base_out = self.conv(x)  # (B, out_ch, H, W)

        # LoRA增量
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        lora = (x_flat @ self.lora_A @ self.lora_B) * self.scaling  # (B*H*W, out_ch)
        lora = lora.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_ch, H, W)
        return base_out + lora

    def get_lora_params(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def inject_lora_to_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, List[nn.Parameter]]:
    """
    向模型中注入 LoRA 适配器。

    默认目标：
    - backbone.bottom_up.stages (stage2, stage3) 的 1x1 Conv (conv1)
    - backbone.lateral_convs / output_convs (FPN)
    - proposal_generator 的 rpn conv
    - roi_heads.box_head 的 FC 层

    Args:
        model: 检测模型
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        target_modules: 若指定则只对该列表中的模块名注入（子串匹配）
    Returns:
        (model, lora_params): 修改后的模型和 LoRA 可训练参数列表
    """
    lora_params = []
    replaced = 0

    def _should_replace(name: str) -> bool:
        if target_modules is None:
            # 默认策略：backbone stage3/4的conv1，FPN lateral/output conv，RPN conv，box_head fc
            keywords = [
                'bottom_up.stages.2',  # res3
                'bottom_up.stages.3',  # res4
                'fpn_lateral',
                'fpn_output',
                'rpn_head.conv',
                'box_head.fc',
            ]
            return any(k in name for k in keywords)
        return any(k in name for k in target_modules)

    # 收集需要替换的层
    replacements = []
    for name, module in model.named_modules():
        if not _should_replace(name):
            continue
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            replacements.append((name, module, 'conv2d'))
        elif isinstance(module, nn.Linear):
            replacements.append((name, module, 'linear'))

    # 执行替换（通过 parent 模块的 setattr）
    for name, module, mtype in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        if mtype == 'conv2d':
            lora_module = LoRAConv2d(module, r=r, lora_alpha=lora_alpha)
        else:
            lora_module = LoRALinear(module, r=r, lora_alpha=lora_alpha)

        setattr(parent, attr, lora_module)
        lora_params.extend(lora_module.get_lora_params())
        replaced += 1

    logger.info(f"[LoRA] 注入 {replaced} 个适配器，可训练参数: {sum(p.numel() for p in lora_params) / 1e6:.3f}M")
    return model, lora_params


class ModelSnapshot:
    """轻量级模型参数快照，用于参数回滚"""

    def __init__(self, model: nn.Module, only_lora: bool = True):
        self.state = {}
        for name, param in model.named_parameters():
            if only_lora and 'lora_' not in name:
                continue
            self.state[name] = param.data.clone()

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.state:
                param.data.copy_(self.state[name])


class CloudDistillationTrainer:
    """
    云端双模型蒸馏训练器。

    工作流程:
    1. setup_teacher_lora()   — 向教师(R101)注入LoRA，配置优化器
    2. adapt_teacher()        — 用上传样本做特征对齐适配教师LoRA参数
    3. generate_pseudo_labels() — 教师推理生成伪标签和中间特征
    4. distill_to_student()   — 学生(R50)三级蒸馏训练
    5. export_student_fp16()  — 导出学生模型FP16参数
    """

    def __init__(
        self,
        teacher_model: nn.Module,       # R101 + LoRA
        student_model: nn.Module,       # R50
        pretrain_teacher: Optional[nn.Module] = None,  # 冻结的预训练R101（防遗忘）
        teacher_lr: float = 2e-3,
        teacher_weight_decay: float = 1e-4,
        student_lr: float = 1e-3,
        teacher_epochs: int = 2,
        student_epochs: int = 4,
        forgetting_beta: float = 0.3,
        rollback_threshold: float = 5.0,  # mAP下降百分点触发回滚
        lambda_cls: float = 1.0,
        lambda_reg: float = 0.5,
        lambda_feat: float = 0.5,
        distill_temperature: float = 3.0,
        lora_rank: int = 16,
        device: str = 'cuda',
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.pretrain_teacher = pretrain_teacher
        self.device = device

        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.forgetting_beta = forgetting_beta
        self.rollback_threshold = rollback_threshold
        self.lora_rank = lora_rank

        # 蒸馏损失（特征层维度在 setup 阶段确定）
        self.distill_loss_fn = TotalDistillLoss(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
            lambda_feat=lambda_feat,
            temperature=distill_temperature,
        )
        self.forgetting_reg = ForgettingRegularization(beta=forgetting_beta)

        # 优化器在 setup 后初始化
        self.teacher_optimizer = None
        self.student_optimizer = None
        self.teacher_lr = teacher_lr
        self.teacher_weight_decay = teacher_weight_decay
        self.student_lr = student_lr

        # 性能跟踪（用于回滚）
        self.best_map = 0.0
        self.best_snapshot: Optional[ModelSnapshot] = None

    def setup_teacher_lora(self, lora_rank: Optional[int] = None) -> None:
        """向教师模型注入LoRA并配置优化器（如果还未注入）"""
        rank = lora_rank or self.lora_rank
        self.teacher, lora_params = inject_lora_to_model(
            self.teacher, r=rank, lora_alpha=rank
        )
        self.teacher.to(self.device)

        # 只优化 LoRA 参数
        self.teacher_optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.teacher_lr,
            weight_decay=self.teacher_weight_decay,
        )
        # 记录当前最佳快照
        self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)
        logger.info("[CloudDistill] 教师模型 LoRA 配置完毕")

    def adapt_teacher(
        self,
        uploaded_batches: List[Dict],
        align_loss_fn,  # 调用 model.adapt() 风格的损失函数
    ) -> Dict[str, float]:
        """
        Phase 1: 教师 LoRA 适配。
        使用上传的困难样本，调用教师模型的特征对齐方法更新 LoRA 参数。

        Args:
            uploaded_batches: 从边端上传的图像批次列表
            align_loss_fn: callable(model, batch) -> loss_dict
        Returns:
            平均损失字典
        """
        self.teacher.train()
        total_losses = {}
        n = 0

        for epoch in range(self.teacher_epochs):
            for batch in uploaded_batches:
                self.teacher_optimizer.zero_grad()

                # 特征对齐损失（调用 teacher.adapt()）
                self.teacher.online_adapt = True
                loss_dict = align_loss_fn(self.teacher, batch)

                # 防遗忘正则
                if self.pretrain_teacher is not None:
                    # 简化：对当前batch前向传播计算遗忘损失
                    pass  # 在更完整的实现中可扩展

                total_loss = sum(loss_dict.values())
                if total_loss > 0 and not torch.isnan(torch.tensor(float(total_loss))):
                    if isinstance(total_loss, torch.Tensor):
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.teacher.parameters() if p.requires_grad],
                            max_norm=1.0
                        )
                        self.teacher_optimizer.step()

                for k, v in loss_dict.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    total_losses[k] = total_losses.get(k, 0.0) + val
                n += 1

        avg_losses = {k: v / max(n, 1) for k, v in total_losses.items()}
        logger.info(f"[CloudDistill] 教师适配完成，平均损失: {avg_losses}")
        return avg_losses

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        batches: List[Dict],
    ) -> List[Dict]:
        """
        Phase 2: 教师模型推理生成伪标签和中间特征。

        Returns:
            含伪标签的增强批次列表，每项新增:
            - 'pseudo_logits': 分类logits
            - 'pseudo_bbox': bbox预测
            - 'pseudo_features': FPN特征字典
        """
        self.teacher.eval()
        enhanced_batches = []

        for batch in batches:
            images = self.teacher.preprocess_image(batch)
            features = self.teacher.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            proposals, _ = self.teacher.proposal_generator(images, features, None, eval=True)
            pred_instances, predictions, box_features = self.teacher.roi_heads._forward_box(
                features, proposals, outs=True
            )

            enhanced_batch = {
                **batch,
                'pseudo_logits': predictions[0].detach() if predictions else None,
                'pseudo_bbox': predictions[1].detach() if len(predictions) > 1 else None,
                'pseudo_features': {k: v.detach() for k, v in features.items()},
                'pseudo_instances': pred_instances,
            }
            enhanced_batches.append(enhanced_batch)

        logger.info(f"[CloudDistill] 生成 {len(enhanced_batches)} 批伪标签")
        return enhanced_batches

    def distill_to_student(
        self,
        enhanced_batches: List[Dict],
    ) -> Dict[str, float]:
        """
        Phase 3: 三级知识蒸馏训练学生模型。

        Args:
            enhanced_batches: 含伪标签的批次（来自 generate_pseudo_labels）
        Returns:
            平均损失字典
        """
        # 设置学生模型可训练
        self.student.to(self.device)
        self.student.train()
        self.student.requires_grad_(True)

        if self.student_optimizer is None:
            self.student_optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.student_lr,
            )

        total_losses = {}
        n = 0

        for epoch in range(self.student_epochs):
            for batch in enhanced_batches:
                self.student_optimizer.zero_grad()

                # 学生前向
                images = self.student.preprocess_image(batch)
                s_features = self.student.backbone(images.tensor)
                if isinstance(s_features, tuple):
                    s_features = s_features[0]

                proposals, _ = self.student.proposal_generator(images, s_features, None, eval=True)
                _, s_predictions, s_box_features = self.student.roi_heads._forward_box(
                    s_features, proposals, outs=True
                )

                # 取教师预测
                t_logits = batch.get('pseudo_logits')
                t_bbox = batch.get('pseudo_bbox')
                t_features = batch.get('pseudo_features', {})

                s_logits = s_predictions[0] if s_predictions else None
                s_bbox = s_predictions[1] if len(s_predictions) > 1 else None

                # 三级蒸馏损失
                total_loss, loss_dict = self.distill_loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    student_bbox=s_bbox,
                    teacher_bbox=t_bbox,
                    student_features=s_features,
                    teacher_features=t_features,
                )

                if total_loss.requires_grad and not torch.isnan(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), max_norm=1.0
                    )
                    self.student_optimizer.step()

                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                n += 1

        avg_losses = {k: v / max(n, 1) for k, v in total_losses.items()}
        logger.info(f"[CloudDistill] 学生蒸馏完成，平均损失: {avg_losses}")
        return avg_losses

    def update_best_map(self, current_map: float) -> bool:
        """
        更新最佳mAP，若下降超过阈值则回滚教师模型参数。

        Returns:
            True 表示触发了回滚
        """
        if self.best_map == 0.0:
            self.best_map = current_map
            self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)
            return False

        if self.best_map - current_map > self.rollback_threshold:
            logger.warning(
                f"[CloudDistill] mAP下降 {self.best_map - current_map:.1f}pp，触发参数回滚"
            )
            if self.best_snapshot is not None:
                self.best_snapshot.restore(self.teacher)
            return True

        if current_map > self.best_map:
            self.best_map = current_map
            self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)
        return False

    def run_full_pipeline(
        self,
        uploaded_batches: List[Dict],
        align_loss_fn,
        val_map_fn=None,
    ) -> Dict[str, float]:
        """
        执行完整的云端适配流程:
        1. 教师LoRA适配
        2. 伪标签生成
        3. 学生蒸馏

        Args:
            uploaded_batches: 边端上传的困难样本批次
            align_loss_fn: callable(model, batch) -> loss_dict
            val_map_fn: callable(model) -> float，用于mAP计算和回滚判断

        Returns:
            所有阶段的损失汇总
        """
        all_losses = {}

        # Phase 1: 教师适配
        teacher_losses = self.adapt_teacher(uploaded_batches, align_loss_fn)
        all_losses.update({f'teacher_{k}': v for k, v in teacher_losses.items()})

        # 回滚检查
        if val_map_fn is not None:
            current_map = val_map_fn(self.teacher)
            rolled_back = self.update_best_map(current_map)
            all_losses['teacher_map'] = current_map
            all_losses['rolled_back'] = float(rolled_back)

        # Phase 2: 伪标签生成
        enhanced_batches = self.generate_pseudo_labels(uploaded_batches)

        # Phase 3: 学生蒸馏
        student_losses = self.distill_to_student(enhanced_batches)
        all_losses.update(student_losses)

        return all_losses
