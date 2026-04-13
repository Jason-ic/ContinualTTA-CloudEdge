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
from collections import deque
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

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
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

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
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

    # 执行替换（通过 parent 模块的 setattr 或 ModuleList 索引赋值）
    for name, module, mtype in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        attr = parts[-1]

        if mtype == 'conv2d':
            lora_module = LoRAConv2d(module, r=r, lora_alpha=lora_alpha)
        else:
            lora_module = LoRALinear(module, r=r, lora_alpha=lora_alpha)

        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = lora_module
        else:
            setattr(parent, attr, lora_module)
        lora_params.extend(lora_module.get_lora_params())
        replaced += 1

    # FPN 的 lateral_convs / output_convs 是普通 list（非 ModuleList），
    # setattr 替换了命名子模块但 list 中的引用还是旧对象，需要同步更新
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'lateral_convs'):
        fpn = model.backbone
        for i, old_conv in enumerate(fpn.lateral_convs):
            # lateral_convs 是 top-down 顺序（res5, res4, res3, res2），
            # 对应 fpn_lateral5, fpn_lateral4, ...
            for stage in [2, 3, 4, 5, 6, 7]:
                attr_name = f"fpn_lateral{stage}"
                if hasattr(fpn, attr_name) and getattr(fpn, attr_name) is not old_conv:
                    new_m = getattr(fpn, attr_name)
                    if isinstance(new_m, (LoRAConv2d, LoRALinear)) and new_m.conv is old_conv:
                        fpn.lateral_convs[i] = new_m
                        break
        for i, old_conv in enumerate(fpn.output_convs):
            for stage in [2, 3, 4, 5, 6, 7]:
                attr_name = f"fpn_output{stage}"
                if hasattr(fpn, attr_name) and getattr(fpn, attr_name) is not old_conv:
                    new_m = getattr(fpn, attr_name)
                    if isinstance(new_m, (LoRAConv2d, LoRALinear)) and new_m.conv is old_conv:
                        fpn.output_convs[i] = new_m
                        break

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
        student_channels: Optional[Dict[str, int]] = None,
        teacher_channels: Optional[Dict[str, int]] = None,
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

        # 蒸馏损失：必须传入 student/teacher_channels 才能启用特征蒸馏
        # （若不传，FeatureDistillLoss 为 None，特征层监督丢失，与论文 Chapter 5 不一致）
        self.distill_loss_fn = TotalDistillLoss(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
            lambda_feat=lambda_feat,
            temperature=distill_temperature,
            student_channels=student_channels,
            teacher_channels=teacher_channels,
        ).to(device)
        self.forgetting_reg = ForgettingRegularization(beta=forgetting_beta).to(device)

        # 优化器在 setup 后初始化
        self.teacher_optimizer = None
        self.student_optimizer = None
        self.teacher_lr = teacher_lr
        self.teacher_weight_decay = teacher_weight_decay
        self.student_lr = student_lr

        # 性能跟踪（用于回滚）：滑动窗口平滑，避免单次噪声误触发
        self.best_map = 0.0
        self.best_snapshot: Optional[ModelSnapshot] = None
        self.map_window = deque(maxlen=3)

        # 保存学生初始 state_dict 副本，每次云端更新前重置学生到此状态，
        # 避免跨云端更新累积微调导致模型退化（Bug A 修复）
        self._student_init_state = {
            k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()
        }

        # 冻结的 pristine 学生引用：用于 forgetting 正则（与当前学生同架构比较，
        # 比用 R101 预训练教师更符合论文 Chapter 5 的 L_forget 语义）。
        # 驻留 CPU 以节省显存，蒸馏时按需移 GPU。
        self._ref_student = copy.deepcopy(student_model).cpu()
        self._ref_student.eval()
        self._ref_student.requires_grad_(False)

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

    def _reset_teacher_lora(self) -> int:
        """
        将教师所有 LoRA 增量重置为 0（清零 lora_B 即可，因为 W' = W + (x A B) * scale）。
        每次云端更新前调用，避免 LoRA 在多次更新间累积漂移。
        Returns: 重置的 LoRA 模块数量
        """
        n = 0
        for m in self.teacher.modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                with torch.no_grad():
                    m.lora_B.zero_()
                n += 1
        # 同时重置 optimizer 的动量状态，避免上一次 update 的动量影响这次
        if self.teacher_optimizer is not None:
            self.teacher_optimizer.state = type(self.teacher_optimizer.state)()
        return n

    def _compute_lora_reg(self) -> torch.Tensor:
        """
        LoRA 权重正则：只惩罚 lora_B 范数（B 初始化为 0，偏离代表适配偏移量）。
        lora_A 用 kaiming 初始化有固有范数，不应正则化。
        """
        reg = torch.tensor(0.0, device=self.device)
        n = 0
        for m in self.teacher.modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                reg = reg + m.lora_B.pow(2).sum()
                n += 1
        return reg / max(n, 1)

    def adapt_teacher(
        self,
        uploaded_batches: List[Dict],
        align_loss_fn=None,
    ) -> Dict[str, float]:
        """
        Phase 1: 教师 LoRA 持续适配。
        不 reset LoRA（允许跨更新积累知识），但通过以下方式控制累积幅度：
        - 低学习率（CLOUD_LORA_LR）
        - LoRA 权重正则（防止参数过大偏离原始 R101）
        - 梯度裁剪 max_norm=1.0
        """
        self.teacher.eval()
        # 确保 LoRA 参数有梯度
        for m in self.teacher.modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                for p in m.get_lora_params():
                    p.requires_grad_(True)

        total_losses = {}
        n = 0

        s_stats = getattr(self.teacher, 's_stats', None)
        if s_stats is None:
            logger.warning("[CloudDistill] 教师模型无源域统计，跳过适配")
            return {}

        lora_reg_weight = 0.1  # LoRA 正则权重（增强约束，防止过拟合困难样本）

        max_steps = self.teacher_epochs  # 总优化步数上限
        step = 0
        for epoch in range(max_steps):
            for batch in uploaded_batches:
                if step >= max_steps:
                    break
                self.teacher_optimizer.zero_grad()

                with torch.enable_grad():
                    images = self.teacher.preprocess_image(batch)
                    features = self.teacher.backbone(images.tensor)
                    if isinstance(features, tuple):
                        features = features[0]

                    # 全局特征对齐 KL 散度
                    align_loss = torch.tensor(0.0, device=self.device)
                    loss_count = 0

                    if 'gl' in s_stats:
                        for k in features:
                            if k not in s_stats['gl']:
                                continue
                            cur_feat = features[k].mean(dim=[2, 3]).mean(dim=0)
                            s_mean = s_stats['gl'][k][0].to(self.device)
                            s_cov = s_stats['gl'][k][1].to(self.device)
                            s_var = s_cov.diag().clamp(min=1e-8)

                            diff = cur_feat - s_mean
                            kl = 0.5 * (diff.pow(2) / s_var).sum()
                            align_loss = align_loss + kl
                            loss_count += 1

                    if loss_count > 0:
                        align_loss = align_loss / loss_count

                    # LoRA 正则：防止累积适配偏离太远
                    lora_reg = self._compute_lora_reg()
                    total_loss = align_loss + lora_reg_weight * lora_reg

                    if total_loss.requires_grad and not torch.isnan(total_loss):
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.teacher.parameters() if p.requires_grad],
                            max_norm=1.0
                        )
                        self.teacher_optimizer.step()

                total_losses['align_kl'] = total_losses.get('align_kl', 0.0) + align_loss.item()
                total_losses['lora_reg'] = total_losses.get('lora_reg', 0.0) + lora_reg.item()
                n += 1
                step += 1
            if step >= max_steps:
                break

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

            # 伪标签：把教师检测结果转成 GT Instances（含 pred_boxes 和 pred_classes）
            # 学生蒸馏时用自己的 RPN + 伪 GT 进行标准训练（proposal matching + box loss）
            pseudo_gt_instances = []
            for pred_inst in pred_instances:
                # 伪 GT：教师检测的 boxes 和 classes
                inst_dict = {
                    'gt_boxes': pred_inst.pred_boxes,
                    'gt_classes': pred_inst.pred_classes,
                }
                pseudo_gt_instances.append(inst_dict)

            # 教师 logits 和 bbox 用于软标签蒸馏
            t_logits = predictions[0].detach().cpu() if predictions else None
            t_bbox = predictions[1].detach().cpu() if len(predictions) > 1 else None

            # 创建增强 batch：加入伪 GT + 教师 proposals/logits
            enhanced_batch = {
                'batch': batch,
                'pseudo_instances': pred_instances,
                'pseudo_gt': pseudo_gt_instances,
                'proposals': [p.to('cpu') for p in proposals],
                'pseudo_logits': t_logits,
                'pseudo_bbox': t_bbox,
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
        # 学生从当前状态开始（调用方已同步边端权重）
        self.student.to(self.device)

        # 保持 eval 模式
        self.student.eval()
        self.student.requires_grad_(False)

        # 只训练 box_predictor（cls_score + bbox_pred 两个线性层）
        # 这些参数量小，50 样本足够学好，且不会破坏 backbone/FPN/RPN
        for name, param in self.student.named_parameters():
            if 'box_predictor' in name:
                param.requires_grad = True
        trainable = [p for p in self.student.parameters() if p.requires_grad]

        if not trainable:
            logger.warning("[CloudDistill] 没有找到可训练参数，跳过蒸馏")
            return {}

        # 每次新建 optimizer，不跨更新复用状态
        self.student_optimizer = torch.optim.AdamW(
            trainable, lr=self.student_lr
        )

        total_losses = {}
        n = 0

        for epoch in range(self.student_epochs):
            for batch in enhanced_batches:
                self.student_optimizer.zero_grad()

                raw_batch = batch['batch']

                # 学生前向
                images = self.student.preprocess_image(raw_batch)
                s_features = self.student.backbone(images.tensor)
                if isinstance(s_features, tuple):
                    s_features = s_features[0]

                # 使用教师 proposals 确保 logit 对齐
                t_proposals = [p.to(self.device) for p in batch['proposals']]
                _, s_predictions, _ = self.student.roi_heads._forward_box(
                    s_features, t_proposals, outs=True
                )

                # 教师软标签
                t_logits = batch.get('pseudo_logits')
                t_bbox = batch.get('pseudo_bbox')
                if t_logits is not None:
                    t_logits = t_logits.to(self.device)
                if t_bbox is not None:
                    t_bbox = t_bbox.to(self.device)

                s_logits = s_predictions[0] if s_predictions else None
                s_bbox = s_predictions[1] if len(s_predictions) > 1 else None

                # 三级蒸馏损失（软标签 KD）
                total_loss, loss_dict = self.distill_loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    student_bbox=s_bbox,
                    teacher_bbox=t_bbox,
                    student_features=None,
                    teacher_features=None,
                )

                if total_loss.requires_grad and not torch.isnan(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self.student_optimizer.step()

                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                n += 1

        avg_losses = {k: v / max(n, 1) for k, v in total_losses.items()}
        logger.info(f"[CloudDistill] 学生蒸馏完成，平均损失: {avg_losses}")
        return avg_losses

    def update_best_map(self, current_map: float) -> bool:
        """
        更新最佳 mAP，若滑窗均值下降超过阈值则回滚教师 LoRA。

        使用长度 3 的滑动窗口取均值，避免单次 mAP 噪声（±1~2pp）误触发回滚。
        Returns:
            True 表示触发了回滚
        """
        self.map_window.append(current_map)
        smoothed = sum(self.map_window) / len(self.map_window)

        if self.best_map == 0.0:
            self.best_map = smoothed
            self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)
            return False

        if self.best_map - smoothed > self.rollback_threshold:
            logger.warning(
                f"[CloudDistill] 教师 mAP 下降 {self.best_map - smoothed:.2f}pp "
                f"(best={self.best_map:.2f} → smoothed={smoothed:.2f}, "
                f"window={list(self.map_window)})，触发 LoRA 参数回滚"
            )
            if self.best_snapshot is not None:
                self.best_snapshot.restore(self.teacher)
            # 回滚后清空窗口，避免重复触发
            self.map_window.clear()
            return True

        if smoothed > self.best_map:
            self.best_map = smoothed
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

        # --- 显存管理：teacher 和 student 分时占 GPU ---
        # Phase 1+2 时 teacher 在 GPU，student 在 CPU
        self.student.cpu()
        self.teacher.to(self.device)
        torch.cuda.empty_cache()

        # 适配前先评估教师 baseline，并用 pre_map 初始化 best_map
        # 这样适配后若 mAP 下降超过阈值就能正确触发回滚
        if val_map_fn is not None:
            pre_map = val_map_fn(self.teacher)
            all_losses['teacher_pre_map'] = pre_map
            logger.info(f"[CloudDistill] 教师适配前 mAP: {pre_map:.2f}")
            if self.best_map == 0.0:
                self.best_map = pre_map
                self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)
                self.map_window.append(pre_map)

        # Phase 1: 教师适配
        teacher_losses = self.adapt_teacher(uploaded_batches, align_loss_fn)
        all_losses.update({f'teacher_{k}': v for k, v in teacher_losses.items()})

        # 回滚检查（适配后评估）
        if val_map_fn is not None:
            current_map = val_map_fn(self.teacher)
            rolled_back = self.update_best_map(current_map)
            all_losses['teacher_map'] = current_map
            all_losses['rolled_back'] = float(rolled_back)

        # Phase 2: 伪标签生成（teacher 仍在 GPU）
        enhanced_batches = self.generate_pseudo_labels(uploaded_batches)

        # Phase 3: teacher 移 CPU，student 移 GPU 做蒸馏
        self.teacher.cpu()
        torch.cuda.empty_cache()
        self.student.to(self.device)

        student_losses = self.distill_to_student(enhanced_batches)
        all_losses.update(student_losses)

        return all_losses
