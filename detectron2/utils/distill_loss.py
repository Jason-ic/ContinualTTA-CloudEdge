"""
知识蒸馏损失模块
Knowledge Distillation Losses for Cloud-Edge Collaborative TTA.

包含：
- QFLv2: 原有的 Quality Focal Loss v2（均保留）
- ClassificationDistillLoss: 分类KL散度蒸馏（温度缩放）
- RegressionDistillLoss: 回归Smooth L1蒸馏
- FeatureDistillLoss: FPN特征MSE蒸馏（含通道对齐适配器）
- ForgettingRegularization: 与预训练模型输出KL散度正则（防遗忘）
- TotalDistillLoss: 三级蒸馏总损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------- #
# 原有 QFLv2
# ---------------------------------------------------------------------------- #

def QFLv2(pred_sigmoid,          # (n, 80)
          teacher_sigmoid,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


# ---------------------------------------------------------------------------- #
# 分类蒸馏损失
# ---------------------------------------------------------------------------- #

class ClassificationDistillLoss(nn.Module):
    """
    分类知识蒸馏损失（KL散度 + 温度缩放）。

    L_KD^cls = (T^2 / N) * sum_i D_KL(p_i^T || p_i^S)
    其中 p^T = softmax(z^T / T), p^S = softmax(z^S / T), T=3

    论文 lambda_cls = 1.0
    """

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,  # (N, C)
        teacher_logits: torch.Tensor,  # (N, C)
    ) -> torch.Tensor:
        """
        Args:
            student_logits: 学生模型分类logits
            teacher_logits: 教师模型分类logits
        Returns:
            标量损失
        """
        if student_logits.shape[0] == 0:
            return student_logits.sum() * 0

        p_t = F.softmax(teacher_logits / self.T, dim=1).detach()
        log_p_s = F.log_softmax(student_logits / self.T, dim=1)

        # KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
        loss = F.kl_div(log_p_s, p_t, reduction='batchmean')
        # 乘以 T^2 恢复梯度量级
        return loss * (self.T ** 2)


# ---------------------------------------------------------------------------- #
# 回归蒸馏损失
# ---------------------------------------------------------------------------- #

class RegressionDistillLoss(nn.Module):
    """
    回归知识蒸馏损失（Smooth L1）。

    L_KD^reg = (1/N) * sum_i SmoothL1(b_i^S, b_i^T)
    论文 lambda_reg = 0.5
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta  # Smooth L1 折点

    def forward(
        self,
        student_bbox: torch.Tensor,  # (N, 4) 或 (N, C*4)
        teacher_bbox: torch.Tensor,  # (N, 4) 或 (N, C*4)
    ) -> torch.Tensor:
        if student_bbox.shape[0] == 0:
            return student_bbox.sum() * 0
        return F.smooth_l1_loss(student_bbox, teacher_bbox.detach(), beta=self.beta)


# ---------------------------------------------------------------------------- #
# 特征蒸馏损失（含通道对齐适配器）
# ---------------------------------------------------------------------------- #

class ChannelAdapter(nn.Module):
    """1x1 卷积通道对齐：将学生模型特征通道数映射到教师模型通道数"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.adapter.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class FeatureDistillLoss(nn.Module):
    """
    FPN特征蒸馏损失（逐层MSE）。

    L_KD^feat = (1/|F|) * sum_f ||phi_f(F_f^S) - F_f^T||^2
    phi_f 为通道对齐适配器（当 student/teacher 通道不同时）
    论文 lambda_feat = 0.5
    """

    def __init__(
        self,
        student_channels: Dict[str, int],   # {'p2': 256, ...}
        teacher_channels: Dict[str, int],   # {'p2': 256, ...}
    ):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for key in student_channels:
            if key in teacher_channels:
                s_c = student_channels[key]
                t_c = teacher_channels[key]
                if s_c != t_c:
                    self.adapters[key] = ChannelAdapter(s_c, t_c)

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],  # {'p2': tensor, ...}
        teacher_features: Dict[str, torch.Tensor],  # {'p2': tensor, ...}
    ) -> torch.Tensor:
        total_loss = None
        num_layers = 0

        for key in student_features:
            if key not in teacher_features:
                continue
            s_feat = student_features[key]
            t_feat = teacher_features[key].detach()

            # 通道对齐
            if key in self.adapters:
                s_feat = self.adapters[key](s_feat)

            # 空间尺寸对齐（若不同则插值）
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)

            # 按通道 L2 归一化后再算 MSE，避免特征幅值差异导致 loss 爆炸
            s_norm = F.normalize(s_feat, dim=1)
            t_norm = F.normalize(t_feat, dim=1)
            layer_loss = F.mse_loss(s_norm, t_norm)
            if total_loss is None:
                total_loss = layer_loss
            else:
                total_loss = total_loss + layer_loss
            num_layers += 1

        if total_loss is None or num_layers == 0:
            return torch.tensor(0.0)
        return total_loss / num_layers


# ---------------------------------------------------------------------------- #
# 遗忘正则化
# ---------------------------------------------------------------------------- #

class ForgettingRegularization(nn.Module):
    """
    防遗忘正则化：当前教师模型输出与预训练教师输出的KL散度。

    L_forget = beta * D_KL(p_large || p_pretrain)
    论文 beta ∈ [0.1, 0.5]
    """

    def __init__(self, beta: float = 0.3):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        current_logits: torch.Tensor,   # (N, C) 当前教师模型输出
        pretrain_logits: torch.Tensor,  # (N, C) 冻结预训练教师输出
    ) -> torch.Tensor:
        if current_logits.shape[0] == 0:
            return current_logits.sum() * 0

        p_pretrain = F.softmax(pretrain_logits.detach(), dim=1)
        log_p_current = F.log_softmax(current_logits, dim=1)

        kl = F.kl_div(log_p_current, p_pretrain, reduction='batchmean')
        return self.beta * kl


# ---------------------------------------------------------------------------- #
# 三级总蒸馏损失
# ---------------------------------------------------------------------------- #

class TotalDistillLoss(nn.Module):
    """
    云端三级知识蒸馏总损失。

    L_KD = lambda_cls * L_cls + lambda_reg * L_reg + lambda_feat * L_feat

    论文默认权重: lambda_cls=1.0, lambda_reg=0.5, lambda_feat=0.5
    温度: T=3
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_reg: float = 0.5,
        lambda_feat: float = 0.5,
        temperature: float = 3.0,
        student_channels: Optional[Dict[str, int]] = None,
        teacher_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_feat = lambda_feat

        self.cls_loss = ClassificationDistillLoss(temperature=temperature)
        self.reg_loss = RegressionDistillLoss()

        if student_channels is not None and teacher_channels is not None:
            self.feat_loss = FeatureDistillLoss(student_channels, teacher_channels)
        else:
            self.feat_loss = None

    def forward(
        self,
        student_logits: Optional[torch.Tensor],
        teacher_logits: Optional[torch.Tensor],
        student_bbox: Optional[torch.Tensor],
        teacher_bbox: Optional[torch.Tensor],
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        total = torch.tensor(0.0, requires_grad=True)

        if student_logits is not None and teacher_logits is not None:
            l_cls = self.cls_loss(student_logits, teacher_logits)
            total = total + self.lambda_cls * l_cls
            loss_dict['distill_cls'] = l_cls.item()

        if student_bbox is not None and teacher_bbox is not None:
            l_reg = self.reg_loss(student_bbox, teacher_bbox)
            total = total + self.lambda_reg * l_reg
            loss_dict['distill_reg'] = l_reg.item()

        if (self.feat_loss is not None
                and student_features is not None
                and teacher_features is not None):
            l_feat = self.feat_loss(student_features, teacher_features)
            total = total + self.lambda_feat * l_feat
            loss_dict['distill_feat'] = l_feat.item()

        loss_dict['distill_total'] = total.item()
        return total, loss_dict
