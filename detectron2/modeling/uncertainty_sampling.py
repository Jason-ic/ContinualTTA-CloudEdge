"""
边端不确定性采样模块
Edge-side uncertainty-based hard sample selection for cloud-edge collaborative TTA.

实现论文 Chapter 3 的核心算法：
- 预测熵不确定性 (Prediction Entropy)
- 特征KL散度不确定性 (Feature KL Divergence)
- 组合不确定性 (Combined Uncertainty)
- 动态阈值选择 (Dynamic Threshold)
- 类均衡采样上限 (Class-Balanced Upload Cap)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Tuple


class UncertaintySampler:
    """
    边端不确定性采样器

    对每帧图像计算组合不确定性得分，基于滑动窗口动态阈值决定是否上传至云端。
    上传样本按类别均衡约束，防止易类样本主导。

    论文 Eq: U(x) = max{ H(x)/log(C), D_KL(x)/D_KL_ref }
    其中 H(x) 为加权预测熵，D_KL(x) 为特征KL散度
    """

    def __init__(
        self,
        num_classes: int,
        source_feat_stats: Optional[Dict] = None,
        sliding_window_size: int = 100,
        percentile: float = 0.7,
        kl_ref: float = 1.0,
        s_min: float = 1024.0,  # 最小proposal面积 32*32
        max_upload_per_class: int = 10,
    ):
        """
        Args:
            num_classes: 类别数（不含背景）
            source_feat_stats: 源域特征统计 {'global': (mean, var), 'fg': [(mean, var)] * C}
            sliding_window_size: 滑动窗口大小 W
            percentile: 百分位阈值 p（选择 p 分位以上的样本上传）
            kl_ref: KL散度归一化参考值 D_KL_ref（训练集95百分位）
            s_min: 最小面积归一化常数
            max_upload_per_class: 每类最大上传数量上限基准
        """
        self.num_classes = num_classes
        self.source_feat_stats = source_feat_stats
        self.sliding_window_size = sliding_window_size
        self.percentile = percentile
        self.kl_ref = kl_ref
        self.s_min = s_min
        self.max_upload_per_class = max_upload_per_class

        # 滑动窗口存储不确定性历史
        self.uncertainty_window = deque(maxlen=sliding_window_size)
        # 各类别已上传计数（用于类均衡约束）
        self.class_upload_counts = np.zeros(num_classes, dtype=np.int32)
        # 总上传数
        self.total_uploaded = 0
        self.total_seen = 0

    def compute_prediction_entropy(
        self,
        class_scores: torch.Tensor,  # (N, C+1) softmax scores
        proposal_areas: Optional[torch.Tensor] = None,  # (N,) area in pixels
    ) -> float:
        """
        计算一张图的加权预测熵。

        H(x) = sum_i [ w_i * (-sum_c p_{i,c} * log(p_{i,c})) ] / N
        w_i = max_c(p_{i,c}) * min(1, S_i / S_min)

        Args:
            class_scores: RoI分类得分，softmax后，shape (N, C+1)
            proposal_areas: proposal面积（像素），shape (N,)
        Returns:
            归一化预测熵标量
        """
        if class_scores.shape[0] == 0:
            return 0.0

        # 预测熵 per proposal
        eps = 1e-8
        entropy = -(class_scores * (class_scores + eps).log()).sum(dim=1)  # (N,)

        # 置信度权重：最大类别概率
        max_conf, _ = class_scores.max(dim=1)  # (N,)

        # 面积权重
        if proposal_areas is not None:
            area_weight = torch.clamp(proposal_areas / self.s_min, max=1.0)
        else:
            area_weight = torch.ones_like(max_conf)

        weight = max_conf * area_weight  # (N,)
        weight_sum = weight.sum() + eps
        weighted_entropy = (weight * entropy).sum() / weight_sum

        # 归一化到 [0, 1]
        log_c = math.log(self.num_classes + 1)  # +1 for background
        return (weighted_entropy / log_c).item()

    def compute_feature_kl(
        self,
        target_feat_mean: torch.Tensor,  # (D,)
        target_feat_var: torch.Tensor,   # (D,)
    ) -> float:
        """
        计算目标域特征分布与源域分布之间的对角KL散度。
        D_KL(N(mu_T, diag(var_T)) || N(mu_S, diag(var_S)))

        Args:
            target_feat_mean: 当前批次特征均值
            target_feat_var: 当前批次特征方差
        Returns:
            KL散度标量
        """
        if self.source_feat_stats is None:
            return 0.0

        mu_s = self.source_feat_stats['global'][0].to(target_feat_mean.device)
        var_s = self.source_feat_stats['global'][1].to(target_feat_mean.device)

        # 对角高斯KL散度（闭式解）
        # KL(N1||N2) = 0.5 * sum[ log(var2/var1) + var1/var2 + (mu1-mu2)^2/var2 - 1 ]
        eps = 1e-8
        var_s = var_s.clamp(min=eps)
        var_t = target_feat_var.clamp(min=eps)

        kl = 0.5 * (
            (var_s / var_t).log()
            + var_t / var_s
            + (target_feat_mean - mu_s).pow(2) / var_s
            - 1
        ).sum().item()

        return max(kl, 0.0)

    def compute_combined_uncertainty(
        self,
        entropy_score: float,
        kl_score: float,
    ) -> float:
        """
        组合不确定性：U(x) = max{ H(x)/log(C), D_KL(x)/D_KL_ref }
        """
        normalized_entropy = entropy_score  # 已归一化
        normalized_kl = kl_score / (self.kl_ref + 1e-8)
        return max(normalized_entropy, normalized_kl)

    def _compute_dynamic_threshold(self) -> float:
        """基于滑动窗口历史计算动态百分位阈值"""
        if len(self.uncertainty_window) < 5:
            return 0.0  # 窗口不足时接受所有样本
        history = np.array(list(self.uncertainty_window))
        return float(np.percentile(history, self.percentile * 100))

    def _class_balanced_cap(self, pred_classes: Optional[List[int]]) -> Dict[int, int]:
        """
        计算各类别上传上限（逆频率加权）。
        N_c^max = ceil(N_batch / C * alpha_c)
        alpha_c = max(1, n_bar / (n_c + eps))

        Args:
            pred_classes: 当前图中预测到的类别列表
        Returns:
            {class_id: max_upload_count}
        """
        eps = 1.0
        n_bar = np.mean(self.class_upload_counts + eps)
        caps = {}
        for c in range(self.num_classes):
            alpha_c = max(1.0, n_bar / (self.class_upload_counts[c] + eps))
            caps[c] = math.ceil(self.max_upload_per_class * alpha_c)
        return caps

    def should_upload(
        self,
        class_scores: torch.Tensor,         # (N, C+1)
        feat_mean: Optional[torch.Tensor],   # (D,)
        feat_var: Optional[torch.Tensor],    # (D,)
        proposal_areas: Optional[torch.Tensor] = None,  # (N,)
        pred_classes: Optional[List[int]] = None,
    ) -> Tuple[bool, float]:
        """
        判断当前图像是否应上传至云端。

        Returns:
            (should_upload, uncertainty_score)
        """
        self.total_seen += 1

        # 1. 计算熵不确定性
        entropy_score = self.compute_prediction_entropy(class_scores, proposal_areas)

        # 2. 计算特征KL散度
        if feat_mean is not None and feat_var is not None:
            kl_score = self.compute_feature_kl(feat_mean, feat_var)
        else:
            kl_score = 0.0

        # 3. 组合不确定性
        uncertainty = self.compute_combined_uncertainty(entropy_score, kl_score)

        # 4. 更新滑动窗口
        self.uncertainty_window.append(uncertainty)

        # 5. 动态阈值
        threshold = self._compute_dynamic_threshold()

        # 6. 判断是否上传
        if uncertainty >= threshold:
            # 7. 类均衡约束
            if pred_classes is not None:
                caps = self._class_balanced_cap(pred_classes)
                # 检查主要预测类别是否超出上限
                from collections import Counter
                class_counts = Counter(pred_classes)
                upload_blocked = False
                for c, cnt in class_counts.items():
                    if 0 <= c < self.num_classes:
                        if self.class_upload_counts[c] >= caps[c]:
                            upload_blocked = True
                            break
                if upload_blocked:
                    return False, uncertainty

                # 更新类别计数
                for c, cnt in class_counts.items():
                    if 0 <= c < self.num_classes:
                        self.class_upload_counts[c] += 1

            self.total_uploaded += 1
            return True, uncertainty

        return False, uncertainty

    def get_upload_rate(self) -> float:
        """当前上传率"""
        if self.total_seen == 0:
            return 0.0
        return self.total_uploaded / self.total_seen

    def reset_stats(self):
        """重置统计（新序列开始时调用）"""
        self.class_upload_counts = np.zeros(self.num_classes, dtype=np.int32)
        self.total_uploaded = 0
        self.total_seen = 0
        self.uncertainty_window.clear()


def extract_image_features(model_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从FPN特征字典中提取图像级全局特征（GAP后的均值和方差）。
    取所有FPN层的平均。

    Args:
        model_features: FPN特征字典，如 {'p2': ..., 'p3': ..., ...}
    Returns:
        (feat_mean, feat_var): 全局特征统计，shape (D,)
    """
    gap_feats = []
    for key in sorted(model_features.keys()):
        feat = model_features[key]  # (1, C, H, W)
        gap = feat.mean(dim=[2, 3]).squeeze(0)  # (C,)
        gap_feats.append(gap)

    if not gap_feats:
        return None, None

    # 取所有层级concat
    all_feats = torch.cat(gap_feats, dim=0)  # (sum_C,)
    # 由于单张图无法估计方差，用标准差近似（多个层级提供统计）
    stacked = torch.stack(gap_feats, dim=0)  # (L, C)
    feat_mean = stacked.mean(dim=0)
    feat_var = stacked.var(dim=0) + 1e-8

    return feat_mean.detach(), feat_var.detach()
