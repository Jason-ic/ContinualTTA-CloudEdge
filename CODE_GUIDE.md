# 代码讲解与演示指南

本文档用于论文答辩/汇报时的代码讲解与现场演示。

---

## 一、所使用的代码框架

### 基础框架：Detectron2 (Facebook AI Research)

本项目基于 **Detectron2** 框架构建，而非 YOLO 或 MMDetection。

**检测器架构：**
- 边端模型：Faster R-CNN + ResNet-50-FPN（参数量约 25M）
- 云端教师：Faster R-CNN + ResNet-101-FPN（参数量约 45M）

**项目核心改动：** 不修改检测器本身，而是在其外围添加云边协同适配层。

### 关键文件总览

```
ContinualTTA-CloudEdge/
├── configs/TTA/
│   └── CloudEdge_COCO_R101_R50.yaml        # 主配置文件（80行）
├── tools/
│   ├── cloud_edge_adapt.py                  # 云边协同主入口（536行）
│   └── train_net.py                         # 标准训练/评估入口（203行）
├── detectron2/modeling/
│   ├── meta_arch/rcnn.py                    # 扩展的 GeneralizedRCNN（669行）
│   ├── cloud_distillation.py                # 云端蒸馏训练器（679行）
│   ├── uncertainty_sampling.py              # 不确定性采样器（271行）
│   └── configure_adaptation_model.py        # 模型配置构建器（292行）
└── detectron2/utils/
    ├── distill_loss.py                      # 蒸馏损失函数
    └── param_injection.py                   # 双缓冲参数注入器
```

---

## 二、创新点在代码中的实现位置（详细代码解读）

---

### 创新点 1：基于不确定性的样本筛选（论文第3章）

**文件：** `detectron2/modeling/uncertainty_sampling.py` (271行)

**核心类：** `UncertaintySampler`，在边端每帧推理后被调用，决定该帧是否上传云端。

#### 1.1 预测熵计算 -- 对应论文公式 (3-1)(3-2)(3-3)

```python
# uncertainty_sampling.py 第 68-107 行
def compute_prediction_entropy(self, class_scores, proposal_areas):
```

**做了什么：** 接收 RoI Head 输出的分类得分 `class_scores`（shape: N个候选框 x C+1个类别），计算整张图的加权平均预测熵。

**实现细节：**
- 第 90 行：逐候选框计算信息熵 `H(p_i) = -sum(p * log(p))`，对应论文公式 (3-1)
- 第 93 行：权重 = 最大类别概率（置信度高的框更可靠）
- 第 97 行：面积权重 = `min(1, S_i/S_min)`，S_min=32x32=1024，抑制小框噪声，对应论文公式 (3-3)
- 第 101-103 行：加权平均 `H(x) = sum(w_i * H_i) / sum(w_i)`，对应论文公式 (3-2)
- 第 107 行：除以 `log(C)` 归一化到 [0,1]

#### 1.2 特征 KL 散度 -- 对应论文公式 (3-4)(3-7)

```python
# uncertainty_sampling.py 第 109-142 行
def compute_feature_kl(self, target_feat_dict):
```

**做了什么：** 计算当前帧 FPN 各层特征与源域统计量之间的 KL 散度，衡量域偏移程度。

**实现细节：**
- 第 124 行：从预存的源域统计量 `self.source_feat_stats['gl']` 中取出各 FPN 层的均值 `mu_s` 和协方差 `cov_s`
- 第 135 行：对角化简化 `var_s = cov_s.diag()`，将复杂度从 O(k^3) 降到 O(k)，对应论文公式 (3-7)
- 第 138 行：KL 散度简化为 `0.5 * sum((diff^2) / var_s)`，对应论文公式 (3-4) 的对角形式
- 第 142 行：多层取平均

#### 1.3 组合不确定性 -- 对应论文公式 (3-8)

```python
# uncertainty_sampling.py 第 144-154 行
def compute_combined_uncertainty(self, entropy_score, kl_score):
    normalized_entropy = entropy_score          # 已归一化到 [0,1]
    normalized_kl = kl_score / (self.kl_ref)    # kl_ref 为源域 95 分位参考值
    return max(normalized_entropy, normalized_kl)
```

**做了什么：** 取两种不确定性的归一化最大值，任一指标高就标记为困难样本。对应论文公式 (3-8)：`U(x) = max{H(x)/logC, D_KL(x)/D_KL_ref}`

#### 1.4 动态阈值 -- 对应论文公式 (3-10)

```python
# uncertainty_sampling.py 第 156-161 行
def _compute_dynamic_threshold(self):
    history = np.array(list(self.uncertainty_window))  # 滑动窗口 W=100
    return float(np.percentile(history, self.percentile * 100))  # p=0.85
```

**做了什么：** 基于最近 100 帧的不确定性历史，取第 85 分位数作为动态阈值。环境变化时阈值自动调整。对应论文公式 (3-10)：`tau_t = Percentile(W, p)`

#### 1.5 类别平衡采样 -- 对应论文公式 (3-12)(3-13)

```python
# uncertainty_sampling.py 第 163-179 行
def _class_balanced_cap(self, pred_classes):
    n_bar = np.mean(self.class_upload_counts + eps)    # 所有类平均上传数
    alpha_c = max(1.0, n_bar / (n_c + eps))            # 逆频率权重
    caps[c] = ceil(max_upload_per_class * alpha_c)      # 稀有类上限更大
```

**做了什么：** 计算每类的上传配额上限。出现少的类 `alpha_c` 更大，配额更多，防止高频类主导。对应论文公式 (3-12)：`N_c^max = ceil(N_batch/C * alpha_c)` 和公式 (3-13)：`alpha_c = max(1, n_bar/(n_c+eps))`

#### 1.6 完整决策流程 -- 对应论文算法 1

```python
# uncertainty_sampling.py 第 182-241 行
def should_upload(self, class_scores, feat_dict, proposal_areas, pred_classes):
```

**完整链路：**
1. 第 198 行：计算预测熵
2. 第 202 行：计算特征 KL 散度
3. 第 207 行：组合不确定性 U(x)
4. 第 210 行：加入滑动窗口
5. 第 213 行：计算动态阈值
6. 第 216 行：`U(x) >= threshold` 则进入类别平衡检查
7. 第 226-232 行：只要有任一类别未超配额就上传，全部超配额才阻断

---

### 创新点 2：多粒度特征对齐（论文第4章）

**文件：** `detectron2/modeling/meta_arch/rcnn.py` 中的 `adapt()` 方法 (第 283-387 行)

**调用入口：** `forward()` 方法根据 `self.online_adapt` 标志分发：
- `online_adapt=True` -> 调用 `adapt()`（边端在线适配模式）
- `online_adapt=False` -> 调用 `inference()`（标准推理模式）

#### 2.1 特征提取与 RoI 前向

```python
# rcnn.py 第 285-303 行
images = self.preprocess_image(batched_inputs)
features = self.backbone(images.tensor)              # ResNet-50 + FPN -> 多尺度特征 {p2,p3,p4,p5,p6}
proposals, _ = self.proposal_generator(images, features, None, eval=True)  # RPN 生成候选框
pred_instances, predictions, box_features = self.roi_heads._forward_box(features, proposals, outs=True)
# predictions[0] = 分类 logits (N, 81), predictions[1] = bbox 回归
# box_features = RoI Pooling 后的特征向量 (N, 1024)
```

#### 2.2 区域级类别感知对齐 -- 对应论文公式 (4-7)(4-8)(4-9)

```python
# rcnn.py 第 307-352 行  (self.fg_align is not None 分支)
```

**逐行解读：**

```python
# 第 308-310 行：从 RoI Head 的分类 logits 中获取前景/背景得分
_scores = nn.Softmax(dim=1)(predictions[0])       # (N, 81) softmax
bg_scores = _scores[:, -1]                         # 背景概率
fg_scores, fg_preds = _scores[:, :-1].max(dim=1)  # 前景最大概率和对应类别

# 第 312 行：背景过滤 -- 对应论文 tau_bg 阈值
valid = fg_scores >= self.th_bg    # 默认 0.5，只保留高置信前景框

# 第 315-316 行：低置信框标记为背景
fg_preds[~valid] = self.num_classes   # 标记为背景类
fg_scores[~valid] = bg_scores[~valid]
```

```python
# 第 319-349 行：逐类别计算加权 KL 散度
for _k in fg_preds[fg_preds != self.num_classes].unique():
    k = _k.item()
    cur_feats = box_features[fg_preds == k]          # 该类别所有 RoI 特征

    # EMA 统计量更新 -- 对应论文公式 (4-3)(4-4)
    self.ema_n[k] += cur_feats.shape[0]
    diff = cur_feats - self.t_stats["fg"][k][0]      # 与目标域均值的偏差
    delta = (1/self.ema_gamma) * diff.sum(dim=0)     # EMA 增量
    cur_target_mean = self.t_stats["fg"][k][0] + delta  # 更新目标域均值

    # 构建多元高斯分布 -- 对应论文公式 (4-7)(4-8)
    t_dist = MultivariateNormal(cur_target_mean, s_stats_cov + template_cov)  # 目标域分布
    s_dist = MultivariateNormal(s_stats_mean, s_stats_cov + template_cov)     # 源域分布

    # 对称 KL 散度 -- 对应论文公式 (4-9)
    cur_loss = (kl_divergence(s_dist, t_dist) + kl_divergence(t_dist, s_dist)) / 2

    # 第 337-341 行：频率加权（可选）-- 对应论文公式 (4-5) 类别权重 w_c
    if self.freq_weight:
        class_weight = np.log(max_count / self.ema_n[k])  # 逆频率对数权重
        cur_loss = (class_weight + 0.01)**2 * cur_loss     # 稀有类权重更大
```

#### 2.3 图像级全局对齐 -- 对应论文公式 (4-1)(4-2)

```python
# rcnn.py 第 354-382 行  (self.gl_align == "KL" 分支)
for k in features.keys():                        # 遍历 FPN 各层 {p2,p3,p4,p5,p6}
    cur_feats = features[k].mean(dim=[2,3])       # GAP: (B,C,H,W) -> (B,C)，对应论文 GAP 操作

    # EMA 更新目标域统计量 -- 对应论文公式 (4-3)
    diff = cur_feats - self.t_stats["gl"][k][0]
    delta = (1/self.ema_gamma) * diff.sum(dim=0)
    cur_target_mean = self.t_stats["gl"][k][0] + delta

    # 构建高斯分布并计算对称 KL 散度 -- 对应论文公式 (4-1)
    t_dist = MultivariateNormal(cur_target_mean, s_cov + template_cov)
    s_dist = MultivariateNormal(s_mean, s_cov + template_cov)
    loss_gl = (kl_divergence(s_dist, t_dist) + kl_divergence(t_dist, s_dist)) / 2
```

#### 2.4 总损失合并

```python
# 第 352 行：区域级损失
adapt_loss["fg_align"] = self.alpha_fg * loss_fg_align    # lambda_roi

# 第 382 行：全局损失
adapt_loss["global_align"] = self.alpha_gl * loss_gl_align  # lambda_img

# 总损失 = lambda_img * L_img + lambda_roi * L_roi，对应论文公式 (4-10)
```

---

### 创新点 3：云端双模型 LoRA 微调 + 知识蒸馏（论文第5章）

**文件：** `detectron2/modeling/cloud_distillation.py` (679行)

#### 3.1 LoRA 模块实现 -- 对应论文公式 (5-1)

```python
# cloud_distillation.py 第 36-66 行
class LoRALinear(nn.Module):
    """W' = W + B*A * (alpha/r)"""

    def __init__(self, linear, r=16, lora_alpha=16):
        self.lora_A = nn.Parameter(torch.empty(in_features, r))     # A: (d, r)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))    # B: (r, d)
        self.scaling = lora_alpha / r
        nn.init.kaiming_uniform_(self.lora_A)   # A 用 kaiming 初始化
        nn.init.zeros_(self.lora_B)             # B 初始化为 0（初始时 LoRA 增量为 0）

    def forward(self, x):
        base_out = self.linear(x)                            # 原始权重前向（冻结）
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling  # LoRA 低秩增量
        return base_out + lora_out    # W' = W + BA * (alpha/r)，对应论文公式 (5-1)
```

**为什么 B 初始化为 0：** 确保 LoRA 注入时模型输出不变（增量为 0），然后通过训练逐步学习域适应。

#### 3.2 LoRA 注入位置 -- 对应论文第5.1.3节

```python
# cloud_distillation.py 第 131-143 行
keywords = [
    'bottom_up.stages.2',   # ResNet stage3 的 1x1 卷积
    'bottom_up.stages.3',   # ResNet stage4 的 1x1 卷积
    'fpn_lateral',          # FPN 横向连接卷积
    'fpn_output',           # FPN 输出卷积
    'rpn_head.conv',        # RPN 卷积头
    'box_head.fc',          # R-CNN Head 全连接层
]
```

**注入策略：** 只在深层（对域偏移敏感的层）注入 LoRA，浅层保持冻结。总可训练参数约 0.5M，是全参数 45M 的约 1%。

#### 3.3 教师 LoRA 自适应 (Phase 1) -- 对应论文第5.2.2节

```python
# cloud_distillation.py 第 350-435 行
def adapt_teacher(self, uploaded_batches, align_loss_fn):
```

**逐步解读：**
```python
# 第 387-389 行：教师前向提取特征
images = self.teacher.preprocess_image(batch)
features = self.teacher.backbone(images.tensor)   # R101 + FPN

# 第 394-408 行：计算特征对齐 KL 散度（与论文第4章一致）
for k in features:
    cur_feat = features[k].mean(dim=[2,3]).mean(dim=0)  # GAP
    diff = cur_feat - s_mean
    kl = 0.5 * (diff.pow(2) / s_var).sum()             # 对角 KL
    align_loss += kl

# 第 415-416 行：LoRA 正则防止过拟合
lora_reg = self._compute_lora_reg()                     # sum(B^2) / n
total_loss = align_loss + 0.1 * lora_reg

# 第 418-424 行：只更新 LoRA 参数
total_loss.backward()
clip_grad_norm_(lora_params, max_norm=1.0)              # 梯度裁剪
self.teacher_optimizer.step()                            # AdamW 更新
```

#### 3.4 伪标签生成 (Phase 2) -- 对应论文第5.2.1节

```python
# cloud_distillation.py 第 437-493 行
def generate_pseudo_labels(self, batches):
```

**做了什么：** 适配后的教师模型对困难样本做推理，保存：
- `proposals`：教师 RPN 生成的候选框（后续学生用同一组 proposals 对齐 logits）
- `pseudo_logits`：教师分类 logits（软标签，用于 KL 蒸馏）
- `pseudo_bbox`：教师 bbox 回归预测（用于 Smooth L1 蒸馏）

#### 3.5 学生蒸馏 (Phase 3) -- 对应论文第5.2.3节

```python
# cloud_distillation.py 第 495-583 行
def distill_to_student(self, enhanced_batches):
```

**逐步解读：**
```python
# 第 514-518 行：只训练 box_predictor（cls_score + bbox_pred 两个线性层）
for name, param in self.student.named_parameters():
    if 'box_predictor' in name:
        param.requires_grad = True

# 第 546-548 行：用教师的 proposals 做学生前向（确保 logit 空间对齐）
t_proposals = [p.to(self.device) for p in batch['proposals']]
_, s_predictions, _ = self.student.roi_heads._forward_box(s_features, t_proposals)
# 关键：学生和教师在同一组 proposals 上产出 logits，这样 KL 散度才有意义

# 第 563-570 行：三级蒸馏损失
total_loss, loss_dict = self.distill_loss_fn(
    student_logits=s_logits,    teacher_logits=t_logits,     # -> KL 散度
    student_bbox=s_bbox,        teacher_bbox=t_bbox,         # -> Smooth L1
    student_features=None,      teacher_features=None,       # -> MSE（可选）
)
```

#### 3.6 三级蒸馏损失函数 -- 对应论文公式 (5-2)(5-3)(5-4)(5-5)

**文件：** `detectron2/utils/distill_loss.py`

**分类蒸馏 (第 53-88 行)** -- 对应论文公式 (5-2)(5-3)：
```python
class ClassificationDistillLoss(nn.Module):
    def forward(self, student_logits, teacher_logits):
        p_t = F.softmax(teacher_logits / self.T, dim=1)    # 教师软标签，T=3
        log_p_s = F.log_softmax(student_logits / self.T, dim=1)  # 学生对数概率
        loss = F.kl_div(log_p_s, p_t, reduction='batchmean')
        return loss * (self.T ** 2)    # 乘 T^2 恢复梯度量级（Hinton 2015）
```

**回归蒸馏 (第 95-114 行)** -- 对应论文公式 (5-4)：
```python
class RegressionDistillLoss(nn.Module):
    def forward(self, student_bbox, teacher_bbox):
        return F.smooth_l1_loss(student_bbox, teacher_bbox.detach())
```

**特征蒸馏 (第 133-190 行)** -- 对应论文公式 (5-5)：
```python
class FeatureDistillLoss(nn.Module):
    def forward(self, student_features, teacher_features):
        for key in student_features:
            s_feat = self.adapters[key](s_feat)  # 1x1 卷积通道对齐 phi_f
            s_norm = F.normalize(s_feat, dim=1)  # L2 归一化防止 loss 爆炸
            t_norm = F.normalize(t_feat, dim=1)
            layer_loss = F.mse_loss(s_norm, t_norm)  # 逐层 MSE
        return total_loss / num_layers
```

**总损失 (第 228-294 行)** -- 对应论文公式 (5-6)：
```python
# L_KD = lambda_cls * L_cls + lambda_reg * L_reg + lambda_feat * L_feat
# 默认权重: lambda_cls=1.0, lambda_reg=0.5, lambda_feat=0.5
total = lambda_cls * l_cls + lambda_reg * l_reg + lambda_feat * l_feat
```

#### 3.7 回滚机制 -- 对应论文第5.2.4节

```python
# cloud_distillation.py 第 585-616 行
def update_best_map(self, current_map):
    self.map_window.append(current_map)                       # 滑动窗口 maxlen=3
    smoothed = sum(self.map_window) / len(self.map_window)    # 窗口均值

    if self.best_map - smoothed > self.rollback_threshold:    # 下降超过 5pp
        self.best_snapshot.restore(self.teacher)              # 回滚 LoRA 参数
        return True

    if smoothed > self.best_map:
        self.best_map = smoothed
        self.best_snapshot = ModelSnapshot(self.teacher, only_lora=True)  # 保存最优快照
```

**做了什么：** 每次云端更新后评估教师 mAP，若相比历史最优下降超过 5 个百分点，自动回滚 LoRA 参数到上一个最优状态，防止灾难性遗忘。

#### 3.8 防遗忘正则 -- 对应论文公式 (5-7)

```python
# distill_loss.py 第 197-221 行
class ForgettingRegularization(nn.Module):
    def forward(self, current_logits, pretrain_logits):
        p_pretrain = F.softmax(pretrain_logits.detach(), dim=1)   # 冻结预训练输出
        log_p_current = F.log_softmax(current_logits, dim=1)      # 当前模型输出
        kl = F.kl_div(log_p_current, p_pretrain, reduction='batchmean')
        return self.beta * kl    # beta=0.3，对应论文公式 (5-7)
```

---

### 创新点 4：双缓冲参数注入（论文第5章第3节）

**文件：** `detectron2/utils/param_injection.py` (277行)

#### 4.1 FP16 参数导出 -- 对应论文第5.3.1节

```python
# param_injection.py 第 36-87 行
def export_student_fp16(model, save_path):
    # 排除 BN 运行统计（边端有自己的统计）和 adapter 参数（边端适配状态）
    skip_suffixes = ('running_mean', 'running_var', 'num_batches_tracked')
    skip_keywords = ('adapter',)

    for key, val in model.state_dict().items():
        if any(key.endswith(s) for s in skip_suffixes): continue
        fp16_state[key] = val.detach().half().cpu()    # FP32 -> FP16 压缩

    torch.save(fp16_state, tmp_path)                   # 保存约 50MB
    md5 = _compute_md5(save_path)                      # MD5 校验码
```

**为什么排除 BN 统计：** 边端模型在线适配过程中 BN 的 running_mean/var 已经跟踪了目标域分布，云端学生的 BN 统计是在困难样本子集上的，注入会覆盖边端更准确的统计。

#### 4.2 内存映射加载 -- 对应论文第5.3.3节

```python
# param_injection.py 第 102-141 行
def mmap_load_state_dict(path, verify_md5=True):
    # MD5 验证：防止传输损坏
    if actual_md5 != expected_md5:
        raise RuntimeError("MD5 校验失败")

    # 内存映射加载：比标准 torch.load 快约 30%
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buffer = io.BytesIO(mm.read())
    state_dict = torch.load(buffer, map_location='cpu')
```

#### 4.3 双缓冲注入器 -- 对应论文第5.3.2节

```python
# param_injection.py 第 148-232 行
class DualBufferInjector:
```

**核心方法 `try_swap()`（第 199-232 行）：**
```python
def try_swap(self):
    # 原子取出新参数（线程安全）
    with self._lock:
        next_state = self._next_state
        self._next_ready = False

    # 只注入 box_predictor 参数（cls_score + bbox_pred 两个线性层）
    # 不替换 backbone/FPN/RPN，避免破坏边端适配状态
    for key, cloud_val in next_state.items():
        if 'box_predictor' not in key: continue
        edge_val = current_state[key]
        # EMA 混合：new = (1-alpha)*edge + alpha*cloud, alpha=0.1
        current_state[key] = (1 - alpha) * edge_val + alpha * cloud_val
    self.model.load_state_dict(current_state)
```

**EMA 混合的意义：** 不直接替换，而是以 10% 的比例混入云端参数。这样保留了边端在线适配积累的 90% 状态，只引入少量云端增益，避免注入冲击。

**异步加载（第 174-197 行）：**
```python
def prepare_next(self, param_path):
    # 后台线程加载参数，不阻塞主线程推理
    self._loading_thread = threading.Thread(target=_load_worker, daemon=True)
    self._loading_thread.start()
```

---

### 创新点 5：完整云边协同流水线

**文件：** `tools/cloud_edge_adapt.py` (536行) -- 主入口

#### 5.1 单帧处理流程 -- `process_single_frame()`

**调用链：**
```
process_single_frame(batch)
  ├── edge_model.adapt(batch)           # 多粒度特征对齐（创新点2）
  ├── backward + optimizer.step()        # 更新 adapter/BN 参数
  ├── extract_image_features(features)   # 提取 FPN 特征用于不确定性计算
  ├── uncertainty_sampler.should_upload() # 不确定性采样（创新点1）
  ├── if 上传: upload_queue.append(batch)
  └── if len(upload_queue) >= interval: _trigger_cloud_update()
```

#### 5.2 云端更新流程 -- `_trigger_cloud_update()`

**调用链：**
```
_trigger_cloud_update()
  ├── 同步边端权重到学生模型
  ├── distill_trainer.run_full_pipeline()    # 完整云端流水线
  │     ├── adapt_teacher()                  # Phase 1: LoRA 适配（创新点3）
  │     ├── update_best_map() + 回滚检查     # 回滚机制
  │     ├── generate_pseudo_labels()          # Phase 2: 伪标签生成
  │     └── distill_to_student()              # Phase 3: 学生蒸馏
  ├── export_student_fp16()                   # FP16 导出（创新点4）
  └── injector.prepare_next() + try_swap()    # 双缓冲注入（创新点4）
```

#### 5.3 主循环 -- `main()`

```python
# 按域循环：brightness -> defocus_blur -> elastic_transform
for corrupt in corruption_domains:
    # 每个域重置模型（防止跨域遗忘）
    if d_idx > 0:
        pipeline.edge_model = configure_model(cfg, revert=True)
        pipeline.distill_trainer = configure_cloud_edge_models(cfg)

    # 逐帧处理
    for batch in data_loader:
        outputs = pipeline.process_single_frame(batch)
        evaluator.process(batch, outputs)

    # 评估该域的最终 AP50
    results = evaluator.evaluate()
```

---

## 三、数据传输流程图

建议用 draw.io 或 PPT 按以下结构绘制（标注端侧为绿色，云端为蓝色）：

```
┌──────────────────────────────────────────────────────────────────────┐
│                          端 侧 (Edge)                                │
│                                                                      │
│  COCO-C 图像 ──> DataLoader ──> preprocess_image()                   │
│                                       |                              │
│                                  images.tensor                       │
│                                       |                              │
│                             ┌─────────v──────────┐                   │
│                             │  ResNet-50 Backbone │                   │
│                             │  + FPN (5层特征图)  │                   │
│                             └─────────┬──────────┘                   │
│                                       | features (dict: p2~p6)       │
│                             ┌─────────v──────────┐                   │
│                             │  online_adapt=True? │                   │
│                             └──┬─────────────┬───┘                   │
│                           Yes  |             |  No                   │
│                     ┌──────────v───┐   ┌─────v────────┐              │
│                     │   adapt()    │   │  inference()  │              │
│                     │ 多粒度对齐    │   │ RPN -> ROI -> │              │
│                     │ global_kl    │   │ NMS -> 输出   │              │
│                     │ + fg_kl      │   └─────┬────────┘              │
│                     │ + EMA 更新   │         |                       │
│                     └──────────────┘         | predictions           │
│                                              |                       │
│                  ┌───────────────────────────v──────────────┐        │
│                  │       UncertaintySampler                  │        │
│                  │  entropy + KL -> U(x) > threshold ?       │        │
│                  └────────┬──────────────────┬──────────────┘        │
│                      上传  |                  | 不上传                 │
│                           |                  └──> 继续推理            │
└───────────────────────────┼──────────────────────────────────────────┘
                            | 困难样本上传 (~15%, ~3.6 Mbps)
         ┌──────────────────v──────────────────────────────────────────┐
         │                       云 端 (Cloud)                          │
         │                                                              │
         │  ┌───────────────────────────────────────────────┐           │
         │  │ Phase 1: 教师 LoRA 自适应                      │           │
         │  │ ResNet-101 + LoRA(r=16) + L_align              │           │
         │  │ 仅更新低秩矩阵，主干冻结                         │           │
         │  └───────────────────┬───────────────────────────┘           │
         │                      | 适配后的教师模型                       │
         │  ┌───────────────────v───────────────────────────┐           │
         │  │ Phase 2: 伪标签生成                             │           │
         │  │ 教师推理 -> proposals + logits + bbox_deltas     │           │
         │  └───────────────────┬───────────────────────────┘           │
         │                      | 伪标签（软标签）                       │
         │  ┌───────────────────v───────────────────────────┐           │
         │  │ Phase 3: 学生蒸馏                               │           │
         │  │ R50 学生 <- 三级 KD (cls + reg + feat)          │           │
         │  │ 使用教师 proposals 对齐 logit 空间               │           │
         │  └───────────────────┬───────────────────────────┘           │
         │                      | 学生 box_predictor 参数               │
         │  ┌───────────────────v───────────────────────────┐           │
         │  │ export_student_fp16() -> ~50MB FP16 参数包      │           │
         │  └───────────────────┬───────────────────────────┘           │
         └──────────────────────┼───────────────────────────────────────┘
                                | 参数下发 (~50MB)
         ┌──────────────────────v───────────────────────────────────────┐
         │  DualBufferInjector.try_swap()                                │
         │  EMA 混合: new_param = (1-alpha) * edge + alpha * cloud       │
         │  alpha = 0.1, 原子切换, 耗时约 12ms                           │
         │  端侧模型更新完成 -> 继续推理                                   │
         └──────────────────────────────────────────────────────────────┘
```

**流程图绘制要点：**
1. 标注每个节点的数据格式变化：`images -> features(dict) -> proposals -> logits -> predictions`
2. 标注上行链路数据量：困难样本 ~15%，约 3.6 Mbps
3. 标注下行链路数据量：FP16 参数包约 50MB
4. 端侧标绿色，云端标蓝色，通信链路标橙色

---

## 四、代码演示方案

### 演示前准备清单

- [ ] 确认 conda 环境 `tta` 已激活
- [ ] 确认数据集路径 `datasets/coco/val2017` 存在
- [ ] 确认模型权重 `outputs/official/r50_model_final.pkl` 和 `r101_model_final.pkl` 存在
- [ ] 确认源域统计量 `models/stats/` 下文件存在
- [ ] 提前跑一遍确认无报错（避免演示翻车）
- [ ] 终端字体调大，方便投影展示日志

### 演示 1：Source Only -- 展示性能下降问题

```bash
python tools/train_net.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS outputs/official/r50_model_final.pkl \
    TEST.ONLINE_ADAPTATION False
```

**讲解话术：** "这是在 COCO 源域训练好的 R50 检测器，直接用于亮度偏移的目标域。可以看到 AP50 只有 47.18，相比源域下降明显——这就是分布偏移带来的性能退化问题。"

### 演示 2：Edge-only TTA -- 展示特征对齐效果

```bash
python tools/train_net.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS outputs/official/r50_model_final.pkl \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE False
```

**讲解话术：** "加入第四章的多粒度特征对齐后，AP50 提升到 49.78。但边端算力有限，适应能力到此为止。"

### 演示 3：Cloud-Edge Collaborative -- 展示云端增益

```bash
python tools/cloud_edge_adapt.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --cloud-update-interval 50
```

**讲解话术：** "完整的云边协同框架。注意观察日志中几个关键事件——"

**重点关注的日志输出：**

```
# 1. 不确定性采样触发（创新点1）
[UncertaintySampler] 上传样本 entropy=0.82 kl=1.35

# 2. 云端更新触发
[CloudEdge] 触发云端更新 #1，上传 52 个样本

# 3. 教师 LoRA 适配（创新点3）
[CloudDistill] 教师适配前 mAP: 52.30
[CloudDistill] 教师适配后 mAP: 53.15

# 4. 学生蒸馏（创新点3）
[CloudDistill] 学生蒸馏完成，平均损失: {cls: 0.42, reg: 0.18}

# 5. 参数注入（创新点4）
[ParamInjector] 注入 box_predictor 参数, EMA alpha=0.1

# 6. 最终统计
[CloudEdge] 总帧数: 1250, 上传帧数: 156 (12.5%), 云端更新次数: 3
```

### 如果时间紧张的简化方案

只演示实验 3（云边协同），因为它包含所有创新点的运行过程。提前截图好实验 1 和实验 2 的结果作为对比幻灯片。

---

## 五、实验结果展示

### COCO-C 数据集结果（mAP@0.5 / %）

| 方法 | 噪声类 | 模糊类 | 天气类 | 数字类 | 平均 | RS |
|------|--------|--------|--------|--------|------|------|
| Source Only | 38.2 | 39.1 | 41.3 | 43.8 | 40.6 | 67.4% |
| BN Adapt | 41.5 | 42.3 | 44.2 | 46.1 | 43.5 | 72.2% |
| TENT | 43.8 | 44.6 | 46.5 | 47.9 | 45.7 | 75.8% |
| EATA | 45.1 | 44.8 | 47.3 | 48.2 | 46.4 | 77.0% |
| SAR | 44.2 | 45.5 | 47.8 | 49.3 | 46.7 | 77.5% |
| CoTTA | 45.5 | 47.3 | 49.1 | 50.2 | 48.0 | 79.7% |
| ActMAD | 46.8 | 48.2 | 48.7 | 50.5 | 48.6 | 80.6% |
| **Ours** | **48.3** | **49.5** | **51.3** | **52.1** | **50.3** | **83.4%** |

> 相比最优基线 ActMAD 提升 **1.5pp**，相比 CoTTA 提升 **2.1pp**

### SHIFT 数据集结果（mAP@0.5:0.95 / %）

| 方法 | 白天-晴天 | 黄昏-雨天 | 夜晚-雾天 | 平均 |
|------|-----------|-----------|-----------|------|
| Source Only | 42.1 | 28.5 | 15.2 | 28.6 |
| TENT | 44.5 | 32.3 | 19.5 | 32.1 |
| CoTTA | 45.5 | 34.6 | 22.1 | 34.1 |
| ActMAD | 46.5 | 34.2 | 22.5 | 34.4 |
| **Ours** | **47.2** | **37.1** | **25.8** | **36.7** |

> 相比 ActMAD 提升 **2.3pp**，极端场景（夜晚-雾天）相比 CoTTA 提升 **3.7pp**

### 通信效率对比

| 方法 | 上传率 | 下行传输量 | 通信带宽 |
|------|--------|-----------|----------|
| 全量上传 | 100% | --- | ~24 Mbps |
| 随机采样 (20%) | 20% | --- | ~4.8 Mbps |
| 熵值采样 | 18.5% | --- | ~4.4 Mbps |
| **Ours** | **15.3%** | **~50 MB/次** | **~3.6 Mbps** |

### 计算效率对比

| 方法 | 云端处理 | 注入延迟 | 端侧训练 | 端侧 FPS |
|------|---------|---------|---------|----------|
| Full Fine-tuning | 180s | 5000ms | 需要 | 15 |
| TENT | 0s | 50ms | 需要 | 28 |
| CoTTA | 0s | 100ms | 需要 | 25 |
| **Ours** | **16s** | **12ms** | **不需要** | **30** |

### 关键消融实验结果

**知识蒸馏策略消融（mAP@0.5 / %）：**

| 大模型直接下发 | 分类蒸馏 | 回归蒸馏 | 特征蒸馏 | mAP |
|---------------|---------|---------|---------|-----|
| Yes | --- | --- | --- | 48.2 |
| --- | Yes | --- | --- | 50.8 |
| --- | Yes | Yes | --- | 51.6 |
| --- | Yes | Yes | Yes | **52.5** |

**不确定性采样策略消融：**

| 采样策略 | mAP (%) | 上传率 (%) |
|---------|---------|-----------|
| 全量上传 | 52.5 | 100.0 |
| 随机采样 | 50.1 | 20.0 |
| 熵值采样 | 51.8 | 18.5 |
| KL散度采样 | 52.1 | 16.2 |
| **动态阈值+熵&KL组合** | **52.5** | **15.3** |

---

## 六、讲解时间规划（约 27 分钟）

| 环节 | 时间 | 内容 | 操作 |
|------|------|------|------|
| 框架介绍 | 2 min | Detectron2 + Faster R-CNN 架构 | PPT |
| 代码结构 | 3 min | 打开 `cloud_edge_adapt.py` 讲主入口 | IDE/编辑器 |
| 创新点讲解 | 10 min | 逐个打开 5 个关键文件，对照论文公式 | IDE/编辑器 |
| 数据流程图 | 2 min | 展示端->云->端数据流 | PPT |
| 现场演示 | 8 min | 依次运行 3 组实验，对比结果 | 终端 |
| 结果总结 | 2 min | 展示三域对比表格 + 效率指标 | PPT |
