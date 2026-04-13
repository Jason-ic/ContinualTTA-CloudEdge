# AutoDL 服务器部署指南

本指南面向在 AutoDL 云服务器上部署和运行本项目（云边协同持续自适应目标检测）。

---

## 1. 服务器选型与存储规划

**推荐配置**：RTX 4090 / A5000 以上，24GB+ 显存，CUDA 11.3+

**AutoDL 存储说明**：

| 路径 | 空间 | 用途 |
|------|------|------|
| `/root`（系统盘） | ~30 GB | 代码和 conda 环境 |
| `/root/autodl-tmp`（数据盘） | 数百 GB | 数据集、权重、实验输出 |

> **重要**：数据集和实验输出必须放在 `/root/autodl-tmp`，系统盘极易被占满。

---

## 2. 获取代码

```bash
cd /root
git clone https://github.com/youfangdajiankang/ContinualTTA-CloudEdge.git
cd ContinualTTA-CloudEdge
```

---

## 3. 环境配置

### 3.1 修复 conda yaml 文件

原始 `tta_jayeon.yaml` 包含硬编码路径，需先修复：

```bash
sed -i 's|^name:.*|name: tta|' tta_jayeon.yaml
sed -i '/^prefix:/d' tta_jayeon.yaml
```

### 3.2 移除不可用的 pip 包

`clip==1.0` 在 PyPI 上不存在，ResNet 实验不需要它：

```bash
sed -i '/clip==1.0/d' tta_jayeon.yaml
```

### 3.3 创建 conda 环境

```bash
conda env create -f tta_jayeon.yaml
conda activate tta
```

> 约需 15-30 分钟。如提示磁盘满，参见第 9 节。

### 3.4 安装项目

```bash
pip install -e .
```

### 3.5 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期：1.11.0 True

python -c "import detectron2; print(detectron2.__file__)"
```

### 3.6 修复缺失的 datasets 模块

`detectron2/data/datasets/` 目录可能未被 git 追踪，需要从原始仓库补充：

```bash
ls detectron2/data/datasets/ 2>/dev/null || echo "目录缺失，需要补充"

# 如果缺失，从原始仓库复制
git clone --depth=1 https://github.com/natureyoo/ContinualTTA_ObjectDetection.git /tmp/d2src
cp -r /tmp/d2src/detectron2/data/datasets detectron2/data/
rm -rf /tmp/d2src
```

---

## 4. 数据集准备

### 4.1 COCO val2017

```bash
mkdir -p /root/autodl-tmp/datasets/coco/annotations
cd /root/autodl-tmp/datasets

# 下载图片
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco/ && rm val2017.zip

# 下载标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco/ && rm annotations_trainval2017.zip
```

### 4.2 COCO-C（腐蚀数据集）

代码会使用 `imagecorruptions` 库在线生成腐蚀图像。当前代码默认使用 3 种腐蚀：
- brightness, defocus_blur, elastic_transform

生成的腐蚀图像缓存在 `datasets/coco/val2017-{corruption_name}/` 目录下。

### 4.3 SHIFT 数据集（可选）

参考 [SHIFT 官网](https://www.vis.xyz/shift/) 下载，放于 `/root/autodl-tmp/datasets/shift/`。

---

## 5. 模型权重

### 5.1 R50 基础模型

需要在 COCO 上训练好的 Faster R-CNN R50-FPN 权重：

```bash
mkdir -p /root/autodl-tmp/outputs/coco_base_r50

# 方式1：使用 Detectron2 预训练权重
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl \
     -O /root/autodl-tmp/outputs/coco_base_r50/model_final.pth

# 方式2：自行训练（参考 scripts/coco_adapt_R50.sh）
```

### 5.2 R101 基础模型（云端教师）

```bash
mkdir -p /root/autodl-tmp/outputs/coco_base_r101

wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl \
     -O /root/autodl-tmp/outputs/coco_base_r101/model_final.pth
```

### 5.3 源域特征统计

```bash
mkdir -p /root/autodl-tmp/models/stats

# 收集 R50 源域特征统计
python tools/train_net.py \
    --config-file configs/TTA/COCO_R50.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS /root/autodl-tmp/outputs/coco_base_r50/model_final.pth \
    OUTPUT_DIR /root/autodl-tmp/outputs/coco_base_r50 \
    TEST.COLLECT_SOURCE_STATS True
```

统计文件保存到 `models/stats/COCO_R50_stats.pt`。

---

## 6. 符号链接配置

将项目默认相对路径指向数据盘：

```bash
cd /root/ContinualTTA-CloudEdge

ln -sfn /root/autodl-tmp/outputs outputs
ln -sfn /root/autodl-tmp/datasets datasets
ln -sfn /root/autodl-tmp/models models
```

验证：

```bash
ls -la outputs datasets models
ls outputs/coco_base_r50/model_final.pth
ls outputs/coco_base_r101/model_final.pth
ls models/stats/COCO_R50_stats.pt
```

---

## 7. 运行完整实验

每次使用前：

```bash
conda activate tta
cd /root/ContinualTTA-CloudEdge
export OMP_NUM_THREADS=1
```

### 7.1 实验1：Source Only（无适应 baseline）

评估源域训练模型在目标域上的性能，不进行任何在线适应。

```bash
python tools/train_net.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS /root/lyh2/ContinualTTA-CloudEdge/outputs/official/r50_model_final.pkl \
    TEST.ONLINE_ADAPTATION False
```

### 7.2 实验2：纯边端 TTA（Edge-only）

只使用边端模型的多粒度特征对齐进行在线适应，不启用云边协同。

```bash
python tools/train_net.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS /root/lyh2/ContinualTTA-CloudEdge/outputs/official/r50_model_final.pkl \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE False
```

### 7.3 实验3：云边协同 TTA（Cloud-Edge Collaborative）

完整框架：边端自适应 + 云端教师 LoRA 适配 + 知识蒸馏 + 参数注入。

```bash
python tools/cloud_edge_adapt.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --cloud-update-interval 50
```

**参数说明**：
- `--cloud-update-interval 50`: 每 50 帧触发一次云端更新（上传样本 → 云端适配 → 蒸馏 → 注入）
- `--num-gpus 1`: 单卡仿真模式（云端和边端共享 GPU）

---

## 8. 实验结果

### 8.1 COCO-C 三域测试结果

| Domain | Source Only | Cloud-Edge | Δ AP50 | Δ AP |
|--------|-------------|------------|--------|------|
| **brightness** | 47.18 / 30.05 | **49.97** / 31.72 | **+2.79** | +1.67 |
| **defocus_blur** | 14.32 / 8.28 | 9.70 / 5.31 | -4.62 | -2.97 |
| **elastic_transform** | 25.38 / 14.70 | **40.08** / 23.21 | **+14.70** | +8.51 |
| **平均** | 28.96 / 17.68 | **33.25** / 20.08 | +4.29 | +2.40 |

### 8.2 结果分析

1. **brightness**: 云边协同带来 **+2.79 AP50** 提升，验证了框架在亮度偏移域的有效性
2. **elastic_transform**: 巨大提升 **+14.70 AP50**，说明云端教师在形变域上能提供强有力的指导
3. **defocus_blur**: 性能下降 **-4.62 AP50**，原因可能是：
   - R101 教师对模糊更敏感（深层网络对模糊的容忍度低）
   - 跨架构的 proposal mismatch 在模糊域上更严重
   - 可考虑针对模糊域关闭云边协同，或使用更保守的注入策略

**整体评估**：三域平均提升 +4.29 AP50，框架在亮度偏移和形变域上效果显著。

---

## 9. 运行实验（旧版，保留参考）

```bash
python tools/cloud_edge_adapt.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 --edge-only \
    OUTPUT_DIR /root/autodl-tmp/outputs/CloudEdge_COCO/edge_only_tta \
    CLOUD_EDGE.CLOUD_DIR /root/autodl-tmp/cloud_edge/cloud \
    CLOUD_EDGE.EDGE_DIR /root/autodl-tmp/cloud_edge/edge \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE False \
    TEST.ADAPTATION.WHERE adapter
```

### 7.2 云边协同完整框架（实验3，论文方法）

```bash
python tools/cloud_edge_adapt.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 \
    --cloud-update-interval 15 \
    OUTPUT_DIR /root/autodl-tmp/outputs/CloudEdge_COCO/cloud_edge_full \
    CLOUD_EDGE.CLOUD_DIR /root/autodl-tmp/cloud_edge/cloud \
    CLOUD_EDGE.EDGE_DIR /root/autodl-tmp/cloud_edge/edge \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE True \
    TEST.ADAPTATION.UNCERTAINTY_PERCENTILE 0.3
```

### 7.3 一键运行所有实验

```bash
bash scripts/cloud_edge_coco.sh
```

### 7.4 使用原始 TTA 框架（对比基线）

```bash
python tools/train_net.py \
    --config-file configs/TTA/COCO_R50.yaml \
    --test-only \
    OUTPUT_DIR /root/autodl-tmp/outputs/edge_baseline
```

---

## 8. 关键路径汇总

| 内容 | 路径 |
|------|------|
| 代码仓库 | `/root/ContinualTTA-CloudEdge/` |
| COCO 数据集 | `/root/autodl-tmp/datasets/coco/` |
| R50 权重 | `/root/autodl-tmp/outputs/coco_base_r50/model_final.pth` |
| R101 权重 | `/root/autodl-tmp/outputs/coco_base_r101/model_final.pth` |
| 源域特征统计 | `/root/autodl-tmp/models/stats/COCO_R50_stats.pt` |
| 实验输出 | `/root/autodl-tmp/outputs/CloudEdge_COCO/` |
| 云端参数缓存 | `/root/autodl-tmp/cloud_edge/cloud/` |

---

## 9. 常见问题与解决方案

### 9.1 conda 报错：Invalid environment name

**原因**：`tta_jayeon.yaml` 中 name 字段含绝对路径 `/data/jayeon/env/tta_jayeon`

**解决**：按 3.1 节修复 yaml 文件

---

### 9.2 pip 报错：clip==1.0 找不到

**原因**：`clip==1.0` 不在 PyPI 上

**解决**：按 3.2 节删除该行。本项目使用 ResNet，不需要 CLIP

---

### 9.3 磁盘空间不足 / No space left on device

```bash
# 清理 conda 缓存（通常可释放几 GB）
conda clean --all -y
rm -rf /root/miniconda3/pkgs/*

# 清理 pip 缓存
rm -rf /root/.cache/pip

# 清理回收站
rm -rf /root/.local/share/Trash

# 清理云端导出的参数文件
rm -rf /root/autodl-tmp/cloud_edge/cloud/*.pt

# 查看占用
df -h /
du -sh /root/miniconda3/ /root/.cache/ /tmp/
```

---

### 9.4 /tmp 不可用 (FileNotFoundError)

**原因**：之前的进程 crash 可能损坏 /tmp

```bash
mkdir -p /tmp
chmod 1777 /tmp
```

---

### 9.5 OMP_NUM_THREADS 警告

```bash
export OMP_NUM_THREADS=1

# 永久生效
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
```

---

### 9.6 ImportError: cannot import name 'datasets'

**原因**：`detectron2/data/datasets/` 目录缺失

**解决**：按 3.6 节从原始仓库补充

---

### 9.7 CUDA out of memory

- 减小 `--cloud-update-interval`（如从 30 → 15）
- 减小 `SOLVER.IMS_PER_BATCH_TEST`（如从 4 → 2）
- 实验输出放在数据盘，避免系统盘占满导致 swap 失败

---

## 推荐 ~/.bashrc 追加

```bash
echo '
conda activate tta
export OMP_NUM_THREADS=1
alias cdp="cd /root/ContinualTTA-CloudEdge"
' >> ~/.bashrc
```
