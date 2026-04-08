# ResNet-101 配置说明

## 📁 新增的配置文件

### 1. 基础配置
- **文件**: `configs/Base/COCO_faster_rcnn_R101_FPN_1x.yaml`
- **说明**: ResNet-101 + FPN 基础配置
- **参数**: DEPTH=101, ImageNet预训练权重

### 2. TTA配置
- **文件**: `configs/TTA/COCO_R101.yaml`
- **说明**: 域自适应配置
- **特点**: Adapter模块、特征对齐、持续域适应

### 3. 运行脚本
- **文件**: `scripts/coco_adapt_R101.sh`
- **说明**: 一键运行完整流程

### 4. Checkpoint链接
- **文件**: `models/checkpoints/faster_rcnn_R101_coco.pth`
- **当前指向**: `../../outputs/coco_base_r101/model_final.pth`
- **状态**: 待训练

### 5. 特征统计链接
- **文件**: `models/stats/COCO_R101_stats.pt`
- **当前指向**: `../faster_rcnn_R101_coco_feature_stats_new.pt`
- **状态**: 待收集

## 🚀 使用步骤

### 步骤1: 使用R101从头训练模型（如果还没有）

```bash
# 需要先在COCO数据集上训练R101模型
# 参考 R50 的训练命令
```

### 步骤2: 收集源域特征统计

```bash
cd /root/lyh/ContinualTTA_ObjectDetection
conda activate cta_od

python tools/train_net.py \
    --config-file "./configs/Base/COCO_faster_rcnn_R101_FPN_1x.yaml" \
    --eval-only \
    MODEL.WEIGHTS models/checkpoints/faster_rcnn_R101_coco.pth \
    TEST.CONTINUAL_DOMAIN "False" \
    TEST.COLLECT_FEATURES "True" \
    OUTPUT_DIR outputs/COCO/R101_collect_feature_stats
```

### 步骤3: 运行直接测试（基线）

```bash
python tools/train_net.py \
    --config-file "./configs/Base/COCO_faster_rcnn_R101_FPN_1x.yaml" \
    --eval-only \
    MODEL.WEIGHTS models/checkpoints/faster_rcnn_R101_coco.pth \
    TEST.CONTINUAL_DOMAIN "True" \
    OUTPUT_DIR outputs/COCO/R101_direct_test
```

### 步骤4: 运行域自适应算法

```bash
python tools/train_net.py \
    --config-file "./configs/TTA/COCO_R101.yaml" \
    --eval-only \
    SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
    TEST.ADAPTATION.WHERE "adapter" \
    TEST.ADAPTATION.CONTINUAL "True" \
    TEST.ADAPTATION.SOURCE_FEATS_PATH "./models/stats/COCO_R101_stats.pt" \
    OUTPUT_DIR /root/autodl-tmp/COCO_TTA/R101_ours_continual_clip_1_0
```

### 快捷方式：使用脚本

```bash
bash scripts/coco_adapt_R101.sh
```

## 📊 R101 vs R50 对比

| 特性 | ResNet-50 | ResNet-101 |
|------|-----------|-------------|
| **层数** | 50层 | 101层 |
| **参数量** | ~41.7M | ~60.2M |
| **COCO mAP** | 37.4% | **39.8%** (+2.4%) |
| **推理速度** | 1x | 1.3x (慢30%) |
| **训练时间** | 1x | 1.5x |
| **内存占用** | 基准 | +1.5x |

## ⚠️ 注意事项

1. **必须先训练R101模型**：当前checkpoint链接指向不存在的文件
2. **需要更多GPU内存**：R101比R50需要更多显存
3. **训练时间更长**：约是R50的1.5倍

## 📝 配置差异

与R50相比，R101的主要差异：
- DEPTH: 50 → 101
- 参数量增加：~41.7M → ~60.2M
- 性能提升：约+2.4% mAP
- 推理速度：慢约30%
