# 云边协同框架下持续动态的测试时自适应目标检测

> **毕设论文实现代码**
> 西安电子科技大学 · 电子信息（软件工程） · 2026年3月
>
> 本项目基于 [natureyoo/ContinualTTA_ObjectDetection](https://github.com/natureyoo/ContinualTTA_ObjectDetection)（CVPR 2024）进行扩展，
> 加入云边协同框架，解决边缘端在资源受限条件下的持续域适应问题。

---

## 研究背景

边缘部署的目标检测模型在实际环境中面临持续的域漂移（天气、光照、遮挡等），
但边缘设备算力有限，无法在线微调模型；而将全部数据上传至云端又面临极高的带宽压力。

**核心矛盾**：边缘算力约束 vs 持续变化的环境分布。

---

## 三大核心贡献

### 贡献1：不确定性驱动的边云协同采样（Chapter 3）

边缘设备通过组合不确定性度量，**只上传约 15% 的最困难样本**：

$$U(x) = \max\left\{\frac{H(x)}{\log C},\ \frac{D_{KL}(x)}{D_{KL}^{ref}}\right\}$$

- **预测熵** H(x)：按置信度和面积加权，归一化到 [0,1]
- **特征KL散度** D_KL(x)：目标域特征分布与源域分布的散度
- **动态阈值**：滑动窗口百分位自适应调整
- **类均衡约束**：防止易类样本主导，逆频率加权上传上限

实测效果：上传率 **15.3%**，通信带宽 **3.6 Mbps**（vs 全量上传 24 Mbps），性能与全量上传等价。

---

### 贡献2：多粒度特征对齐算法（Chapter 4）

云端教师模型通过两级特征对齐损失，无监督地适配到目标域：

- **图像级对齐**：GAP特征的KL散度，EMA平滑（gamma=128）
- **区域级类感知对齐**：RoI特征按类别分组，逆频率加权

总损失：L_align = lambda_img * L_img + lambda_roi * L_roi

---

### 贡献3：云端双模型架构与多级知识蒸馏（Chapter 5）

- **教师模型**：ResNet-101-FPN + LoRA（rank=16, ~0.5M 可训练参数），云端适配
- **学生模型**：ResNet-50-FPN（与边端完全相同），全参数蒸馏训练

**三级蒸馏损失**：L_KD = lambda_cls * L_cls + lambda_reg * L_reg + lambda_feat * L_feat

| 级别 | 损失 | 默认权重 |
|------|------|---------|
| 分类蒸馏 | KL散度，温度 T=3，缩放 T^2 | lambda_cls=1.0 |
| 回归蒸馏 | Smooth L1 bbox预测对齐 | lambda_reg=0.5 |
| 特征蒸馏 | FPN层级MSE，1x1 Conv通道对齐 | lambda_feat=0.5 |

**边端参数注入**：FP16序列化（~50MB），双缓冲机制，mmap加载（~8ms），注入期间保持 30+ FPS。

---

## 实验结果

### COCO-C（mAP@0.5）

| 方法 | Noise | Blur | Weather | Digital | 平均 | RS |
|------|-------|------|---------|---------|------|----|
| Source Only | 38.2 | 39.1 | 41.3 | 43.8 | 40.6 | 67.4% |
| TENT | 43.8 | 44.6 | 46.5 | 47.9 | 45.7 | 75.8% |
| CoTTA | 45.5 | 47.3 | 49.1 | 50.2 | 48.0 | 79.7% |
| ActMAD | 46.8 | 48.2 | 48.7 | 50.5 | 48.6 | 80.6% |
| **本文方法** | **48.3** | **49.5** | **51.3** | **52.1** | **50.3** | **83.4%** |

### SHIFT（mAP@0.5:0.95）

| 方法 | Day-Clear | Dusk-Rain | Night-Fog | 平均 |
|------|-----------|-----------|-----------|------|
| Source Only | 42.1 | 28.5 | 15.2 | 28.6 |
| ActMAD | 46.5 | 34.2 | 22.5 | 34.4 |
| **本文方法** | **47.2** | **37.1** | **25.8** | **36.7** |

---

## 环境配置

```bash
conda env create -f tta_jayeon.yaml
conda activate tta
pip install -e .
```

环境要求：Python 3.10, PyTorch 1.11.0, CUDA 11.3+

---

## 数据集准备

**COCO-C**：下载 COCO val2017，使用 ImageCorruptions 生成15种腐蚀。

**SHIFT 数据集**：参考 [SHIFT Dataset](https://www.vis.xyz/shift/) 官网下载。

---

## 运行实验

### 纯边端 TTA（多粒度特征对齐，复现上游论文）

```bash
# 1. 收集源域特征统计
# 2. 直接测试基线
# 3. 边端 TTA
bash scripts/coco_adapt_R50.sh
```

### 云边协同完整框架（论文完整方法）

```bash
# 前提：需要 R50 和 R101 基础模型权重
# - outputs/coco_base_r50/model_final.pth
# - outputs/coco_base_r101/model_final.pth  (参考 R101_README.md)

bash scripts/cloud_edge_coco.sh
```

或手动运行：

```bash
python tools/cloud_edge_adapt.py \
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \
    --num-gpus 1 \
    --cloud-update-interval 200 \
    OUTPUT_DIR ./outputs/CloudEdge_COCO_full
```

### SHIFT 数据集实验

```bash
bash scripts/shift_discrete_adapt.sh
bash scripts/shift_continuous_adapt.sh
```

---

## 项目结构

```
ContinualTTA_ObjectDetection/
├── configs/
│   ├── Base/                       # 基础模型配置 (R50/R101/SwinT)
│   └── TTA/
│       ├── COCO_R50.yaml           # 边端TTA (R50)
│       ├── COCO_R101.yaml          # 边端TTA (R101)
│       └── CloudEdge_COCO_R101_R50.yaml  # 云边协同框架配置 (新增)
├── detectron2/
│   ├── modeling/
│   │   ├── meta_arch/rcnn.py       # adapt() 特征对齐核心
│   │   ├── cloud_distillation.py   # 云端双模型蒸馏 (新增)
│   │   ├── uncertainty_sampling.py # 边端不确定性采样 (新增)
│   │   └── configure_adaptation_model.py  # 模型配置（含云边函数）
│   ├── utils/
│   │   ├── distill_loss.py         # 三级蒸馏损失（扩展）
│   │   └── param_injection.py      # 双缓冲参数注入 (新增)
│   └── config/defaults.py          # 配置项（含云边协同参数）
├── tools/
│   ├── train_net.py                # 标准训练/TTA入口
│   └── cloud_edge_adapt.py         # 云边协同流程入口 (新增)
└── scripts/
    ├── coco_adapt_R50.sh           # R50 实验
    ├── coco_adapt_R101.sh          # R101 实验
    └── cloud_edge_coco.sh          # 云边协同实验 (新增)
```

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `UNCERTAINTY_PERCENTILE` | 0.7 | 不确定性采样百分位阈值 |
| `SLIDING_WINDOW_SIZE` | 100 | 动态阈值滑动窗口大小 |
| `EMA_GAMMA` | 128 | 目标域特征统计EMA平滑系数 |
| `DISTILL_TEMPERATURE` | 3.0 | 知识蒸馏温度 |
| `LAMBDA_OUTPUT` | 1.0 | 分类蒸馏权重 |
| `LAMBDA_REG` | 0.5 | 回归蒸馏权重 |
| `LAMBDA_FEATURE` | 0.5 | 特征蒸馏权重 |
| `FORGETTING_BETA` | 0.3 | 防遗忘正则强度 |
| `ROLLBACK_THRESHOLD` | 5.0 | mAP下降触发回滚阈值（pp） |

---

## 致谢

本项目代码基于以下开源工作：

- [natureyoo/ContinualTTA_ObjectDetection](https://github.com/natureyoo/ContinualTTA_ObjectDetection)：CVPR 2024 持续TTA目标检测框架
- [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)：通用目标检测框架
