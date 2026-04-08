#!/bin/bash
# 云边协同持续自适应目标检测 - COCO-C 实验脚本
# Cloud-Edge Collaborative Continual TTA on COCO-Corruption
#
# 实验前准备：
#   1. 训练 R50 基础模型:  scripts/coco_adapt_R50.sh (collect stats 阶段)
#   2. 训练 R101 基础模型: scripts/coco_adapt_R101.sh
#   3. 确认以下权重文件存在:
#      - outputs/coco_base_r50/model_final.pth
#      - outputs/coco_base_r101/model_final.pth
#      - models/stats/COCO_R50_stats.pt

set -e

CONFIG="configs/TTA/CloudEdge_COCO_R101_R50.yaml"
NGPU=1
OUTPUT_BASE="./outputs/CloudEdge_COCO"

# ============================================================
# 实验1: 纯边端推理（基线，无适配）
# ============================================================
echo "================================================"
echo "[实验1] 纯边端推理基线（Source Only）"
echo "================================================"
python tools/cloud_edge_adapt.py \
    --config-file ${CONFIG} \
    --num-gpus ${NGPU} \
    --eval-only \
    OUTPUT_DIR "${OUTPUT_BASE}/source_only" \
    TEST.ONLINE_ADAPTATION False

# ============================================================
# 实验2: 纯边端 TTA（无云端蒸馏）
# ============================================================
echo "================================================"
echo "[实验2] 纯边端 TTA（多粒度特征对齐）"
echo "================================================"
python tools/cloud_edge_adapt.py \
    --config-file ${CONFIG} \
    --num-gpus ${NGPU} \
    --edge-only \
    OUTPUT_DIR "${OUTPUT_BASE}/edge_only_tta" \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE False \
    TEST.ADAPTATION.WHERE "adapter"

# ============================================================
# 实验3: 云边协同完整框架（论文方法）
# ============================================================
echo "================================================"
echo "[实验3] 云边协同完整框架（论文方法）"
echo "================================================"
python tools/cloud_edge_adapt.py \
    --config-file ${CONFIG} \
    --num-gpus ${NGPU} \
    --cloud-update-interval 200 \
    OUTPUT_DIR "${OUTPUT_BASE}/cloud_edge_full" \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE True \
    TEST.ADAPTATION.UNCERTAINTY_PERCENTILE 0.7

# ============================================================
# 实验4: 云边协同 - 消融：随机采样（不用不确定性）
# ============================================================
echo "================================================"
echo "[实验4] 消融：随机采样对比"
echo "================================================"
python tools/cloud_edge_adapt.py \
    --config-file ${CONFIG} \
    --num-gpus ${NGPU} \
    --cloud-update-interval 200 \
    OUTPUT_DIR "${OUTPUT_BASE}/cloud_edge_random_sample" \
    TEST.ADAPTATION.ENABLE_CLOUD_EDGE True \
    TEST.ADAPTATION.UNCERTAINTY_PERCENTILE 0.0  # 阈值0 = 随机保留约20%

echo "================================================"
echo "所有实验完成！结果保存在 ${OUTPUT_BASE}/"
echo "================================================"
