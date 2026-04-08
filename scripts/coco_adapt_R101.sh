#!/bin/bash

# Backbone Type
export bb="R101"

# Set config file
export config_path="configs/TTA/COCO_${bb}.yaml"

# Collect Feature Statistics (if needed)
python tools/train_net.py \
  	--config-file "./configs/Base/COCO_faster_rcnn_${bb}_FPN_1x.yaml" \
  	--eval-only \
  	MODEL.WEIGHTS models/checkpoints/faster_rcnn_${bb}_coco.pth \
  	TEST.CONTINUAL_DOMAIN "False" \
  	TEST.COLLECT_FEATURES "True" \
  	OUTPUT_DIR outputs/COCO/${bb}_collect_feature_stats

# Direct Test (Baseline)
python tools/train_net.py \
  	--config-file "./configs/Base/COCO_faster_rcnn_${bb}_FPN_1x.yaml" \
  	--eval-only \
  	MODEL.WEIGHTS models/checkpoints/faster_rcnn_${bb}_coco.pth \
  	TEST.CONTINUAL_DOMAIN "True" \
  	OUTPUT_DIR outputs/COCO/${bb}_direct_test

# Ours (Domain Adaptation with Adapter)
python tools/train_net.py \
  	--config-file ${config_path} \
  	--eval-only \
  	SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
  	TEST.ADAPTATION.WHERE "adapter" \
  	TEST.ADAPTATION.CONTINUAL "True" \
  	TEST.ADAPTATION.SOURCE_FEATS_PATH "./models/stats/COCO_${bb}_stats.pt" \
  	OUTPUT_DIR outputs/COCO/${bb}_ours_continual_clip_1_0
