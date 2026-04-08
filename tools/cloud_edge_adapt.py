#!/usr/bin/env python3
"""
云边协同持续自适应目标检测 - 完整流程入口
Cloud-Edge Collaborative Continual TTA Object Detection - Full Pipeline

流程:
  边端: 连续推理 → 不确定性采样 → 上传困难样本
  云端: LoRA特征对齐适配教师 → 知识蒸馏到学生 → FP16导出
  边端: 双缓冲参数注入 → 继续推理

使用示例:
  python tools/cloud_edge_adapt.py \\
    --config-file configs/TTA/CloudEdge_COCO_R101_R50.yaml \\
    --num-gpus 1 \\
    MODEL.WEIGHTS ./outputs/coco_base_r50/model_final.pth \\
    OUTPUT_DIR ./outputs/CloudEdge_test

详细说明参见 R101_README.md
"""

import argparse
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

# Detectron2 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.configure_adaptation_model import (
    configure_model,
    configure_cloud_edge_models,
)
from detectron2.modeling.uncertainty_sampling import UncertaintySampler, extract_image_features
from detectron2.utils.param_injection import export_student_fp16, DualBufferInjector
import detectron2.utils.comm as comm

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_argument_parser():
    parser = argparse.ArgumentParser(description="云边协同持续自适应目标检测")
    parser.add_argument(
        "--config-file",
        default="configs/TTA/CloudEdge_COCO_R101_R50.yaml",
        metavar="FILE",
        help="配置文件路径",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="GPU数量"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="只评估，不执行适配（用于基线测试）"
    )
    parser.add_argument(
        "--edge-only", action="store_true",
        help="只运行边端适配（不调用云端蒸馏）"
    )
    parser.add_argument(
        "--cloud-update-interval", type=int, default=200,
        help="每处理多少帧触发一次云端更新（默认200）",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="使用 wandb 记录实验"
    )
    parser.add_argument(
        "opts",
        help="通过命令行覆盖配置",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


class CloudEdgePipeline:
    """
    云边协同完整 pipeline。

    模拟实际部署场景：
    1. 边端持续推理，计算每帧不确定性
    2. 超过动态阈值的帧加入上传队列
    3. 积累足够样本后触发云端更新
    4. 云端蒸馏完成后导出参数，边端双缓冲注入

    注意：本脚本为仿真模式（边端云端在同一进程），
    实际部署时边端和云端为独立进程/机器，通过文件系统或网络交换参数。
    """

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 创建输出目录
        os.makedirs(cfg.CLOUD_EDGE.EDGE_DIR, exist_ok=True)
        os.makedirs(cfg.CLOUD_EDGE.CLOUD_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # 边端模型（R50）
        self.edge_model, self.edge_optimizer, _ = configure_model(
            cfg, DefaultTrainer, revert=True
        )
        self.edge_model.to(self.device)
        self.edge_model.eval()
        self.edge_model.online_adapt = True

        # 云边模型配置（教师R101 + 学生R50）
        self.distill_trainer = None
        if cfg.TEST.ADAPTATION.ENABLE_CLOUD_EDGE and not args.edge_only:
            logger.info("[CloudEdge] 初始化云端双模型蒸馏框架...")
            try:
                self.distill_trainer = configure_cloud_edge_models(cfg, DefaultTrainer)
                logger.info("[CloudEdge] 云端框架初始化成功")
            except Exception as e:
                logger.warning(f"[CloudEdge] 云端框架初始化失败（R101权重可能未准备好）: {e}")
                logger.warning("[CloudEdge] 降级为纯边端模式")

        # 不确定性采样器
        source_stats = None
        if os.path.exists(cfg.TEST.ADAPTATION.SOURCE_FEATS_PATH):
            source_stats = torch.load(cfg.TEST.ADAPTATION.SOURCE_FEATS_PATH, map_location='cpu')

        self.uncertainty_sampler = UncertaintySampler(
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            source_feat_stats=source_stats,
            sliding_window_size=cfg.TEST.ADAPTATION.SLIDING_WINDOW_SIZE,
            percentile=cfg.TEST.ADAPTATION.UNCERTAINTY_PERCENTILE,
            kl_ref=cfg.TEST.ADAPTATION.KL_REF,
            s_min=cfg.TEST.ADAPTATION.S_MIN,
        )

        # 双缓冲参数注入器
        self.injector = DualBufferInjector(self.edge_model, device=self.device)

        # 上传队列和统计
        self.upload_queue = []
        self.cloud_update_interval = args.cloud_update_interval
        self.frame_count = 0
        self.cloud_update_count = 0
        self.stats = {
            'total_frames': 0,
            'uploaded_frames': 0,
            'cloud_updates': 0,
            'edge_adapt_time': 0.0,
            'cloud_adapt_time': 0.0,
        }

    def process_single_frame(self, batch):
        """
        处理单帧：边端推理 + 不确定性评估 + 按需加入上传队列。

        Returns:
            (predictions, should_upload, uncertainty_score)
        """
        t0 = time.time()
        self.frame_count += 1
        self.stats['total_frames'] += 1

        # 边端前向推理 + 特征对齐适配（adapt 返回 postprocessed results）
        if self.edge_optimizer is not None:
            self.edge_optimizer.zero_grad()

        results, adapt_losses, _ = self.edge_model.adapt(batch)

        if self.edge_optimizer is not None and adapt_losses:
            loss = sum(adapt_losses.values())
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                self.edge_optimizer.step()

        # 提取特征和预测用于不确定性评估
        with torch.no_grad():
            images = self.edge_model.preprocess_image(batch)
            features = self.edge_model.backbone(images.tensor)
            if isinstance(features, tuple):
                features = features[0]
            proposals, _ = self.edge_model.proposal_generator(images, features, None, eval=True)
            _, predictions, _ = self.edge_model.roi_heads._forward_box(features, proposals, outs=True)

        # 不确定性评估
        should_upload = False
        uncertainty = 0.0
        if self.cfg.TEST.ADAPTATION.ENABLE_CLOUD_EDGE and self.distill_trainer is not None:
            class_scores = torch.nn.functional.softmax(predictions[0], dim=1) if predictions else None
            feat_dict = extract_image_features(features)

            if class_scores is not None:
                pred_cls = predictions[0].argmax(dim=1).tolist()
                should_upload, uncertainty = self.uncertainty_sampler.should_upload(
                    class_scores=class_scores,
                    feat_dict=feat_dict,
                    pred_classes=pred_cls,
                )

            if should_upload:
                self.upload_queue.append(batch)
                self.stats['uploaded_frames'] += 1

        # 检查是否触发云端更新
        if (len(self.upload_queue) >= self.cloud_update_interval
                and self.distill_trainer is not None):
            self._trigger_cloud_update()

        # 尝试切换最新参数
        self.injector.try_swap()

        self.stats['edge_adapt_time'] += time.time() - t0
        return results, should_upload, uncertainty

    def _trigger_cloud_update(self):
        """触发云端双模型蒸馏更新"""
        if not self.upload_queue:
            return

        logger.info(
            f"[CloudEdge] 触发云端更新 (第{self.cloud_update_count+1}次)"
            f"，上传样本: {len(self.upload_queue)}"
        )
        t0 = time.time()

        try:
            def align_loss_fn(model, batch):
                model.online_adapt = True
                _, adapt_losses, _ = model.adapt(batch)
                return adapt_losses if adapt_losses else {}

            all_losses = self.distill_trainer.run_full_pipeline(
                uploaded_batches=self.upload_queue,
                align_loss_fn=align_loss_fn,
            )
            logger.info(f"[CloudEdge] 云端更新完成，损失: {all_losses}")

            # 导出学生模型参数到共享目录
            param_path = os.path.join(
                self.cfg.CLOUD_EDGE.CLOUD_DIR,
                f"student_update_{self.cloud_update_count:04d}.pt"
            )
            export_student_fp16(self.distill_trainer.student, param_path)

            # 触发边端异步参数加载
            self.injector.prepare_next(param_path)

            self.cloud_update_count += 1
            self.stats['cloud_updates'] += 1
            self.stats['cloud_adapt_time'] += time.time() - t0

        except Exception as e:
            logger.error(f"[CloudEdge] 云端更新失败: {e}")
        finally:
            # 清空上传队列（保留最新 10 帧防止冷启动）
            self.upload_queue = self.upload_queue[-10:]

    def print_stats(self):
        """打印运行统计"""
        s = self.stats
        upload_rate = s['uploaded_frames'] / max(s['total_frames'], 1) * 100
        avg_edge_ms = s['edge_adapt_time'] / max(s['total_frames'], 1) * 1000
        logger.info("=" * 60)
        logger.info(f"[CloudEdge] 运行统计:")
        logger.info(f"  总帧数:       {s['total_frames']}")
        logger.info(f"  上传帧数:     {s['uploaded_frames']} ({upload_rate:.1f}%)")
        logger.info(f"  云端更新次数: {s['cloud_updates']}")
        logger.info(f"  边端平均耗时: {avg_edge_ms:.1f} ms/帧")
        logger.info(f"  参数注入次数: {self.injector.swap_count}")
        logger.info("=" * 60)


def run_evaluation(cfg, args):
    """仅评估模式：运行标准推理评估"""
    from detectron2.engine.defaults import DefaultTrainer as DT
    model = DT.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()

    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = DT.build_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        result = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = result
        if comm.is_main_process():
            logger.info(f"Evaluation results for {dataset_name}: {result}")
    return results


def run_cloud_edge_adaptation(cfg, args):
    """云边协同适配主流程（仿真模式）"""
    from detectron2.engine.defaults import DefaultTrainer as DT
    from detectron2.evaluation import COCOEvaluator
    from detectron2.modeling.configure_adaptation_model import configure_model

    pipeline = CloudEdgePipeline(cfg, args)

    results = OrderedDict()

    # 确定腐蚀域列表
    for dataset_name in cfg.DATASETS.TEST:
        if 'coco' in dataset_name:
            corruption_domains = ['brightness', 'defocus_blur', 'elastic_transform']
        else:
            corruption_domains = ['fog', 'rain', 'snow']

        for d_idx, corrupt in enumerate(corruption_domains):
            corrupt_dataset = f"{dataset_name}-{corrupt}"
            logger.info(f"\n{'='*60}")
            logger.info(f"[CloudEdge] 开始处理域: {corrupt_dataset} ({d_idx+1}/{len(corruption_domains)})")

            # 非持续模式下每个域重置模型
            if d_idx == 0 or not cfg.TEST.ADAPTATION.CONTINUAL:
                pipeline.edge_model, pipeline.edge_optimizer, _ = configure_model(
                    cfg, DefaultTrainer, revert=True
                )
                pipeline.edge_model.to(pipeline.device)
                pipeline.edge_model.eval()
                pipeline.edge_model.online_adapt = True

            data_loader = DT.build_test_loader(cfg, corrupt_dataset)
            evaluator = COCOEvaluator(corrupt_dataset, output_dir=cfg.OUTPUT_DIR)
            evaluator.reset()
            pipeline.uncertainty_sampler.reset_stats()

            for batch_idx, batch in enumerate(data_loader):
                outputs, uploaded, uncertainty = pipeline.process_single_frame(batch)

                # detach + cpu for evaluation
                for o in outputs:
                    inst = o["instances"]
                    inst.pred_boxes.tensor = inst.pred_boxes.tensor.detach().cpu()
                    inst.scores = inst.scores.detach().cpu()
                    inst.pred_classes = inst.pred_classes.detach().cpu()
                evaluator.process(batch, outputs)

                if batch_idx % 100 == 0:
                    logger.info(
                        f"[CloudEdge] {corrupt_dataset}: {batch_idx}/{len(data_loader)} 帧 "
                        f"| 上传率: {pipeline.uncertainty_sampler.get_upload_rate():.1%} "
                        f"| 云端更新: {pipeline.cloud_update_count}"
                    )

            result = evaluator.evaluate()
            results[corrupt_dataset] = result
            if comm.is_main_process():
                logger.info(f"域 {corrupt_dataset} 评估结果: {result}")

    pipeline.print_stats()
    return results


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)

    if args.wandb and comm.is_main_process():
        try:
            import wandb
            wandb.init(project="ContinualTTA_CloudEdge", config=dict(cfg))
        except ImportError:
            logger.warning("wandb 未安装，跳过日志记录")

    if args.eval_only:
        results = run_evaluation(cfg, args)
    else:
        results = run_cloud_edge_adaptation(cfg, args)

    if comm.is_main_process():
        logger.info("最终结果:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    main()
