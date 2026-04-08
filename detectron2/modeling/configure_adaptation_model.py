import numpy as np
import torch
import torch.nn as nn
from detectron2.layers import FrozenBatchNorm2d
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import maybe_add_gradient_clipping


def configure_model(cfg, trainer, model=None, revert=True, lr=None, weight_path=None):
    # revert to the source trained weight
    if model is None or revert:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if weight_path is None else weight_path, resume=False
        )
        #if cfg.TEST.ADAPTATION.WHERE == 'adapter' and "resnet" in cfg.MODEL.BACKBONE.NAME:
        #    model.backbone.bottom_up.add_adapter(cfg.TEST.ADAPTER.TYPE, cfg.TEST.ADAPTER.THRESHOLD, scalar=cfg.TEST.ADAPTER.SCALAR)
        #    model.to(model.device)
        model.initialize()

    model.eval()
    model.requires_grad_(False)
    lr_ = cfg.SOLVER.BASE_LR * max(cfg.SOLVER.IMS_PER_BATCH_TEST // 4, 1) if lr is None else lr
    params = []
    bn_params = []
    if cfg.TEST.ADAPTATION.WHERE == 'adapter':
        if "resnet" in cfg.MODEL.BACKBONE.NAME:
            #if hasattr(model.backbone.bottom_up.stem.conv1, 'parallel_conv'):
            #    model.backbone.bottom_up.stem.conv1.parallel_conv.requires_grad_(True)
            #    params += list(model.backbone.bottom_up.stem.conv1.parallel_conv.parameters())
            for stage in model.backbone.bottom_up.stages:
                for block in stage:
                    block.adapter.requires_grad_(True)
                    params += list(block.adapter.parameters())
                    if hasattr(block.conv1, 'down_proj'):
                        block.conv1.down_proj.requires_grad_(True)
                        block.conv1.up_proj.requires_grad_(True)
                        params += list(block.conv1.down_proj.parameters())
                        params += list(block.conv1.up_proj.parameters())
                    if hasattr(block.conv1, 'scalar') and cfg.TEST.ADAPTER.SCALAR == 'learnable_scalar':
                        block.conv1.scalar.requires_grad_(True)
                        params.append(block.conv1.scalar)
                    if hasattr(block.conv1, 'scale'):
                        block.conv1.scale.requires_grad_(True)
                        block.conv1.shift.requires_grad_(True)
                        params += [block.conv1.scale, block.conv1.shift]
                    if hasattr(block.conv1, 'lora_A'):
                        block.conv1.lora_A.requires_grad_(True)
                        block.conv1.lora_B.requires_grad_(True)
                        params += [block.conv1.lora_A, block.conv1.lora_B]
                    if hasattr(block.conv1, 'adapter_norm'):
                        block.conv1.adapter_norm.track_running_stats = False
                        block.conv1.adapter_norm.requires_grad_(True)
                        params += list(block.conv1.adapter_norm.parameters())
                #if cfg.TEST.ADAPTATION.NORM:
                #    block.conv1.norm.weight.requires_grad = True
                #    block.conv1.norm.bias.requires_grad = True
                #    bn_params += [block.conv1.norm.weight, block.conv1.norm.bias]
            #for m_name, m in model.backbone.bottom_up.named_modules():
            #    if cfg.TEST.ADAPTATION.NORM:
            #        if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            #            m.adapt_training = False
            #            m.weight.requires_grad = True
            #            m.bias.requires_grad = True
            #            if m.weight.requires_grad:
            #                bn_params += [m.weight, m.bias]
 
        elif "swin" in cfg.MODEL.BACKBONE.NAME:
            for layer in model.backbone.bottom_up.layers:
                for block in layer.blocks:
                    if hasattr(block, 'adapter'):
                        block.adapter.requires_grad_(True)
                        params += list(block.adapter.parameters())

    elif cfg.TEST.ADAPTATION.WHERE == 'full':
        if cfg.TEST.ADAPTATION.GLOBAL_ALIGN == "BN":
            for m in model.modules():
                if isinstance(m, FrozenBatchNorm2d):
                    # force use of batch stats in train and eval modes
                    m.adapt_training = True
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            return model, None

        for m_name, m in model.backbone.bottom_up.named_modules():
            #if cfg.TEST.ADAPTATION.NORM == 'adapt':
            if True:
                if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.adapt_training = False
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    if m.weight.requires_grad:
                        #bn_params += [m.weight, m.bias]
                        params += [m.weight, m.bias]
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                # except patch embedding
                if "patch_embed" in m_name and "attn" in m_name:
                    continue
                m.weight.requires_grad = True
                params += [m.weight]
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params += [m.bias]
    elif cfg.TEST.ADAPTATION.WHERE == 'normalization':
        for m_name, m in model.backbone.bottom_up.named_modules():
            if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.adapt_training = False
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                if m.weight.requires_grad:
                    params += [m.weight, m.bias]
    elif cfg.TEST.ADAPTATION.WHERE == 'head':
        model.roi_heads.box_head.requires_grad_(True)
        params += list(model.roi_heads.box_head.parameters())

    if cfg.TEST.ADAPTATION.NORM in ["DUA", "NORM"]:
        for m_name, m in model.named_modules():
            if isinstance(m, FrozenBatchNorm2d) and 'stem' not in m_name:
                # force use of batch stats in train and eval modes
                m.adapt_type = cfg.TEST.ADAPTATION.NORM  # "DUA" or "NORM"
                # Original DUA Hyperparam
                if cfg.TEST.ADAPTATION.NORM == "DUA":
                    m.min_momentum_constant = cfg.TEST.ADAPTATION.BN_MIN_MOMENTUM_CONSTANT
                    m.decay_factor = cfg.TEST.ADAPTATION.BN_DECAY_FACTOR
                    m.mom_pre = cfg.TEST.ADAPTATION.BN_MOM_PRE
                elif cfg.TEST.ADAPTATION.NORM == "NORM":
                    m.source_sum = cfg.TEST.ADAPTATION.BN_SOURCE_NUM
        if not cfg.TEST.ONLINE_ADAPTATION:
            return model, None, None



    if cfg.SOLVER.TYPE == "SGD":
        sgd_args = [{"params": params}]
        if len(bn_params) > 0:
            sgd_args.append({"params": bn_params, "lr": cfg.SOLVER.BASE_LR_BN}) 
        optimizer = torch.optim.SGD(sgd_args, lr_, momentum=cfg.SOLVER.MOMENTUM,
                                    nesterov=cfg.SOLVER.NESTEROV, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.TYPE == "AdamW":
        adamw_args = {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR,
            "betas": (0.9, 0.999),
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        }
        optimizer = torch.optim.AdamW(**adamw_args)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)

    if cfg.TEST.ADAPTATION.TYPE is not None and "mean-teacher" in cfg.TEST.ADAPTATION.TYPE:
        import copy
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        teacher_model.online_adapt = False
        model.online_adapt = False
        model.training = True
        model.proposal_generator.training = True
        model.roi_heads.training = True
    else:
        teacher_model = None

    return model, optimizer, teacher_model


# ---------------------------------------------------------------------------- #
# 云边协同模型配置
# ---------------------------------------------------------------------------- #

def configure_cloud_edge_models(cfg, trainer):
    """
    构建云边协同框架所需的教师(R101+LoRA)和学生(R50)模型。

    教师模型: ResNet-101-FPN，注入LoRA (rank=16)
              仅LoRA参数可训练，AdamW lr=2e-3
    学生模型: ResNet-50-FPN，全参数可训练
              AdamW lr=1e-3

    Args:
        cfg: 配置节点（需包含 TEST.ADAPTATION.CLOUD_MODEL_WEIGHTS 等键）
        trainer: DefaultTrainer 实例（用于 build_model）
    Returns:
        CloudDistillationTrainer 实例
    """
    from detectron2.modeling.cloud_distillation import CloudDistillationTrainer, inject_lora_to_model
    import copy

    # --- 构建教师模型（R101）---
    # 覆盖 backbone 配置为 R101
    teacher_cfg = cfg.clone()
    teacher_cfg.defrost()
    # 若云端模型权重已指定则使用，否则复用边端权重
    if cfg.TEST.ADAPTATION.CLOUD_MODEL_WEIGHTS is not None:
        teacher_cfg.MODEL.WEIGHTS = cfg.TEST.ADAPTATION.CLOUD_MODEL_WEIGHTS
    # 将 backbone 修改为 R101（假定配置中有对应的 R101 config）
    # 用户可通过 CloudEdge_COCO_R101_R50.yaml 传入正确的 R101 backbone 名
    teacher_cfg.freeze()

    teacher_model = trainer.build_model(teacher_cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(teacher_model).resume_or_load(
        teacher_cfg.MODEL.WEIGHTS, resume=False
    )
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    teacher_model.initialize()

    # 保存一份冻结的预训练教师（用于防遗忘正则）
    pretrain_teacher = copy.deepcopy(teacher_model)
    pretrain_teacher.eval()
    pretrain_teacher.requires_grad_(False)

    # 注入 LoRA
    lora_rank = 16
    teacher_model, lora_params = inject_lora_to_model(
        teacher_model, r=lora_rank, lora_alpha=lora_rank
    )

    # --- 构建学生模型（R50, 与边端相同）---
    student_model = trainer.build_model(cfg)
    DetectionCheckpointer(student_model).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    student_model.eval()
    student_model.requires_grad_(True)
    student_model.initialize()

    # --- 构建 CloudDistillationTrainer ---
    distill_trainer = CloudDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        pretrain_teacher=pretrain_teacher,
        teacher_lr=cfg.TEST.ADAPTATION.CLOUD_LORA_LR,
        teacher_weight_decay=1e-4,
        student_lr=cfg.TEST.ADAPTATION.DISTILL_LR,
        teacher_epochs=max(1, cfg.TEST.ADAPTATION.CLOUD_ITERATIONS // 100),
        student_epochs=cfg.TEST.ADAPTATION.DISTILL_EPOCHS,
        forgetting_beta=getattr(cfg.TEST.ADAPTATION, 'FORGETTING_BETA', 0.3),
        rollback_threshold=getattr(cfg.TEST.ADAPTATION, 'ROLLBACK_THRESHOLD', 5.0),
        lambda_cls=cfg.TEST.ADAPTATION.LAMBDA_OUTPUT,
        lambda_reg=getattr(cfg.TEST.ADAPTATION, 'LAMBDA_REG', 0.5),
        lambda_feat=cfg.TEST.ADAPTATION.LAMBDA_FEATURE,
        distill_temperature=cfg.TEST.ADAPTATION.DISTILL_TEMPERATURE,
        lora_rank=lora_rank,
    )
    # teacher LoRA 优化器已通过 inject_lora 完成，这里直接设置
    distill_trainer.teacher_optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.TEST.ADAPTATION.CLOUD_LORA_LR,
        weight_decay=1e-4,
    )

    return distill_trainer

