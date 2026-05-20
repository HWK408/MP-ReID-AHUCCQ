from utils.logger import setup_logger
from datasets.make_dataloader_uniprompt import make_dataloader
from model.make_model_uniprompt import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2astage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_uniprompt_stage1 import do_train_stage1
from processor.processor_uniprompt_stage2 import do_train_stage2
from processor.processor_uniprompt_stage2 import do_inference
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from peft import LoraConfig, TaskType, get_peft_model
from utils.output import resolve_output_dir

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def resolve_stage2_plan(cfg):
    curriculum_cfg = getattr(cfg.DATASETS, "CURRICULUM", None)
    if curriculum_cfg is None or not curriculum_cfg.ENABLED:
        return [("full", int(cfg.SOLVER.STAGE2.MAX_EPOCHS))]

    phases = list(curriculum_cfg.STAGE2_PHASES)
    epochs = [int(epoch) for epoch in curriculum_cfg.STAGE2_EPOCHS]
    if len(phases) != len(epochs):
        raise ValueError("DATASETS.CURRICULUM.STAGE2_PHASES and STAGE2_EPOCHS must have the same length.")
    return [(phase, epoch) for phase, epoch in zip(phases, epochs) if epoch > 0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")# 创建一个参数解析器对象
    parser.add_argument(
        "--config_file", default="configs/ours/cctv_ir_cctv_rgb.yml", help="path to config file", type=str
    )# 添加参数：配置文件路径  --config_file：指定配置文件的路径（通常是 YAML 格式）。

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)# 添加参数：命令行参数覆盖 (opts) opts：这是一个非常灵活的参数，用于在命令行直接修改配置里的具体数值。
    # nargs=argparse.REMAINDER：这是一个特殊的设置。它的意思是：“把所有剩余的参数都收集到一个列表里给 opts”。

    parser.add_argument("--local_rank", default=0, type=int)# 添加参数：分布式训练节点
    args = parser.parse_args()
    # add_expsetting_cfg(cfg)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)# 读取 YAML 文件的内容并更新到全局配置对象 cfg 中。
    cfg.merge_from_list(args.opts)#将 opts 里捕获的参数列表应用到配置中。 注意：这步在加载文件之后，说明命令行的优先级高于配置文件。
    cfg.OUTPUT_DIR = resolve_output_dir(cfg)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN: #False
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:#False
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, \
    num_query, num_classes, camera_num, view_num = make_dataloader(cfg, curriculum_phase="full")

    # Use original config for model creation, as stage1 does not use MoE anyway.
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # =================================================================================
    #                             Two-Stage Prompt Training
    # =================================================================================

    # --- Stage 1a: Train Generic Context ---
    logger.info("===== Configuring and starting Stage 1a training =====")
    # Note: Assumes config file has a `SOLVER.STAGE1A` section
    if cfg.MODEL.DIST_TRAIN:#False
        model.module.enable_stage1a_training()
    else:
        model.enable_stage1a_training()
    optimizer_1a = make_optimizer_1stage(cfg, model, stage_name='STAGE1A')
    scheduler_1a = create_scheduler(optimizer_1a, num_epochs=cfg.SOLVER.STAGE1A.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1A.LR_MIN,
                                    warmup_lr_init=cfg.SOLVER.STAGE1A.WARMUP_LR_INIT, warmup_t=cfg.SOLVER.STAGE1A.WARMUP_EPOCHS, noise_range=None)
    
    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1a,
        scheduler_1a,
        args.local_rank,
        stage_name='STAGE1A',
        is_stage1b=False
    )

    # --- Stage 1b: Train Domain-Specific Context ---
    logger.info("===== Configuring and starting Stage 1b training =====")
    # Note: Assumes config file has a `SOLVER.STAGE1B` section
    if cfg.MODEL.DIST_TRAIN:#False
        model.module.enable_stage1b_training()
    else:
        model.enable_stage1b_training()

    optimizer_1b = make_optimizer_1stage(cfg, model, stage_name='STAGE1B')
    scheduler_1b = create_scheduler(optimizer_1b, num_epochs=cfg.SOLVER.STAGE1B.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1B.LR_MIN,
                                    warmup_lr_init=cfg.SOLVER.STAGE1B.WARMUP_LR_INIT, warmup_t=cfg.SOLVER.STAGE1B.WARMUP_EPOCHS, noise_range=None)

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1b,
        scheduler_1b,
        args.local_rank,
        stage_name='STAGE1B',
        is_stage1b=True
    )
    
    # =================================================================================
    #                         End of Two-Stage Prompt Training
    # =================================================================================

    if cfg.MODEL.MOE.ENABLED:
        logger.warning(
            "MODEL.MOE.ENABLED=True was set, but current training pipeline runs Uni-Prompt without MoE. "
            "Skipping all MoE/Stage2b logic by design."
        )
    else:
        logger.info("Running pure Uni-Prompt pipeline (MoE/Stage2b disabled).")


    # --- Add this block to set requires_grad for Stage 2a ---
    logger.info("Setting parameter `requires_grad` for Stage 2a fine-tuning...")
    logger.info("Goal: Train model excluding text_encoder, experts-related, and prompt_learner params.")

    for name, param in model.named_parameters():
        # Default: Set trainable, then freeze specific parts
        param.requires_grad = True # Start by making everything trainable

        # Freeze text_encoder
        if 'text_encoder' in name:
            param.requires_grad = False
            continue # Skip to next parameter if frozen

        # Freeze parameters containing "experts" in their name
        if 'expert' in name:
            param.requires_grad = False
            continue # Skip to next parameter if frozen

        # Freeze prompt_learner (assuming it's not trained in Stage 2a)
        if 'prompt_learner' in name:
            param.requires_grad = False
            continue # Skip to next parameter if frozen


    # --- End of simplified block ---

    # Optional: Log the final trainable parameters
    # log_trainable_parameters(model, "After Merge & Simple Freeze for Stage 2a")

    # 2a stage, train based on the requires_grad settings above.
    stage2_plan = resolve_stage2_plan(cfg)
    curriculum_enabled = getattr(cfg.DATASETS.CURRICULUM, "ENABLED", False)
    logger.info("2a stage, train parameters marked as trainable...")
    logger.info("Stage 2 curriculum plan: {}".format(stage2_plan))

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2astage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )
    stage2_epoch_offset = 0

    for phase, phase_epochs in stage2_plan:
        if curriculum_enabled:
            logger.info("===== Stage 2 curriculum phase: {} ({} epochs) =====".format(phase, phase_epochs))
            train_loader_stage2, _, _, _, _, _, _ = make_dataloader(cfg, curriculum_phase=phase)

        do_train_stage2(
            cfg,
            model,
            center_criterion,
            train_loader_stage2,
            val_loader,
            optimizer_2stage,
            optimizer_center_2stage,
            scheduler_2stage,
            loss_func,
            num_query,
            args.local_rank,
            max_epochs=phase_epochs,
            log_period=cfg.SOLVER.STAGE2.LOG_PERIOD,
            checkpoint_period=min(cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD, phase_epochs),
            eval_period=min(cfg.SOLVER.STAGE2.EVAL_PERIOD, phase_epochs),
            stage_tag="stage2_{}".format(phase) if curriculum_enabled else None,
            epoch_offset=stage2_epoch_offset
        )
        stage2_epoch_offset += phase_epochs

    do_inference(
        cfg,
        model,
        val_loader,
        num_query
    )
