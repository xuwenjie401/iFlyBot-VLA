import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
check_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(f"check path : {check_path}")
sys.path.append(check_path)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn as nn
import warnings

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from datetime import datetime
from dataclasses import asdict

import transformers
from transformers import AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from utils.overwatch import initialize_overwatch
from utils.utils import set_global_seed
from config.base_configs import CommonArguments,ModelArguments, TrainingArguments, \
    TaskArguments, DatasetsArguments, ManipDataArguments, VQADataArguments, OverallArguments
from model.iFlyBotVLA import iFlyBotVLA
from tasks.accelerator_strategy import AcceleratorStrategy

from datasets.data_loader import make_embodied_data_module, make_vqa_data_module
from utils.metrics import VLAMetrics
from utils.utils import to_json_safe

overwatch = initialize_overwatch(__name__)

def get_world_size(default=1):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return default


def train(cfg: OverallArguments):
    overwatch.info(f"iFlyBotVLA Training Start ......")

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    training_hf_args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    torch.cuda.set_device(overwatch.local_rank())
    torch.cuda.empty_cache()
    
    run_id = cfg.common.run_id
    run_root = cfg.common.run_root_dir
    
    use_lap_aux = cfg.task.use_lap_aux
    use_fast_aux = cfg.task.use_fast_aux
    domain_aware = cfg.task.domain_aware
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    robot_dataset_use = cfg.datasets.manip_data.dataset_use
    action_style = cfg.datasets.manip_data.action_style
    control_reference = cfg.datasets.manip_data.control_reference
    
    overall_run_id = f"{run_id}_{timestamp}_lap-{use_lap_aux}_fast-{use_fast_aux}_domain-{domain_aware}"
    overall_run_id += f"_{robot_dataset_use}_{action_style}_{control_reference}"
    
    worker_init_fn = set_global_seed(cfg.model.seed, get_worker_init_fn=True)
    
    os.makedirs(run_dir := (run_root / overall_run_id), exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    overwatch.info(f"Run_dir: {run_dir}")
    if overwatch.is_rank_zero():
        # 把cfg保存为Json
        with open(run_dir / "config.json", "w") as f:
            json.dump(to_json_safe(cfg), f, indent=4)
    
    # load 模型 =========================================================================
    vla = iFlyBotVLA(cfg=cfg, training_hf_args=training_hf_args)
    if cfg.model.load_from_checkpoint:
        overwatch.info(f"Load checkpoint from {cfg.model.model_checkpoint}")
        vla.load_from_checkpoint(cfg.model.model_checkpoint, vlm_only=cfg.model.load_only_vlm)
    # 进行参数冻结 & lora配置
    vla.check_freeze_and_lora()

    # data ==============================================================================
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.vlm_model_path,
        cache_dir=None,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    image_processor = AutoProcessor.from_pretrained(
        cfg.model.vlm_model_path,
    ).image_processor
    robot_data_modules = make_embodied_data_module(
        tokenizer=tokenizer,
        image_processor=image_processor,
        cfg=cfg,
    )
    vla_dataset = robot_data_modules["train_dataset"]
    
    vla_dataloader = DataLoader(
        vla_dataset,
        batch_size=training_hf_args.batch_size,
        sampler=None,
        collate_fn=robot_data_modules["data_collator"],
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    
    vqa_dataloader = None
    if cfg.datasets.enable_vqa_cotrain:
        vqa_data_modules = make_vqa_data_module(
            tokenizer=tokenizer,
            image_processor=image_processor,
            cfg=cfg,
        )
        vqa_dataloader = DataLoader(
            vqa_data_modules["train_dataset"],
            batch_size=cfg.datasets.vqa_data.vqa_batch_size
        )
        
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    action_expert_num_params = sum(p.numel() for p in vla.action_expert.parameters())
    overwatch.info(
        f"# Parameters of VLA (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable, {action_expert_num_params / 10**6:.3f} of which are in the Action Head"
    )

    world_size = get_world_size()
    global_batch_size = training_hf_args.batch_size * world_size
    
    train_strategy = AcceleratorStrategy(
        vla=vla,
        epochs=10,
        max_steps=200000,
        global_batch_size=global_batch_size,
        per_device_batch_size=training_hf_args.batch_size,
        learning_rate=training_hf_args.learning_rate,
        weight_decay=training_hf_args.weight_decay,
        max_grad_norm=training_hf_args.max_grad_norm,
        lr_scheduler_type=training_hf_args.lr_scheduler_type,
        warmup_ratio=training_hf_args.warmup_ratio,
        enable_gradient_checkpointing=training_hf_args.enable_gradient_checkpointing,
        enable_mixed_precision_training=training_hf_args.enable_mixed_precision_training,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(n_train_examples=len(vla_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.model.trackers}`")
    metrics = VLAMetrics(
        cfg.model.trackers,
        cfg.common.run_id,
        run_dir,
        wandb_project=cfg.model.wandb_project,
        resume_step=cfg.model.resume_step,
        resume_epoch=cfg.model.resume_epoch,
        window_size=cfg.datasets.manip_data.action_chunk_size,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Latent Action Training Loop")
    train_strategy.run_training(
        metrics=metrics,
        vla_dataloader=vla_dataloader,
        vqa_dataloader=vqa_dataloader,
        dataset_length=len(vla_dataset),
        save_step=cfg.model.save_interval,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="src/iFlyBotVLA/config/presets/iflybotvla_pretrain_default.yaml")
    args = parser.parse_args()
    
    base = OmegaConf.structured(OverallArguments)
    override = OmegaConf.load(args.config_yaml)
    
    merged = OmegaConf.merge(base, override)
    
    # 用dataclass   先转dict
    obj = OmegaConf.to_object(merged)
    cfg = obj
    
    # check configs
    print(f"cfg from loaded file: {args.config_yaml}:\n{cfg}")
    
    train(cfg)
