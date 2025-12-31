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
import psutil

import transformers
from transformers import AutoProcessor

from utils.overwatch import initialize_overwatch
from utils.utils import set_global_seed
from config.base_configs import CommonArguments,ModelArguments, TrainingArguments, \
    TaskArguments, DatasetsArguments, ManipDataArguments, VQADataArguments, OverallArguments

from datasets.data_loader import make_embodied_data_module, make_vqa_data_module

overwatch = initialize_overwatch(__name__)

def test_data(cfg: OverallArguments):
    overwatch.info(f"iFlyBotVLA Training Start ......")

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    training_hf_args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    worker_init_fn = set_global_seed(cfg.model.seed, get_worker_init_fn=True)

    # data ==============================================================================

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.vlm_model_path,
        cache_dir=None,
        model_max_length=4096,
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
        
    DEBUG_MEMORY = False
    
    N = len(robot_data_modules["train_dataset"])
    process = psutil.Process(os.getpid())
    for i, batch in tqdm(enumerate(vla_dataloader), total=N):
        # print(f"\n=== Batch {i} ===")
        
        state = batch['state']
        action = batch['action_chunk']

        breakpoint()

        # if cfg.datasets.enable_vqa_cotrain:
        #     vqa_batch = next(vqa_iter)

        if DEBUG_MEMORY:
            if i % 500 == 0:
                rss = process.memory_info().rss
                print(f"\n{i} iterations, batch-{training_hf_args.batch_size}: \nRSS: {rss / 1024 / 1024:.2f} MB")

            if i >= 5000:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="src/iFlyBotVLA/config/presets/debug_data.yaml")
    args = parser.parse_args()
    
    base = OmegaConf.structured(OverallArguments)
    override = OmegaConf.load(args.config_yaml)
    
    merged = OmegaConf.merge(base, override)
    
    # 用dataclass   先转dict
    obj = OmegaConf.to_object(merged)
    cfg = obj
    
    # check configs
    print(f"cfg from loaded file: {args.config_yaml}:\n{cfg}")
    
    test_data(cfg)
