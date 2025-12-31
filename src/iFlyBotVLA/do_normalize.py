import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

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

import transformers
from transformers import AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from utils.overwatch import initialize_overwatch
from utils.utils import set_global_seed
from config.base_configs import CommonArguments,ModelArguments, TrainingArguments, \
    TaskArguments, DatasetsArguments, ManipDataArguments, VQADataArguments, OverallArguments

from datasets.data_loader import make_embodied_data_module
from utils.metrics import VLAMetrics

def main(cfg: OverallArguments = OverallArguments()):
    manip_cfg = ManipDataArguments()
    manip_cfg.dataset_use = "normalize"
    
    manip_cfg.action_chunk_size = 30
    manip_cfg.action_duration = 1

    # manip_cfg.action_style = "joint"
    # manip_cfg.action_style = "eef_pose"
    manip_cfg.action_style = "delta_eef"

    # manip_cfg.control_reference = "anchored_relative"
    manip_cfg.control_reference = "absolute"

    manip_cfg.timestep_conditioned_norm = False
    # manip_cfg.timestep_conditioned_norm = True

    manip_cfg.image_pad_other_view = False
    manip_cfg.mask_virtual_view = True
    
    cfg.datasets.manip_data = manip_cfg
    
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
        normalize_only=True,
    )
    
    
if __name__ == "__main__":
    main()

