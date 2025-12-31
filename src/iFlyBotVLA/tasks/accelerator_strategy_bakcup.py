import math
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from abc import ABC
from tqdm import tqdm

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration, FullyShardedDataParallelPlugin,  DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig
)
from torch.optim import AdamW
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from utils.utils import check_bloat16_supported
from omegaconf import OmegaConf

from utils.overwatch import initialize_overwatch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from utils.metrics import VLAMetrics
import gc
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import time
from datetime import timedelta


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# default deepspeed configs
@dataclass
class DefaultDeepSpeedConfig:
    bf16: dict = field(default_factory=lambda: {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    })
    fp16: dict = field(default_factory=lambda: {
        "enabled": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    })
    zero_optimization: dict = field(default_factory=lambda: {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": False,
        "reduce_bucket_size": 1e9,
        "contiguous_gradients": True,
        "overlap_comm": True,
    })
    # zero_optimization: dict = field(default_factory=lambda: {
    #     "stage": 3,
    #     "allgather_partitions": True,
    #     "allgather_bucket_size": 5e8,        
    #     "reduce_bucket_size": 5e8,

    #     "overlap_comm": False,              
    #     "reduce_scatter": True,
    #     "contiguous_gradients": True,

    #     "stage3_prefetch_bucket_size": 5e7,
    #     "stage3_param_persistence_threshold": 1e5,

    #     # "stage3_max_live_parameters": 1e9,
    #     # "stage3_max_reuse_distance": 1e9,
    #     "stage3_gather_16bit_weights_on_model_save": True,
    # })
    train_batch_size:str = "auto"
    train_micro_batch_size_per_gpu:str = "auto"
    gradient_accumulation_steps:int = 4
    wall_clock_breakdown: bool = False
    find_unused_parameters = True 
    # comms_logger: dict= field(default_factory=lambda: {
    #     "enabled": True,
    #     "verbose": True,
    #     "prof_all": True,
    #     "debug": False
    #     })


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


class AcceleratorStrategy(ABC):
    def __init__(
        self,
        vla,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = False,
        enable_mixed_precision_training: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.vla = vla

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        # save vlm and action_head separately
        # we always save all modules
        if self.accelerator.is_main_process:
            
            model_state_dicts = {mkey: OrderedDict() for mkey in ["vlm", "action_head"]}

            vlm_unwrapped = self.accelerator.unwrap_model(self.vla.module.vlm)
            vlm_state_dict = vlm_unwrapped.state_dict()
            for key, param in vlm_state_dict.items():
                model_state_dicts["vlm"][key] = param
        
            action_head_unwrapped = self.accelerator.unwrap_model(self.vla.module.action_expert)
            action_head_state_dict = action_head_unwrapped.state_dict()
            for key, param in action_head_state_dict.items():
                model_state_dicts["action_head"][key] = param
                
            checkpoint_dir = run_dir / "checkpoints"
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = (
                    checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                )

            # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
            torch.save({"model": model_state_dicts}, checkpoint_path)
            shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, n_train_examples: int) -> None:
        

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
        self.deepspeed_config = OmegaConf.structured(DefaultDeepSpeedConfig)
        # self.deepspeed_config.train_batch_size=int(self.global_batch_size)
        self.deepspeed_config.gradient_accumulation_steps=self.grad_accumulation_steps
        # print("self.grad_accumulation_steps",self.grad_accumulation_steps)
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(self.deepspeed_config, resolve=True),
            gradient_accumulation_steps=self.grad_accumulation_steps,
            # train_batch_size=self.global_batch_size,
            # train_micro_batch_size_per_gpu=self.per_device_batch_size
        )
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,

            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision="bf16" if self.enable_mixed_precision_training else "no",
            kwargs_handlers=[ddp_kwargs, timeout]
        )
        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vla.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0
                
        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vla.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
            
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        overwatch.info(
            "DeepSpeed Training Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        self.accelerator.clip_grad_norm_(self.vla.parameters(), self.max_grad_norm)
    
    def train_step(
        self,
        metrics: VLAMetrics,
        batch_robot,
        batch_qa=None,
        action_loss_weight: float = 1.0,
        qa_loss_scale: float = 1.0,
    ):
        """
        """
        debug = False

        # ====== Forward (Robot) ======
        with torch.autocast(
            "cuda",
            dtype=self.mixed_precision_dtype,
            enabled=self.enable_mixed_precision_training,
        ):
            out = self.vla(batch_robot)  # 现在返回 dict

            action_loss = out["action_loss"]
            vlm_loss = out["vlm_loss"]
            vlm_output = out["vlm_output"]

        logits = vlm_output.logits if hasattr(vlm_output, "logits") else vlm_output["logits"]

        # action accuracy
        action_preds = logits[:, :-1].argmax(dim=2)
        action_gt = batch_robot["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > 0
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # ====== Commit metrics（只用 action_loss，保持与 VLAMetrics 兼容）======
        metrics.commit(action_loss=self.accelerator.gather(action_loss).mean())

        # ====== Loss (Robot) ======
        vla_loss = action_loss_weight * action_loss + vlm_loss
        normalized_vla_loss = vla_loss / self.grad_accumulation_steps

        self.accelerator.backward(normalized_vla_loss)

        if self.accelerator.is_main_process:
            print(
                f"global_step={metrics.global_step}  "
                f"action_loss={action_loss}  vlm_loss={vlm_loss}  action_accuracy={action_accuracy}"
            )

        # ====== Optional QA backward ======
        weighted_qa_loss = None
        if batch_qa is not None:
            batch_qa = {k: v.to("cuda") for k, v in batch_qa.items()}

            with torch.autocast(
                "cuda",
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision_training,
            ):
                qa_loss = self.vla.vlm_qa_forward(batch_qa)
                weighted_qa_loss = qa_loss * qa_loss_scale

            # QA loss 也参与累积（同样会被 grad_accumulation_steps 的节奏一起 step）
            self.accelerator.backward(weighted_qa_loss)

            if self.accelerator.is_main_process:
                print(f"global_step={metrics.global_step}  QA_weighted_loss={weighted_qa_loss}")

        # ====== Optimizer step at accumulation boundary ======
        if (metrics.global_step + 1) % self.grad_accumulation_steps == 0:
            self.clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return normalized_vla_loss, weighted_qa_loss


    def run_training(
        self,
        metrics: VLAMetrics,
        vla_dataloader: DataLoader,
        vqa_dataloader: Optional[DataLoader] = None,
        save_step: int = 2000,
        qa_every_n_steps: int = 1,
        action_loss_weight: float = 1.0,
    ) -> None:
        """

        """
        epoch = metrics.epoch if hasattr(metrics, "epoch") else 0

        # prepare：QA dataloader 可选
        if vqa_dataloader is not None:
            self.vla, self.optimizer, self.lr_scheduler, _, _ = self.accelerator.prepare(
                self.vla, self.optimizer, self.lr_scheduler, vla_dataloader, vqa_dataloader
            )
            vqa_iter = infinite_loader(vqa_dataloader)
        else:
            self.vla, self.optimizer, self.lr_scheduler, _ = self.accelerator.prepare(
                self.vla, self.optimizer, self.lr_scheduler, vla_dataloader
            )
            vqa_iter = None

        # 清梯度：先清一次
        self.optimizer.zero_grad(set_to_none=True)

        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(vla_dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vla.train()

            for step, batch_robot in enumerate(vla_dataloader, start=1):
                # === 在一个 accumulation 窗口开始时清梯度 ===
                # grad_accumulation_steps == 1：每步都会清，然后 train_step 里每步都会 step -> 正常等价非累积
                # grad_accumulation_steps > 1：只在窗口开始清一次，其余 micro-step 累加
                if metrics.global_step % self.grad_accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                if "action_chunk" in batch_robot:
                    batch_robot["actions"] = batch_robot["action_chunk"]
                if "state" in batch_robot:
                    batch_robot["proprios"] = batch_robot["state"]

                batch_robot = {k: v.to("cuda") for k, v in batch_robot.items()}

                # 默认不混 QA：只有 vqa_dataloader 存在才取 batch_qa
                batch_qa = None
                if vqa_iter is not None and (step % qa_every_n_steps == 0):
                    batch_qa = next(vqa_iter)

                vla_loss, qa_loss = self.train_step(
                    metrics=metrics,
                    batch_robot=batch_robot,
                    batch_qa=batch_qa,
                    action_loss_weight=action_loss_weight,
                    qa_loss_scale=1.0,
                )

                # 每个 micro-step global_step + 1
                metrics.commit(
                    global_step=metrics.global_step + 1,
                    lr=self.lr_scheduler.get_last_lr()[0],
                    epoch=epoch,
                )

                # 只在 accumulation 边界 push/progress.update
                if metrics.global_step % self.grad_accumulation_steps == 0:
                    status = metrics.push()

                    # max_steps 终止（按micro-step 计数）
                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, vla_loss.item())
                        dist.barrier()
                        return

                    progress.update()
                    progress.set_description(status)

                if metrics.global_step % 1000 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # 
                previous_epoch = epoch
                epoch = (metrics.global_step + 1) // (len(dataset) // self.global_batch_size)

                terminate = (self.max_steps is not None and metrics.global_step >= self.max_steps)
                if terminate or epoch > previous_epoch or metrics.global_step == 1 or (metrics.global_step + 1) % save_step == 0:
                    self.save_checkpoint(
                        metrics.run_dir,
                        metrics.global_step,
                        epoch,
                        self.accelerator.gather(vla_loss).mean().item(),
                    )
                    dist.barrier()
                    del vla_loss

                    if terminate:
                        return


