from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState
from tasks.lapa_model.genie.lapa.latent_action_quantization import LatentActionQuantization

OptimizerCallable = Callable[[Iterable], Optimizer]

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class LAPA(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_ckpt: str = None,
    ) -> None:
        super(LAPA, self).__init__()

        self.lam = LatentActionQuantization(
            dim = 1024,
            quant_dim=32,
            codebook_size = 32, #8
            image_size = 224,
            patch_size = 32,
            spatial_depth = 8,
            temporal_depth = 8,
            dim_head = 64,
            heads = 16,
            code_seq_len=8 #4
        ).cuda()
        
        if stage_ckpt and path.exists(stage_ckpt):
            lam_ckpt = torch.load(stage_ckpt)['state_dict']
            stage1_ckpt = {}
            self.lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=False)

        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.distributed_state = PartialState()

        self.nan_count = 0
        self.steps = 0
        # if self.distributed_state.is_main_process:
        #     wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']
        # import pdb
        # pdb.set_trace()
        loss, num_unique_indices = self.lam(batch['videos'], step=self.steps)
        self.steps += 1
        if torch.isnan(loss):
            self.nan_count += 1
        # loss = torch.nan_to_num(loss, nan=0.0) 
        # for name, param in self.lam.named_parameters():
        #     if not param.requires_grad:
        #         print(f"冻结参数: {name}")
        #     if param.grad is None and param.requires_grad:
        #         print(f"未更新参数: {name}")

        loss_logs = (
            ("num_unique_indices", num_unique_indices),           
        )

        return loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        loss, aux_losses = self.shared_step(batch)
        
        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}, **{"nan_count": self.nan_count}},
            prog_bar=True,
            logger=True,
            on_step=True,
            # on_epoch=True,
            on_epoch=False,
            sync_dist=True
        )

        # if self.distributed_state.is_main_process:
        #     wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
