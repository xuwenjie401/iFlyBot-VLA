from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import GO1fsqModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class GO1_fsq(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            level: list=[8,8,6,5],
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
    ) -> None:
        super(GO1_fsq, self).__init__()
        assert stage in ['stage-1']

        self.lam = GO1fsqModel(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                    level=level,
                )
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            self.lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=False)

        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        # if self.distributed_state.is_main_process:
        #     wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']
        # import pdb
        # pdb.set_trace()
        # for name, param in self.lam.named_parameters():
        #     if param.grad is not None and "to_codebook" in name:
        #         print(f"{name}: {param.grad}")
        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]
        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()

        loss = mse_loss
        # Compute code usage        
        unique = outputs["indices"].unique().size(0)

        loss_logs = (
            ("unique_code", unique),
            ("in_max", outputs["in_max"]),
            ("in_min", outputs["in_min"]),
            ("query_max", outputs["action_latent"].max()),
            ("query_min", outputs["action_latent"].min()),
        )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )

        # if self.distributed_state.is_main_process:
        #     wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

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
