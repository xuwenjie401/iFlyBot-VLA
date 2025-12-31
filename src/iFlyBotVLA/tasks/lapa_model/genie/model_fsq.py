from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

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

from genie.modules import UncontrolledfsqModel, ControllablefsqModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class LAM_fsq(LightningModule):
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
            level: list=[3,3,3],
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
    ) -> None:
        super(LAM_fsq, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        lam = UncontrolledfsqModel if stage == 'stage-1' else ControllablefsqModel

        self.lam = lam(
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
            if stage == 'stage-1':
                self.lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=False)
            else:
                # self.lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=False)
                # for key in lam_ckpt.keys():
                #     if 'vq' in key:
                #         stage1_ckpt[key.replace("lam.vq", "vq_action")] = lam_ckpt[key]
                #     elif 'action_latent' in key:
                #         stage1_ckpt[key.replace("lam.action_latent", "action_latent_controllable")] = lam_ckpt[key]
                #     elif 'action_up' in key:
                #         stage1_ckpt[key.replace("lam.action_up", "action_up_uncontrol")] = lam_ckpt[key]
                #     elif 'to_codebook' in key:
                #         stage1_ckpt[key.replace("lam.to_codebook", "to_codebook_uncontrol")] = lam_ckpt[key]
                # self.lam.load_state_dict(stage1_ckpt, strict=False)

                for key in lam_ckpt.keys():
                    if 'vq' in key or 'action_latent' in key:
                        stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
                self.lam.load_state_dict(stage1_ckpt, strict=False)

        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()

        self.nan_count = 0
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

        if torch.isnan(loss):
            self.nan_count += 1
        loss = torch.nan_to_num(loss, nan=0.0) 

        unique = outputs["indices"].unique().size(0)

        loss_logs = (
            ("unique_code", unique),
            ("in_max", outputs["in_max"]),
            ("in_min", outputs["in_min"]),
        )

        if "indices_uncontrol" in outputs.keys():
            unique_uncontrol = outputs["indices_uncontrol"].unique().size(0)

            loss_logs = (
                ("unique_code", unique),
                ("unique_code_uncontrol", unique_uncontrol),
                ("in_max", outputs["in_max"]),
                ("in_min", outputs["in_min"]),
            )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)
        
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
