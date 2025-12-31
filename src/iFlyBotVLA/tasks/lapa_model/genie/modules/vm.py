import torch
import torch.nn as nn
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple
import numpy as np
import piq
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
OptimizerCallable = Callable[[Iterable], Optimizer]

class VelocityModel(nn.Module):
    def __init__(
            self, 
            in_dim: int=1024, 
            hidden_dim: int=256, 
            hidden2_dim: int=64, 
            out_dim: int=7):
        super().__init__()
        self.siglip = AutoModel.from_pretrained("/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/siglip-base-patch16-224")
        self.siglip.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained("/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/siglip-base-patch16-224")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, out_dim)     
        )

    def forward(self, front_img): 
        front_input = self.processor(images=front_img, return_tensors="pt", padding=True).to(self.siglip.device) 
        with torch.no_grad():
            front_feat = self.siglip.get_image_features(**front_input)
            
        output = self.mlp(front_feat)
        return output
    # def forward(self, front_img, left_img, right_img): 
    #     front_input = self.processor(images=front_img, return_tensors="pt", padding=True).to(self.siglip.device) 
    #     left_input = self.processor(images=left_img, return_tensors="pt", padding=True).to(self.siglip.device) 
    #     right_input = self.processor(images=right_img, return_tensors="pt", padding=True).to(self.siglip.device) 
    #     with torch.no_grad():
    #         front_feat = self.siglip.get_image_features(**front_input)
    #         left_feat = self.siglip.get_image_features(**left_input)
    #         right_feat = self.siglip.get_image_features(**right_input)
        
    #     combined_feat = torch.cat([front_feat, left_feat, right_feat], dim=1)
    #     output = self.mlp(combined_feat)
    #     return output

# model = VelocityModel().cuda()
# front_path = "/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/latent_action_model/images/OXE/debug_image_000.jpg"
# front_img = Image.open(front_path).convert('RGB') 
# left_path = "/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/latent_action_model/images/OXE/debug_image_001.jpg"
# left_img = Image.open(left_path).convert('RGB')
# right_path = "/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/latent_action_model/images/OXE/debug_image_002.jpg"
# right_img = Image.open(right_path).convert('RGB')
# pred_vel = model(front_img, left_img, right_img)
# print(pred_vel.shape)  

class VM(LightningModule):
    def __init__(
            self, 
            image_channels: int,
            in_dim: int, 
            hidden_dim: int, 
            hidden2_dim: int, 
            out_dim: int,
            log_interval: int = 1000,
            log_path: str = "./logs",
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_ckpt: str = None,
    ) -> None:
        super(VM, self).__init__()
        self.vel_model = VelocityModel(
            in_dim = in_dim, 
            hidden_dim = hidden_dim, 
            hidden2_dim = hidden2_dim, 
            out_dim = out_dim,
            ).cuda()

        if stage_ckpt and path.exists(stage_ckpt):
            model_ckpt = torch.load(stage_ckpt)['state_dict']
            self.vel_model.load_state_dict(model_ckpt, strict=False)

        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair
        self.save_hyperparameters()


    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']
        import pdb
        pdb.set_trace()
        outputs = self.vel_model(batch['videos'][:, 0])
        # for name, param in self.vel_model.named_parameters():
        #     if param.requires_grad:
        #         print(param.grad)
        # for name, param in self.vel_model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"冻结参数: {name}")
        #     else:
        #         print(f"可训练参数: {name}")
        grounded_vel = batch['action'][:,1,:6]-batch['action'][:,0,:6]
        # Compute loss
        loss = ((grounded_vel - outputs) ** 2).mean()
        loss_logs = (("mse_loss", loss))
        return outputs, loss, loss_logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)
        
        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss},},
            prog_bar=True,
            logger=True,
            on_step=True,
            # on_epoch=True,
            on_epoch=False,
            sync_dist=True
        )

        return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}},
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
