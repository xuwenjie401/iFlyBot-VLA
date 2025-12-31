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

from genie.modules import UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class DINO_LAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            mse_weight: float = 1.0,
            q_weight: float = 1.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
    ) -> None:
        super(DINO_LAM, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        lam = UncontrolledDINOLatentActionModel if stage == 'stage-1' else ControllableDINOLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
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

        self.lam_num_latents = lam_num_latents
        self.mse_weight = mse_weight
        self.q_weight = q_weight
        self.vq_beta = vq_beta
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
        #     if param.grad is not None:
        #         print(f"{name}: grad strides = {param.grad.stride()}")
        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]
        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = self.mse_weight * mse_loss + self.q_weight*(q_loss + self.vq_beta * commit_loss)
        
        # Optimize uncontrollable queries in stage-2 (the codebook is frozen though)
        if "z_q_uncontrol" in outputs.keys():
            q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
            commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
            loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        if torch.isnan(loss):
            self.nan_count += 1
        loss = torch.nan_to_num(loss, nan=0.0) 

        # Compute code usage        
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        code_usage_batch = []

        for i in range(4):
            output = outputs["indices"][i]
            unique, counts = torch.unique(output, return_counts=True)
            index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
            index_counts[unique] = counts
            code_usage_onebatch = (index_counts != 0).float().mean()
            code_usage_batch.append(code_usage_onebatch)

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
            # ("dino_feature_min", outputs["patches"][:,0].min()),
            # ("dino_feature_max", outputs["patches"][:,0].max()),
            # ("action_latent_min", outputs["action_latent"].min()),
            # ("action_latent_max", outputs["action_latent"].max()),
            # ("lang_embed_min", outputs["lang"].min()),
            # ("lang_embed_max", outputs["lang"].max()),
            # ("lang_min", outputs["lang_emb"].min()),
            # ("lang_max", outputs["lang_emb"].max()),
            # ("emb_1_min", outputs["emb"].min()),
            # ("emb_1_max", outputs["emb"].max()),
            # ("codebook_1_min", outputs["z_q"].min()),
            # ("codebook_1_max", outputs["z_q"].max()),
            # ("dino_feature_next_min", outputs["target"].min()),
            # ("dino_feature_next_max", outputs["target"].max()),
            # ("decoder_min", outputs["recon"].min()),
            # ("decoder_max", outputs["recon"].max()),
            
            # ("code_usage0", code_usage_batch[0]),
            # ("code_usage1", code_usage_batch[1]),
            # ("code_usage2", code_usage_batch[2]),
            # ("code_usage3", code_usage_batch[3]),
        )

        if "indices_uncontrol" in outputs.keys():
            unique, counts = torch.unique(outputs["indices_uncontrol"], return_counts=True)
            index_counts = torch.zeros(32, dtype=torch.long).cuda()
            index_counts[unique] = counts
            uncontrol_code_usage = (index_counts != 0).float().mean()

            loss_logs = (
                ("mse_loss", mse_loss),
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("q_loss_uncontrol", q_loss_uncontrol),
                ("commit_loss_uncontrol", commit_loss_uncontrol),
                ("code_usage", code_usage),
                ("code_usage_uncontrol", uncontrol_code_usage),
                # ("dino_feature_min", outputs["patches"][:,0].min()),
                # ("dino_feature_max", outputs["patches"][:,0].max()),
                # ("action_latent_min", outputs["action_latent"].min()),
                # ("action_latent_max", outputs["action_latent"].max()),
                # ("action_latent_controllable_min", outputs["action_latent_controllable"].min()),
                # ("action_latent_controllable_max", outputs["action_latent_controllable"].max()),
                # ("emb_2_min", outputs["emb"].min()),
                # ("emb_2_max", outputs["emb"].max()),
                # ("emb_1_min", outputs["emb_uncontrol"].min()),
                # ("emb_1_max", outputs["emb_uncontrol"].max()),
                # ("codebook_2_min", outputs["z_q"].min()),
                # ("codebook_2_max", outputs["z_q"].max()),
                # ("codebook_1_min", outputs["z_q_uncontrol"].min()),
                # ("codebook_1_max", outputs["z_q_uncontrol"].max()),
                # ("dino_feature_next_min", outputs["target"].min()),
                # ("dino_feature_next_max", outputs["target"].max()),
                # ("decoder_min", outputs["recon"].min()),
                # ("decoder_max", outputs["recon"].max()),
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

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
