from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from tasks.lapa_model.genie.modules.blocks_go1_fsq import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, FSQ, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GO1fsqModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            level,
            dropout: float = 0.0
    ) -> None:
        super(GO1fsqModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.level = level
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder = torch.hub.load(
            repo_or_dir='/dmx-b3-ssd01/sppro/permanent/jfni3/code/dinov2-main/',  
            model='dinov2_vitb14_reg',           
            source='local',                      
            pretrained=False                    
        )
        self.dino_encoder.load_state_dict(torch.load('/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/dinov2_vitb14_reg4_pretrain.pth'))
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = torch.nn.Sequential(
            torch.nn.Linear(model_dim, latent_dim), 
            torch.nn.LayerNorm(latent_dim), 
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, len(level)),
            torch.nn.LayerNorm(len(level)), 
        )
        # self.to_codebook = torch.nn.Linear(model_dim, len(level))

        self.vq = FSQ(
            levels=level, 
            eps=1e-3, 
            dim=len(level),
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        # self.action_up = torch.nn.Linear(len(level), model_dim)
        self.action_up = torch.nn.Sequential(
            torch.nn.Linear(len(level), latent_dim), 
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, model_dim))

        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)
        
        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
        # import pdb
        # pdb.set_trace()
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, len(self.level))
        z_q, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, len(self.level))
        return {
            "patches": dion_features,
            "z_q": z_q,
            "indices": indices,
            "action_latent": action_pad,
            "in_max": z.max(),
            "in_min": z.min()
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]
        # import pdb
        # pdb.set_trace()
        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)

        # video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 
        video_recon = video_recon[:, :, -video_patches.shape[2]:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device

