from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from tasks.lapa_model.genie.modules.blocks_fsq import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, FSQ, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class UncontrolledfsqModel(nn.Module):
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
        super(UncontrolledfsqModel, self).__init__()
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
        self.vq = FSQ(
            levels=level, 
            eps=1e-3, 
            dim=len(level),
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
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

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained('/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/t5-base')
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # self.lang_norm = nn.LayerNorm(model_dim) 
        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/t5-base')

    


    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        # dion_features = F.normalize(dion_features, p=2, dim=-1)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        
        # lang_embed = F.normalize(lang_embed, p=2, dim=-1)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask) 

        # Get language embedding
        lang_emb = z[:, 1:, self.num_codes + dion_features.shape[2]:]

        # lang_emb = F.normalize(lang_emb, p=2, dim=-1)
        # Get latent action for all future frames
        # import pdb
        # pdb.set_trace()
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, len(self.level))
        
        z_q, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, len(self.level))
        return {
            "patches": dion_features,
            "z_q": z_q,
            "indices": indices,
            "lang_emb": lang_emb,
            "action_latent": action_pad,
            "lang": lang_embed,
            "in_max": z.max(),
            "in_min": z.min()
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)

        # lang_embed = self.lang_norm(lang_embed)

        attention_mask = torch.cat([torch.ones((B, self.num_codes + (H // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)

        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches, outputs['lang_emb'], attention_mask)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

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




class ControllablefsqModel(nn.Module):
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
        super(ControllablefsqModel, self).__init__()
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
        missing_keys, unexpected_keys = self.dino_encoder.load_state_dict(torch.load('/dmx-b3-ssd01/sppro/permanent/jfni3/code/UniVLA-main/pretrained_model/dinov2_vitb14_reg4_pretrain.pth'))
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
        self.to_codebook_uncontrol = torch.nn.Sequential(
            torch.nn.Linear(model_dim, latent_dim), 
            torch.nn.LayerNorm(latent_dim), 
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, len(level)),
            torch.nn.LayerNorm(len(level)), 
        )
        self.vq = FSQ(
            levels=level, 
            eps=1e-3, 
            dim=len(level),
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = torch.nn.Sequential(
            torch.nn.Linear(len(level), latent_dim), 
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, model_dim))
        self.action_up_uncontrol = torch.nn.Sequential(
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

        self.vq_action = FSQ(
            levels=level, 
            eps=1e-3, 
            dim=len(level),
        )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)
        # dion_features = F.normalize(dion_features, p=2, dim=-1)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 

        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, len(self.level))
        z_q_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, len(self.level))


        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, len(self.level))
        z_q, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, len(self.level))


        return {
            "patches": dion_features,
            "z_q": z_q,
            "z_q_uncontrol": z_q_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
            "action_latent": action_pad,
            "action_latent_controllable": action_pad_controllable,
            "in_max": z_action.max(),
            "in_min": z_action.min()
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
                                          video_patches],
                                          dim=2)
        
        video_recon = self.decoder(video_action_patches)
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