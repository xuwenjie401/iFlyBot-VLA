from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model

from model.base_model import BaseModel
from model.action_model.DiT import DiT
# from model.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionalGenerationSTATE, Qwen2_5_VLMOEForConditionalGeneration
from model.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationSTATE
from model.qwen2_5_vl.modeling_qwen2_5_vlMOE import Qwen2_5_VLMOEForConditionalGeneration
from utils.flow_matching_utils import skewed_timestep_sample, sample_time
from utils.flow_matching.path import CondOTProbPath
from utils.nn_utils import (
    timestep_embedding,
    RotaryEmbedding,
    GemmaRMSNorm,
)
from config.base_configs import ModelArguments, TrainingArguments, OverallArguments
from config.token_configs import *
from config.data_spec import make_datastyle_mask
from utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)

def propagate_mask(mask):
    """
    处理mask矩阵, 将每行中True前面的所有元素变为True, True后面保持False
    参数:
        mask: 二维布尔张量, 每行有且仅有一个True
    返回:
        propagated_mask: 处理后的二维布尔张量
    """
    # 获取每行中True的位置
    positions = torch.argmax(mask.float(), dim=1)
    max_len = max(positions)
    
    # 生成列索引矩阵 [rows, cols]
    cols = torch.arange(mask.size(1), device=mask.device)
    # 扩展positions形状以匹配列索引矩阵
    expanded_positions = positions.unsqueeze(1)
    # 生成新的掩码: 列索引 <= 目标位置的元素为True
    propagate_mask = cols <= expanded_positions
    return propagate_mask[None, :, :, None], max_len


from model.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
class Qwen2_5_VLConfigModified(Qwen2_5_VLConfig):
    model_type = "qwen2_5_vl_state"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def add_entry(self, cfg: OverallArguments):
        self.domain_aware = cfg.task.domain_aware


class iFlyBotVLA(BaseModel):
    def __init__(self, cfg : OverallArguments, training_hf_args):
        super().__init__()

        modified_vlm_cfg = Qwen2_5_VLConfigModified.from_pretrained(
            pretrained_model_name_or_path=cfg.model.vlm_model_path
        )
        modified_vlm_cfg.add_entry(cfg)
        precision = torch.bfloat16 if (training_hf_args.bf16 or cfg.task.train != "train") else None
        
        if cfg.model.vlm_style == "MOE":
            overwatch.info(f"Use MOE VLM Model")
            vlm = Qwen2_5_VLMOEForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=cfg.model.vlm_model_path,
                cache_dir=training_hf_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=precision,
                config=modified_vlm_cfg,
            )
        elif cfg.model.vlm_style == "state":
            overwatch.info(f"Use state Model")
            vlm = Qwen2_5_VLForConditionalGenerationSTATE.from_pretrained(
                pretrained_model_name_or_path=cfg.model.vlm_model_path, 
                cache_dir=training_hf_args.cache_dir, 
                attn_implementation="flash_attention_2", 
                torch_dtype=precision,
                config=modified_vlm_cfg,
            )
        elif cfg.model.vlm_style == "normal":
            overwatch.info(f"Use normal Model")
            vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=cfg.model.vlm_model_path, 
                cache_dir=training_hf_args.cache_dir, 
                attn_implementation="flash_attention_2", 
                torch_dtype=precision,
                config=modified_vlm_cfg,
            )
        else:
            raise NotImplementedError(f"VLMStyle {cfg.vlm.vlm_style} not supported.")
        
        # [Validate] Model should be in Full Precision!
        if cfg.task.train == "train":
            for param in vlm.parameters():
                assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
        
        self.cfg = cfg
        self.vlm = vlm
        # TODO
        self.data_style = cfg.datasets.manip_data.action_style
        self.task = cfg.task
        self.action_chunk_size = cfg.datasets.manip_data.action_chunk_size
        self.action_dim = cfg.model.action_expert_cfg.action_dim
        
        self.action_mask_indices = make_datastyle_mask(cfg=cfg)
        
        dit_num_layers = self.vlm.config.num_hidden_layers
        if isinstance(self.vlm, PeftModel):
            dit_num_layers = self.vlm.model.config.num_hidden_layers
        self.action_expert = DiT(
            cfg=cfg.model.action_expert_cfg,
            num_layers=dit_num_layers,
            llm_emb_dim=self.vlm.model.layers[0].self_attn.k_proj.weight.shape[0],
            domain_aware=cfg.task.domain_aware,
        )
        
        self.enable_multi_noise_sample = cfg.model.enable_multi_noise_sample
        self.multi_noise_sample_num = cfg.model.multi_noise_sample_num
        # flow-matching Utils
        self.path = CondOTProbPath()
        self.skewed_timesteps = cfg.model.skew_timesteps
        
        self.domain_aware = cfg.task.domain_aware
        
        self.truncate_action_grad_to_vlm = cfg.model.truncate_action_grad_to_vlm
        
        
    def load_from_checkpoint(self, pretrained_checkpoint, vlm_only=False, old_version=False):
        model_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')["model"]
        if old_version:
            from deploy.tools.old_ckpt_mapping import remap_ckpt_keys
            model_state_dict = remap_ckpt_keys(model_state_dict)

        self.vlm.load_state_dict(model_state_dict["vlm"], strict=True)
        if not vlm_only:
            self.action_expert.load_state_dict(model_state_dict["action_head"], strict=True)
        del model_state_dict
        
        return
    
    
    def check_freeze_and_lora(self):
        if self.cfg.model.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        if self.cfg.task.domain_aware and self.cfg.task.domain_freeze:
            for name, param in list(self.named_parameters()):
                if "fc.weight" not in name and "bias.weight" not in name and "soft_prompt_hub" not in name:
                    param.requires_grad = False
        
        if self.cfg.model.use_lora:
            overwatch.info(f"Use Lora")
            target_linear_layers = []
            for name, module in self.vlm.named_modules():
                # 检查：
                # （1）模块属于目标父模块的指定层级范围内
                # （2）模块类型是nn.Linear
                if (
                    ("aggregator" not in name and 
                    # "visual" not in name and 
                    "lm_head" not in name)
                    and isinstance(module, nn.Linear)
                ):
                    target_linear_layers.append(name)
            target_linear_layers = list(set(target_linear_layers))
            print(f"筛选出的线性层数量：{len(target_linear_layers)}")
            print("目标线性层名称：", target_linear_layers)
            lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=min(self.cfg.model.lora_rank, 16),
                lora_dropout=self.cfg.model.lora_dropout,
                target_modules=target_linear_layers,
                init_lora_weights="gaussian",
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()

            if self.cfg.task.train != "train":
                self.vlm.merge_and_unload()
    
    
    def _build_vlm_kwargs(self, batch, for_generate: bool = False, **generate_kwargs):
        """_summary_
        管理 各种情况下 传给 vlm的字段
        """
        
        # Qwen的基础字段
        kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            position_ids=batch["position_ids"],
        )
        if not for_generate:
            kwargs["labels"] = batch["labels"]
        
        if self.cfg.model.vlm_style == "state":
            kwargs["proprios"] = batch["proprios"]
        
        if self.domain_aware:
            kwargs["domain_id"] = batch["domain_id"]
        
        # generate时的字段 (max_new_tokens, eos_token_id等)
        if for_generate:
            kwargs.update(generate_kwargs)
            
        return kwargs
    
    
    def get_kv_cache_from_vlm(self, vlm_output, trucate_action_grad_to_vlm=False):
        """  """
        vlm_num_layers = len(vlm_output["past_key_values"])
        vlm_keys, vlm_values = [], []
        for layer_idx in range(vlm_num_layers):
            vlm_keys.append(vlm_output["past_key_values"][layer_idx][0])
            vlm_values.append(vlm_output["past_key_values"][layer_idx][1])
            
        vlm_keys = torch.stack(vlm_keys, dim=0)           # [num_layers, bsz, num_kv_heads, seq_len, kv_head_dim]
        vlm_values = torch.stack(vlm_values, dim=0)       # [num_layers, bsz, num_kv_heads, seq_len, kv_head_dim]
        
        vlm_keys = vlm_keys.permute(0, 1, 3, 2, 4)        # [num_layers, bsz, seq_len, num_kv_heads, kv_head_dim]
        vlm_values = vlm_values.permute(0, 1, 3, 2, 4)    # [num_layers, bsz, seq_len, num_kv_heads, kv_head_dim]
        
        vlm_keys = vlm_keys.reshape(vlm_num_layers, vlm_keys.size(1), vlm_keys.size(2), -1)            # [num_layers, bsz, seq_len, total_kv_dim]
        vlm_values = vlm_values.reshape(vlm_num_layers, vlm_values.size(1), vlm_values.size(2), -1)    # [num_layers, bsz, seq_len, total_kv_dim]
        
        if trucate_action_grad_to_vlm:
            vlm_keys = vlm_keys.detach()
            vlm_values = vlm_values.detach()
        return vlm_keys, vlm_values
    
    
    def vlm_forward(self, batch):
        vlm_output = self.vlm(**self._build_vlm_kwargs(batch, for_generate=False))
        vlm_keys, vlm_values = self.get_kv_cache_from_vlm(vlm_output, trucate_action_grad_to_vlm=self.truncate_action_grad_to_vlm)  

        if self.task.use_lap_aux:
            # TODO: 当前不成文规定, FAST任务的token最后出
            # FAST token的kv不参与 dit模块计算
            auxiliary_mask = (batch["labels"] == LAP_END_TOKEN_INDEX)
        elif self.task.use_fast_aux:
            auxiliary_mask = (batch["labels"] == FAST_START_TOKEN_INDEX)
        else:
            auxiliary_mask = (batch["labels"] == MEANINGLESS_TOKEN_INDEX)
            
        if torch.any(auxiliary_mask).item():
            auxiliary_mask, max_aux_len = propagate_mask(auxiliary_mask)
            vlm_keys = (vlm_keys * auxiliary_mask)[:, :, : max_aux_len]
            vlm_values = (vlm_values * auxiliary_mask)[:, :, : max_aux_len]
            
        vlm_loss = vlm_output.loss
        return vlm_keys, vlm_values, vlm_loss, vlm_output
    
    
    def vlm_inference(self, batch, max_new_tokens=10, do_sample=False, temparature=0.7):
        eos_token_ids = [IMG_END_TOKEN_INDEX]
        if self.cfg.task.use_lap_aux and self.cfg.task.infer_predict_latent:
            max_new_tokens = self.cfg.task.num_latent_tokens + 2
            eos_token_ids.append(LAP_END_TOKEN_INDEX)
        # if self.cfg.task.use_fast_aux:
        #     # 推理默认不使用FAST任务
        #     eos_token_ids.append(FAST_START_TOKEN_INDEX)
        
        # TODO 这里似乎不需要加原本的eos
        gen_kwargs = self._build_vlm_kwargs(
            batch,
            for_generate=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temparature,
            eos_token_id=eos_token_ids,
            return_dict_in_generate=True,
        )
        
        vlm_output = self.vlm.generate(**gen_kwargs)
        vlm_keys, vlm_values = self.get_kv_cache_from_vlm(vlm_output)
        return vlm_keys, vlm_values, vlm_output

    
    def action_expert_forward(self, batch, vlm_keys, vlm_values, training=True):
        precision = torch.bfloat16 if training else torch.float32
        
        if self.skewed_timesteps:
            t = skewed_timestep_sample(batch["actions"].shape[0], device=batch["actions"].device)
        else:
            t = sample_time(batch["actions"].shape[0], batch["actions"].device).to(batch["actions"].dtype)
            
        noise = torch.randn_like(batch["actions"]).to(batch["actions"].device)
        
        time_expanded = t[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * batch["actions"]
        u_t = noise - batch["actions"]
        
        domain_ids = batch["domain_id"] if self.domain_aware else None
        u_t_predicted = self.action_expert(batch["proprios"], x_t, t, vlm_keys.to("cuda"), vlm_values.to("cuda"), bf16=training, domain_id=domain_ids)
        if self.action_mask_indices is not None:
            action_loss = torch.pow((u_t_predicted - u_t)[:, :, self.action_mask_indices], 2).mean()
        else:
            action_loss = torch.pow(u_t_predicted - u_t, 2).mean()
    
        return u_t_predicted, action_loss
    
    
    def forward(self, batch, training=True, return_vlm_kv=False):
        vlm_keys, vlm_values, vlm_loss, vlm_output = self.vlm_forward(batch)
        
        multi_noise_sample_loss = 0.0
        if self.enable_multi_noise_sample:
            for _ in range(self.multi_noise_sample_num):
                predicted_actions, action_loss = self.action_expert_forward(batch, vlm_keys, vlm_values, training=training)
                multi_noise_sample_loss += action_loss
        else:
            predicted_actions, action_loss = self.action_expert_forward(batch, vlm_keys, vlm_values, training=training)
            multi_noise_sample_loss += action_loss
            
        output = {
            "predicted_actions": predicted_actions,
            "action_loss": multi_noise_sample_loss,
            "vlm_loss": vlm_loss,
            "vlm_output": vlm_output,
        }

        if return_vlm_kv:
            output.update({
                "vlm_keys": vlm_keys,
                "vlm_values": vlm_values,
            })

        return output


    def predict_action(
        self,
        batch,
        device: torch.device = torch.device("cuda"),
        bf16: bool = True,
        denoise_steps: int = 5,
    ):
        """
        推理函数, cfg中的重要参数:
            chunk_size, 
            task.use_lap_aux,  -->  推理时是否predict latent  -->  vlm的推理截止token数
        """
        
        # TODO: 合并
        # if self.cfg.task.use_lap_aux and self.cfg.task.infer_predict_latent:
        #     vlm_keys, vlm_values, vlm_output = self.vlm_inference(batch)
        # else:
        #     vlm_keys, vlm_values, _, vlm_output = self.vlm_forward(batch)
        
        vlm_keys, vlm_values, vlm_output = self.vlm_inference(batch)
        bsz = batch["input_ids"].shape[0]
        
        precision = torch.bfloat16 if bf16 else torch.float32
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["proprios"] = batch["proprios"].to(dtype=precision)
        domain_ids = batch["domain_id"] if self.domain_aware else None
        
        dt = torch.tensor(-1.0 / denoise_steps, dtype=precision, device=device)
        x_t = torch.randn((bsz, self.action_chunk_size, self.action_dim), dtype=precision, device=device)
        time = torch.tensor(1.0, dtype=precision, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsz)
            v_t = self.action_expert(
                batch["proprios"], x_t, expanded_time, vlm_keys.to(device), vlm_values.to(device), bf16=bf16, domain_id=domain_ids)
            
            x_t += dt * v_t
            time += dt

        return x_t
        
        