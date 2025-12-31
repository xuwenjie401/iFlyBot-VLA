from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
import dataclasses
import einops

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from collections import deque
import json
from typing import List, Tuple, Union, Dict
import copy
import sys
from pathlib import Path

import torch
from typing_extensions import override

from torch.utils._pytree import tree_map

import transformers
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor

# project_root = Path(__file__).parent.parent
# print(f"project-root: {project_root}")
# sys.path.append(str(project_root))

from qwen2_5_vl import Qwen2_5_VLForConditionalGenerationSTATE
from peft import LoraConfig, PeftModel, get_peft_model
from vla_scripts.vla_os_diffusion import ActionOnlyVLA
from preprocess_data import preprocess_data

from sppro_infer import base_policy as _base_policy
from sppro_infer import infer_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
ASSISTANT_TOKEN_INDEX = 77091
MAGIC_TOKEN_INDEX = 198

# FAST action tokenizer的vocab size是2048
# 所以从OFFSET开始的2048个token被用作FAST
FAST_TOKEN_INDEX_OFFSET = 100000
LAP_TASK_TOKEN = "<lam_act>"
LAP_START_TOKEN = "<|LAPstart|>"
LAP_START_TOKEN_INDEX = 151694
LAP_END_TOKEN = "<|LAPend|>"
LAP_END_TOKEN_INDEX = 151695
FAST_TASK_TOKEN = "<|FASTpredict|>"
FAST_START_TOKEN = "<|FASTstart|>"
FAST_START_TOKEN_INDEX = 151697
FAST_END_TOKEN = "<|FASTend|>"
FAST_END_TOKEN_INDEX = 151698
STATE_PAD_TOKEN = "<|StatePad|>"
STATE_PAD_TOKEN_INDEX = 151699

class ActionHeadConfig:
    paradigm="action-only"
    load_from_checkpoint = False
    checkpoint = None
    
    planning_heads = []
    planning_mode = "implicit"

    three_d = False
    action_dim = 20
    proprio_dim = 20
    hidden_size = 1024
    num_heads = 16
    dropout = 0.1
    gated = False
    action_loss_type = 'l1'
    data_type = 'universal'


@dataclasses.dataclass
class VLAArgs:
    resize_size: int = 448
    replan_steps: int = 10

    num_trials_per_task: int = 1

    use_lora: bool = False
    use_vggt = True

    use_lap: bool  = True
    use_FAST: bool  = True
    use_state_pad: bool  = True
    
    obs_horizon: int = 2
    action_chunk_size: int = 30

    mask_virtual_view: bool = True

    bimanual: bool = False
    # relative_joint: bool = False
    relative_joint: bool = True


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def LindenInputTransform(data: dict, is_bimanual: bool) -> dict:
    # breakpoint()
    state = np.asarray(data["state"])

    # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
    # stores as float32 (C,H,W), gets skipped for policy inference
    base_image = _parse_image(data["images"]["cam_head"])
    wrist_image_left = _parse_image(data["images"]["cam_left"])
    if is_bimanual:
        wrist_image_right = _parse_image(data["images"]["cam_right"])
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
    else:
        names = ("base_0_rgb", "left_wrist_0_rgb")
        images = (base_image, wrist_image_left)

    inputs = {
        "state": state,
        "image": dict(zip(names, images, strict=True))
    }

    if "actions" in data:
        inputs["actions"] = np.asarray(data["actions"])

    if "prompt" in data:
        if isinstance(data["prompt"], bytes):
            data["prompt"] = data["prompt"].decode("utf-8")
        inputs["prompt"] = data["prompt"]

    return inputs


def input_transform(inputs: dict, is_bimanual: bool) -> dict:
    inputs = LindenInputTransform(inputs, is_bimanual)
    return inputs


def output_transform(outputs: dict, is_bimanual: bool) -> dict:

    return outputs


class Policy(BasePolicy):
    def __init__(
        self,
        tokenizer,
        image_processor,
        model,
        vla_args : VLAArgs,
        normalize_stats: dict,
        *,
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda",
    ):
        """Initialize the Policy."""
        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._device = pytorch_device
        
        self.vla_args = vla_args
        self.normalize_stats = normalize_stats

        self._model = self._model.to(pytorch_device)
        self._model.eval()
        
        self.history_states = deque(maxlen=self.vla_args.obs_horizon+1)

        self.bimanual = self.vla_args.bimanual
        if self.bimanual:
            self.real_image_flags = [True, True, True]
        else:
            self.real_image_flags = [True, True, False]
            
        self.relative_joint = self.vla_args.relative_joint


    def prepare_observations(self, inputs: dict) -> dict:
        img_head = inputs["image"]["base_0_rgb"]
        img_wrist_left = inputs["image"]["left_wrist_0_rgb"]
        if self.bimanual:
            img_wrist_right = inputs["image"]["right_wrist_0_rgb"]
        else:
            img_wrist_right = np.zeros_like(img_wrist_left)
        
        # 致盲测试
        # img_head = np.zeros_like(img_head)
        # img_wrist_left = np.zeros_like(img_wrist_left)
        
        img_head = infer_utils.resize_image_np_version(img_head, resize_size=self.vla_args.resize_size)
        img_wrist_left = infer_utils.resize_image_np_version(img_wrist_left, resize_size=self.vla_args.resize_size)
        img_wrist_right = infer_utils.resize_image_np_version(img_wrist_right, resize_size=self.vla_args.resize_size)
        # Image.fromarray(img_head).save("result/img_head.png")
        # Image.fromarray(img_wrist_left).save("result/img_wrist_left.png")
        # Image.fromarray(img_wrist_right).save("result/img_wrist_right.png")
        
        cur_state = inputs["state"]
        last_state = self.history_states[-1] if len(self.history_states) > 0 else cur_state
        obs_horizon = self.vla_args.obs_horizon 
        state_list = []
        # 包含0位state
        self.history_states.append(cur_state)
        if obs_horizon > 0:
            # 如果设置需要历史状态，则尝试获取
            if len(self.history_states) >= obs_horizon:
                for x in list(self.history_states)[-obs_horizon:]:
                    state_list.append(x)
            else:
                remain_num = obs_horizon - len(self.history_states)
                # 如果不足，补上重复的最old state
                repeat_state = self.history_states[0] if len(self.history_states) > 0 else cur_state
                for x in range(remain_num):
                    state_list.append(repeat_state)
                for x in self.history_states:
                    state_list.append(x)
                    
            state = np.stack(state_list, axis=0)
        else:
            state = None
            
        # 由于我们的state-horizon是 帧级的，所以这里的推理要加插值
        # state = infer_utils.interpolate_joint_state(last_state, cur_state, self.vla_args.action_chunk_size, self.vla_args.obs_horizon, self.relative_joint)
        # breakpoint()
            
        # print(f"cur state: {cur_state}")
        state = infer_utils.normalize_joint_state(state, self.normalize_stats, self.bimanual, pad_for_20=True)
        
        # state非current设置为0
        state[0, :] = 0
        
        # state = np.zeros_like(state)
            
        if len(self.history_states) > obs_horizon:
            self.history_states.popleft()
        
        data_dict = {}
        data_dict['images'] = [img_head, img_wrist_left, img_wrist_right]
        data_dict['states'] = state
        return cur_state, data_dict, img_head


    def process_image_unified(self, image):
        # TODO: 是否应该走 加载出的 processor
        visual_processed = self._image_processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def batchify_standard_input(self, lang: str, images: List[np.ndarray], states: np.ndarray) -> dict:
        task_prefix = ""

        task_prefix += LAP_TASK_TOKEN
        task_prefix += FAST_TASK_TOKEN

        state_pad_tokens = ""
        for i in range(self.vla_args.obs_horizon):
            state_pad_tokens += STATE_PAD_TOKEN

        action_tokens = ''
        # breakpoint()
        conversation = [
            {"from": "human", "value": f"{task_prefix}<image><image><image>{state_pad_tokens}What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        results = [self.process_image_unified(im) for im in images]
        image, grid_thw = map(list, zip(*results))
        
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self._image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        # 重点按照qwen的处理流程
        chat_sources = [conversation]
        data_dict = infer_utils.preprocess_qwen_2_visual(
            chat_sources,
            self._tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=None,
            task="infer",
            real_image_flags=self.real_image_flags,
            mask_virtual_view=self.vla_args.mask_virtual_view,
        )
        position_ids, _ = infer_utils.get_rope_index_25(
            self._image_processor.merge_size,
            data_dict['input_ids'],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(None),
            second_per_grid_ts=None,
        )

        data_dict['proprios'] = torch.from_numpy(states)
        data_dict['position_ids'] = position_ids
        # data_dict['attention_mask'] = [data_dict['input_ids'][0].size(0)]
        data_dict['attention_mask'] = data_dict['input_ids'].ne(self._tokenizer.pad_token_id)
        
        data_dict['pixel_values'] = torch.cat(image, dim=0)
        data_dict['image_grid_thw'] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thw], dim=0
        )

        batch = {k: v.to(self._device) if torch.is_tensor(v) else v 
                for k, v in data_dict.items()}
        
        batch["vggt_inputs"] = None
        if 'action_chunk' in batch:
            batch['actions'] = batch['action_chunk'].to(torch.bfloat16)
        if 'state' in batch:
            batch['proprios'] = batch['state'].to(torch.bfloat16)
        
        bsz_field = ['proprios']
        for fi in bsz_field:
            batch[fi] = batch[fi].unsqueeze(0)
        return batch


    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        # inputs = jax.tree.map(lambda x: x, obs)
        inputs = tree_map(lambda x: x, obs)
        inputs = self._input_transform(inputs, self.bimanual)
        
        task_description = inputs["prompt"]

        cur_state, obs_dict, image_show = self.prepare_observations(inputs)
        batch = self.batchify_standard_input(lang=task_description, images=obs_dict["images"], states=obs_dict["states"])

        start_time = time.monotonic()
        with torch.no_grad():
            raw_actions = self._model.select_action(batch, num_steps=5, action_num=30, prediction_latent=True, max_new_token=10, use_state=True)
        model_time = time.monotonic() - start_time
        print(f"infer time cost: {model_time} s")

        raw_actions = raw_actions.to("cpu").detach().to(torch.float32).numpy()
        # breakpoint()
        action_chunk = infer_utils.process_joint_action(raw_actions, self.normalize_stats, cur_state, is_bimanual=self.bimanual, relative_joint=self.relative_joint)
        # breakpoint()

        outputs = {
            "state": inputs["state"],
            "actions": action_chunk,
        }
        # breakpoint()

        outputs = self._output_transform(outputs, self.bimanual)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


def create_trained_policy(
    config : VLAArgs,
) -> Policy:
    
    model_path = "/home/iflytek/sppro/repo/robobrain2_3b"
    
    # 绝对位姿
    # norm_stats_file = "/home/iflytek/sppro/repo/norm/sppro_statistics_sppro_pick_place_p1_bimanual-False_absS-True_absA-True_0915.json"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/abs_step-029999-loss_0.0564.pt"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/1021-abs_step-049999-loss_0.0964.pt"
    
    # 相对位姿
    # norm_stats_file = "/home/iflytek/sppro/repo/norm/sppro_statistics_sppro_pick_place_p1_universal_bimanual-False_absS-False_absA-False_0915.json"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/rel_quat_step-039999-loss_0.1461.pt"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/1021-rel_quat-049999-loss_0.1820.pt"
    
    # 相对关节角
    norm_stats_file = "/home/iflytek/sppro/repo/norm/sppro_statistics_sppro_pick_place_p2_joint_bimanual-False_absS-True_absA-False_0915.json"
    model_checkpoint = "/home/iflytek/sppro/repo/ckpts/joint_p12345_step-019999-loss_0.6102.pt"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/joint_p123_step-059999-loss_0.4427.pt"
    # norm_stats_file = "/home/iflytek/sppro/repo/norm/sppro_statistics_sppro_pick_place_p1_joint_bimanual-False_absS-True_absA-False_0915.json"
    # model_checkpoint = "/home/iflytek/sppro/repo/ckpts/joint_rel_step-049999-loss_0.3231.pt"
    
    normalize_stats = json.load(open(norm_stats_file, "r"))

    image_processor = AutoProcessor.from_pretrained(
        model_path,
    ).image_processor
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )

    vlacfg = ActionHeadConfig
    vlm = Qwen2_5_VLForConditionalGenerationSTATE.from_pretrained(
        model_path, cache_dir=None, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    vla = ActionOnlyVLA(vlm, cfg=vlacfg, training_algo="flow_matching", model_type="state")
    
    args = VLAArgs()
    if args.use_lora:
        target_linear_layers = []
        for name, module in vlm.named_modules():
            # 检查：
            # （1）模块属于目标父模块的指定层级范围内
            # （2）模块类型是nn.Linear
            if (
                ("aggregator" not in name and "visual" not in name and "lm_head" not in name)
                and isinstance(module, torch.nn.Linear)
            ):
                target_linear_layers.append(name)
        target_linear_layers = list(set(target_linear_layers))
        print(f"筛选出的线性层数量：{len(target_linear_layers)}")
        print("目标线性层名称：", target_linear_layers)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=min(32, 16),
            lora_dropout=0.0,
            target_modules=target_linear_layers,
            init_lora_weights="gaussian",
        )
        vlm = get_peft_model(vlm, lora_config)
        vlm.print_trainable_parameters()

    vla.load_from_checkpoint(model_checkpoint)

    if args.use_lora:
        vlm.merge_and_unload()

    vla = vla.to("cuda").to(torch.bfloat16)
    vla.eval()

    return Policy(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=vla,
        vla_args = args,
        normalize_stats=normalize_stats,
        pytorch_device="cuda",
    )

