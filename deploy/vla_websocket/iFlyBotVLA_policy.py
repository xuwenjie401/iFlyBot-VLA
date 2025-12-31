from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
import dataclasses
import einops

import numpy as np
from PIL import Image
from collections import deque
import json
from typing import List, Tuple, Union, Dict
import copy

import torch
from typing_extensions import override

from torch.utils._pytree import tree_map

import transformers
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor

from src.iFlyBotVLA.model.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationSTATE
from peft import LoraConfig, PeftModel, get_peft_model
from src.iFlyBotVLA.model.iFlyBotVLA import iFlyBotVLA
from src.iFlyBotVLA.config.token_configs import *
from src.iFlyBotVLA.config.base_configs import TaskArguments, ManipDataArguments, TrainingArguments, OverallArguments
from deploy.tools.infer_configs import InferArguments

from deploy.vla_websocket import base_policy as _base_policy
from deploy.tools import infer_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def LindenPoseInputTransform(data: dict, is_bimanual: bool) -> dict:
    # breakpoint()
    state = np.asarray(data["state"])
    # 如果双臂模式 但只有单臂数据，右臂用复制
    if is_bimanual == True and state.shape[-1] == 8:
        # state = np.concat([state, state], axis=-1)
        # rightarm_default = np.array([0.45971617102622986, -0.5011505484580994, -0.28870856761932373,
        #                              0.05748261138796806, 0.9487566351890564, -0.19332052767276764,
        #                              -0.24327723681926727, 0.9700000286102295])
        rightarm_default = np.array([0.45971617102622986, -0.5011505484580994, -0.28870856761932373,
                                     0.05748261138796806, 0.9487566351890564, -0.19332052767276764,
                                     -0.24327723681926727, 0.00000286102295])
        state = np.concat([state, rightarm_default], axis=-1)
    # 这里传入的state是16维, left-(x,y,z,qw,qx,qy,qz,grip), right-(x,y,z,qw,qx,qy,qz,grip)
    # 四元数转为xyzw格式
    state = infer_utils.state_qwxyz_to_qxyzw(state, is_bimanual)
    # breakpoint()

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


def LindenJointInputTransform(data: dict, is_bimanual: bool) -> dict:
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


def LIBEROInputTransform(data: dict) -> dict:
    return data
    


def input_transform(inputs: dict, embodiment:str, action_type: str, is_bimanual: bool) -> dict:
    if embodiment == "sppro_pick_place_v4":
        if action_type == "eef_pose":
            inputs = LindenPoseInputTransform(inputs, is_bimanual)
        elif action_type == "joint":
            inputs = LindenJointInputTransform(inputs, is_bimanual)
        else:
            raise ValueError(f"embodiment-{embodiment} does not support action-type {action_type}")
    elif embodiment.startswith("libero"):
        if action_type == "delta_eef":
            inputs = LIBEROInputTransform(inputs)
        else:
            raise ValueError(f"embodiment-{embodiment} does not support action-type {action_type}")

    return inputs


def output_transform(outputs: dict, is_bimanual: bool) -> dict:
    # 临时
    # outputs["actions"][:, -1] = 1.0

    return outputs


class Policy(BasePolicy):
    def __init__(
        self,
        tokenizer,
        image_processor,
        model,
        cfg : OverallArguments,
        infer_cfg: InferArguments,
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
        
        self.manip_cfg = cfg.datasets.manip_data
        self.task_cfg = cfg.task
        self.normalize_stats = json.load(open(infer_cfg.norm_stats_file, "r"))

        self._model = self._model.to(pytorch_device)
        self._model.eval()

        self.embodiment = infer_cfg.embodiment
        self.action_style = self.manip_cfg.action_style
        self.control_reference = self.manip_cfg.control_reference
        self.is_relative = (self.control_reference == "relative" or self.control_reference == "anchored_relative")

        self.ignore_states = infer_cfg.ignore_states

        # 注意这里用的是infer config里的
        self.bimanual = infer_cfg.is_bimanual
        # 这里是用来做图像mask的，当前默认最少的单臂情况也有 头+腕部视角
        if self.bimanual:
            self.real_image_flags = [True, True, True]
        else:
            self.real_image_flags = [True, True, False]

        # 历史遗留  state horizon
        self.state_horizon = infer_cfg.state_horizon
        self.history_states = deque(maxlen=self.state_horizon+1)
        self.old_version_ckpt = infer_cfg.old_version_ckpt
        self.old_version_prompt = infer_cfg.old_version_prompt
        # 获取对应图像字段
        self.image_head_entry = infer_cfg.image_head
        self.image_wrist_left_entry = infer_cfg.image_wrist_left
        self.image_wrist_right_entry = infer_cfg.image_wrist_right
        # breakpoint()

    def prepare_observations(self, inputs: dict) -> dict:
        img_head = inputs["image"][self.image_head_entry]

        img_wrist_left = inputs["image"][self.image_wrist_left_entry]
        if self.bimanual:
            img_wrist_right = inputs["image"][self.image_wrist_right_entry]
        else:
            img_wrist_right = np.zeros_like(img_wrist_left)
            
        # 致盲测试 blind
        # img_head = np.zeros_like(img_head)
        # img_wrist_left = np.zeros_like(img_wrist_left)
        
        img_head = infer_utils.resize_image_np_version(img_head, resize_size=self.manip_cfg.resize_resolution)
        img_wrist_left = infer_utils.resize_image_np_version(img_wrist_left, resize_size=self.manip_cfg.resize_resolution)
        img_wrist_right = infer_utils.resize_image_np_version(img_wrist_right, resize_size=self.manip_cfg.resize_resolution)
        # Image.fromarray(img_head).save("outputs/img_head.png")
        # Image.fromarray(img_wrist_left).save("outputs/img_wrist_left.png")
        # Image.fromarray(img_wrist_right).save("result/img_wrist_right.png")
        
        cur_state = inputs["state"]
        last_state = self.history_states[-1] if len(self.history_states) > 0 else cur_state
        state_list = []
        # 绝对位姿时包含0位state， 相对位姿时是-2, -1，不包含0位
        if not self.is_relative:
            self.history_states.append(cur_state)
        if self.state_horizon > 0:
            # 如果设置需要历史状态，则尝试获取
            if len(self.history_states) >= self.state_horizon:
                for x in list(self.history_states)[-self.state_horizon:]:
                    state_list.append(x)
            else:
                remain_num = self.state_horizon - len(self.history_states)
                # 如果不足，补上重复的最old state
                repeat_state = self.history_states[0] if len(self.history_states) > 0 else cur_state
                for x in range(remain_num):
                    state_list.append(repeat_state)
                for x in self.history_states:
                    state_list.append(x)
                    
            state = np.stack(state_list, axis=0)
        else:
            state = None
        # 相对位姿时在更新完state_list后再添加当前state
        if self.control_reference == "relative":
            self.history_states.append(cur_state)
            
        # 由于我们的state-horizon是 帧级的，所以这里的推理要加插值
        # forget it
        
        # breakpoint()
        if self.action_style == "eef_pose":
            if self.control_reference == "relative":
                state = infer_utils.make_relative(seq=state, ref=cur_state)
            
            if self.manip_cfg.rotation_representation == "quat":
                # print(f"cur state: {cur_state}")
                state = infer_utils.normalize_pose_quat_state(state, self.normalize_stats, self.bimanual, pad_for_20=True)
            else:
                raise ValueError(f"valid-{self.manip_cfg.rotation_representation} infer not implemented yet")
        
        elif self.action_style == "joint":
            state = infer_utils.normalize_joint_state(state, self.normalize_stats, self.bimanual, pad_for_20=True)
        
            # old ckpt 非cur_state设0
            if self.old_version_ckpt and self.state_horizon > 1:
                state[0, :] = 0
        
        elif self.embodiment.startswith("libero") and self.action_style == "delta_eef":
            state = infer_utils.normalize_libero_delta_eef_state(state, self.normalize_stats, pad_for_20=True)
            
        # breakpoint()
        
        # state全部置0
        if self.ignore_states:
            state = np.zeros_like(state)
            
        if len(self.history_states) > self.state_horizon:
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
        task_lap_prefix = ''
        if self.task_cfg.use_lap_aux:
            task_lap_prefix = LAP_TASK_TOKEN

        task_fast_prefix = ''
        if self.task_cfg.use_fast_aux:
            task_fast_prefix = FAST_TASK_TOKEN

        state_pad_tokens = ""
        for i in range(self.state_horizon):
            state_pad_tokens += STATE_PAD_TOKEN

        domain_prefix = ""
        if self.task_cfg.domain_aware:
            for i in range(32):
                domain_prefix += DOMAIN_PAD_TOKEN

        action_tokens = ''
        conversation = [
            {"from": "human", "value": f"<image><image><image>What action should the robot take to {lang}?{task_lap_prefix}{domain_prefix}{state_pad_tokens}{task_fast_prefix}"},
            {"from": "gpt", "value": action_tokens},
        ]
        if self.old_version_prompt:
            conversation = [
                {"from": "human", "value": f"{task_lap_prefix}{task_fast_prefix}<image><image><image>{state_pad_tokens}What action should the robot take to {lang}?"},
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

        # 这里额外拷贝了下面两个函数  是由于之前奇怪的环境问题
        chat_sources = [conversation]
        data_dict = infer_utils.preprocess_qwen_2_visual(
            chat_sources,
            self._tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=None,
            task_cfg=self.task_cfg,
            real_image_flags=self.real_image_flags,
            mask_virtual_view=self.manip_cfg.mask_virtual_view,
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
        if self.manip_cfg.mask_virtual_view:
            # NOTE: 这里mask  pad  0
            attention_masks = [data_dict["attention_mask"]]
            attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            attention_masks = attention_masks[:, : self._tokenizer.model_max_length]
        else:
            data_dict['attention_mask'] = data_dict['input_ids'].ne(self._tokenizer.pad_token_id)
        
        data_dict['pixel_values'] = torch.cat(image, dim=0)
        # data_dict['pixel_values'] = torch.ones_like(data_dict['pixel_values'])
        data_dict['image_grid_thw'] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thw], dim=0
        )

        batch = {k: v.to(self._device) if torch.is_tensor(v) else v 
                for k, v in data_dict.items()}
        
        if 'state' in batch:
            batch['proprios'] = batch['state'].to(torch.bfloat16)
        
        # breakpoint()
        bsz_field = ['proprios'] 
        for fi in bsz_field:
            batch[fi] = batch[fi].unsqueeze(0)
        return batch


    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = tree_map(lambda x: x, obs)
        inputs = self._input_transform(inputs, self.embodiment, self.action_style, self.bimanual)
        
        task_description = inputs["prompt"]

        cur_state, obs_dict, image_show = self.prepare_observations(inputs)
        batch = self.batchify_standard_input(lang=task_description, images=obs_dict["images"], states=obs_dict["states"])

        start_time = time.monotonic()
        with torch.no_grad():
            raw_actions = self._model.predict_action(batch, device=torch.device("cuda"), denoise_steps=5)
        model_time = time.monotonic() - start_time

        raw_actions = raw_actions.to("cpu").detach().to(torch.float32).numpy()
        # breakpoint()
        if self.action_style == "eef_pose":
            action_chunk = infer_utils.process_pose_quat_action(raw_actions, self.normalize_stats, cur_state, is_bimanual=self.bimanual, relative_pose=self.is_relative)
        elif self.action_style == "joint":
            action_chunk = infer_utils.process_joint_action(raw_actions, self.normalize_stats, cur_state, is_bimanual=self.bimanual, relative_joint=self.is_relative)
        elif self.embodiment.startswith("libero") and self.action_style == "delta_eef":
            action_chunk = infer_utils.process_libero_delta_eef_action(raw_actions, self.normalize_stats)

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


def create_trained_policy(
    basic_config: OverallArguments,
    infer_config: InferArguments,
) -> Policy:

    image_processor = AutoProcessor.from_pretrained(
        basic_config.model.vlm_model_path,
    ).image_processor
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        basic_config.model.vlm_model_path,
        cache_dir=None,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )

    # load 模型 =========================================================================
    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    training_hf_args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    vla = iFlyBotVLA(cfg=basic_config, training_hf_args=training_hf_args)
    
    # 注意，优先用InferConfig配置中的ckpt
    print(f"Load checkpoint from {infer_config.checkpoint_path}")
    vla.load_from_checkpoint(infer_config.checkpoint_path, vlm_only=infer_config.load_only_vlm, old_version=infer_config.old_version_ckpt)
    # 进行参数冻结 & lora配置
    vla.check_freeze_and_lora()

    vla = vla.to("cuda").to(torch.bfloat16)
    vla.eval()

    return Policy(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=vla,
        cfg=basic_config,
        infer_cfg=infer_config,
        pytorch_device="cuda",
    )

