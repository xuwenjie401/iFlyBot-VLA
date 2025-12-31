import os
import copy
import json
import random
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Tuple, Any, Protocol, Union, Type
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
import transformers
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, AutoProcessor

from config.base_configs import DatasetsArguments, ManipDataArguments, VQADataArguments, TaskArguments, OverallArguments
from config.robot_dataset_basic import *
from config.token_configs import *
from datasets.qwenvl.rope2d import get_rope_index_25
from datasets.rlds.rlds_dataset import make_interleaved_rlds_dataset


logger = logging.getLogger(__name__)

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    

class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...
    
    
def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor
    

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    task_cfg: TaskArguments = TaskArguments(),
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    real_image_flags: List = [],
    fast_ids: List = [],
    mask_virtual_view: bool = False,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    # tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []
    attn_mask = []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            # for fast-aux, we need to insert fast_ids after FAST_START_TOKEN_INDEX
            if (task_cfg.train == "train" or task_cfg.train == "check_gt") and (task_cfg.use_fast_aux) and (role not in ["user", "system"]):
                modified_id = []
                for id in encode_id:
                    modified_id.append(id)
                    if id == FAST_START_TOKEN_INDEX:
                        # insert fast_ids after FAST_START_TOKEN_INDEX
                        modified_id.extend(fast_ids)
                encode_id = modified_id

            # for infer data test
            if task_cfg.train == "infer" and role not in ["user", "system"]:
                # remove all tokens after ASSISTANT_TOKEN_INDEX + MAGIC_TOKEN_INDEX
                modified_id = []
                mark = False
                for id in encode_id:
                    modified_id.append(id)
                    if id == ASSISTANT_TOKEN_INDEX:
                        mark = True
                    if id == MAGIC_TOKEN_INDEX and mark == True:
                        break
                encode_id = modified_id
            # check_gt mode, remove eos+\n
            elif task_cfg.train == "check_gt" and role not in ["user", "system"]:
                encode_id = encode_id[:-2]

            input_id += encode_id

            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

        # make attention mask for virtual view
        if mask_virtual_view:
            attn_masks = []
            if len(real_image_flags) > 0:
                for k in range(len(input_ids)):
                    pos = 0
                    img_flag_cursor = 0
                    cur_id = input_ids[k]
                    L = len(cur_id)
                    attn_mask = [1] * len(cur_id)
                    while pos < L:
                        if cur_id[pos] == VISION_START_TOKEN_INDEX:
                            block_start = pos
                            pos2 = pos + 1
                            # find the corresponding vision_end
                            while pos2 < L and cur_id[pos2] != VISION_END_TOKEN_INDEX:
                                pos2 += 1
                            if pos2 < L and cur_id[pos2] == VISION_END_TOKEN_INDEX:
                                block_end = pos2
                                
                                # <image> first appearance
                                if img_flag_cursor < len(real_image_flags) and (not bool(real_image_flags[img_flag_cursor])):
                                    # mask the entire block (including vision_start / vision_end)
                                    for k in range(block_start, block_end + 1):
                                        attn_mask[k] = 0
                                img_flag_cursor += 1
                                # skip the block
                                pos = block_end + 1
                                continue
                        pos += 1
                    attn_masks.append(attn_mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    if mask_virtual_view:
        attn_masks = torch.tensor(attn_masks, dtype=torch.int)
    else:
        attn_masks = None

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attn_masks,
    )    


def _pad_to_len(arr: np.ndarray, target_len: int, dim_idx: int):
    """
    Pad the array `arr` in the `dim_idx` dimension to `target_len` (only pad 0 at the end).
    Support 1D/2D (higher dimensions are also supported as long as `dim_idx` is valid).
    """
    if dim_idx >= arr.ndim:
        raise ValueError(f"Invalid dim_idx={dim_idx} for shape={arr.shape}")

    cur = arr.shape[dim_idx]
    if cur >= target_len:
        return arr

    pad_after = target_len - cur
    pad_width = [(0, 0)] * arr.ndim
    pad_width[dim_idx] = (0, pad_after)
    return np.pad(arr, pad_width=tuple(pad_width), mode="constant", constant_values=0)


def _pad_front_back(arr: np.ndarray, dim_idx: int, pad_front: int, pad_back: int):
    """
    pad user-defined length in dim_idx dimension
    """
    if dim_idx >= arr.ndim:
        raise ValueError(f"Invalid dim_idx={dim_idx} for shape={arr.shape}")

    pad_width = [(0, 0)] * arr.ndim
    pad_width[dim_idx] = (pad_front, pad_back)
    return np.pad(arr, pad_width=tuple(pad_width), mode="constant", constant_values=0)


@dataclass
class UniversalActionBatchTransform:
    base_tokenizer: PreTrainedTokenizerBase
    lam_image_transform: ImageTransform
    image_processor: None
    latent_action_tokenizer: None
    FAST_action_tokenizer: None
    get_rope_index: None
    data_manip_cfg: ManipDataArguments
    task_cfg: TaskArguments
    predict_stop_token: bool = True
    
    """返回sample/entry 对标qwen字段
    position_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
    input_ids,

    """
    def process_image_unified(self, image):
        visual_processed = self.image_processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def __call__(
        self, 
        rlds_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        """Converts a RLDS batch to the format expected by all kinds of collator/models."""
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        config_is_bimanual = self.data_manip_cfg.bimanual
        style = self.data_manip_cfg.action_style              
        task = self.task_cfg.train                            

        lam_source = self.task_cfg.lam_source
        # logger.info(f'set bimanual data-format: {is_bimanual}')
        # TODO: hyper-parameter, single-arm action's dim max-length
        single_arm_act_dim = 10

        dataset_name = rlds_batch['dataset_name'].decode("utf-8")
        pad_fields = ['action', 'state']
        if MANIP_DATASETS_BASIC_SETTINGS[dataset_name]['is_bimanual']:
            if style == "eef_pose" or style == "joint":
                # 把左右各 pad 到 single_arm_act_dim (10)
                for pad_field in pad_fields:
                    arr = rlds_batch[pad_field]  # numpy.ndarray

                    # state 是 1D：不能做左右拆分（除非你明确 state 本身是 [left,right] 拼起来的）
                    if pad_field == "state":
                        half = arr.shape[0] // 2
                        left = arr[:half]      # e.g. (8,)
                        right = arr[half:]     # e.g. (8,)

                        left_padded = _pad_to_len(left, target_len=single_arm_act_dim, dim_idx=0)   # -> (10,)
                        right_padded = _pad_to_len(right, target_len=single_arm_act_dim, dim_idx=0) # -> (10,)

                        rlds_batch[pad_field] = np.concatenate([left_padded, right_padded], axis=0) # -> (20,)
                        continue

                    # 非 state：默认是 (B, D) 的 2D，沿 dim=1 拆左右
                    if arr.ndim != 2:
                        raise ValueError(f"{pad_field} expected 2D array in bimanual split, got shape={arr.shape}")

                    half = arr.shape[1] // 2
                    left = arr[:, :half]
                    right = arr[:, half:]

                    left_padded = _pad_to_len(left, target_len=single_arm_act_dim, dim_idx=1)
                    right_padded = _pad_to_len(right, target_len=single_arm_act_dim, dim_idx=1)

                    rlds_batch[pad_field] = np.concatenate([left_padded, right_padded], axis=1)
            else:
                raise ValueError(f"[wjxu sorry] bimanual--openvla  not implemented yes")

        # 单臂数据集
        else:
            if style == "delta_eef":
                # 把所有数据 pad 到 20 维
                for pad_field in pad_fields:
                    arr = rlds_batch[pad_field]

                    # state 是 1D：pad dim=0
                    if pad_field == "state":
                        rlds_batch[pad_field] = _pad_to_len(arr, target_len=20, dim_idx=0)
                    else:
                        # 其他字段默认 2D：pad dim=1
                        rlds_batch[pad_field] = _pad_to_len(arr, target_len=20, dim_idx=1)

            else:
                # 先确保至少 single_arm_act_dim
                for pad_field in pad_fields:
                    arr = rlds_batch[pad_field]
                    if pad_field == "state":
                        rlds_batch[pad_field] = _pad_to_len(arr, target_len=single_arm_act_dim, dim_idx=0)
                    else:
                        rlds_batch[pad_field] = _pad_to_len(arr, target_len=single_arm_act_dim, dim_idx=1)

                # 如果最终需要输出双臂格式：把单臂扩成双臂（随机放左/右，另一侧补零）
                if config_is_bimanual:
                    # 随机 pad 一边
                    side = random.random() if self.data_manip_cfg.single_to_bimanual_random_pad else 0

                    for pad_field in pad_fields:
                        arr = rlds_batch[pad_field]
                        dim_idx = 0 if pad_field == "state" else 1

                        if side < 0.5:
                            # 放左臂：在后面补一个 arm
                            rlds_batch[pad_field] = _pad_front_back(arr, dim_idx=dim_idx,
                                                                pad_front=0, pad_back=single_arm_act_dim)
                        else:
                            # 放右臂：在前面补一个 arm
                            rlds_batch[pad_field] = _pad_front_back(arr, dim_idx=dim_idx,
                                                                pad_front=single_arm_act_dim, pad_back=0)

        state_dropout = self.data_manip_cfg.state_dropout

        if state_dropout == "all":
            rlds_batch["state"] = np.zeros_like(rlds_batch["state"])
        elif state_dropout == "random":
            coin = random.random()
            if coin < 0.5:
                # 保留current state
                if rlds_batch["state"].ndim == 3:
                    rlds_batch["state"][:, 0, :] = np.zeros_like(rlds_batch["state"][:, 0, :])
                # 维度为2时，保留即不操作
            else:
                # state全部置0
                rlds_batch["state"] = np.zeros_like(rlds_batch["state"])

        img_head = Image.fromarray(rlds_batch["observation"]["image_primary"])
        img_wrist = None
        img_wrist2 = None
        real_image_flags = [True, False, False]
        if MANIP_DATASETS_BASIC_SETTINGS[dataset_name]["image_obs_keys"]["wrist"] is not None:
            img_wrist = Image.fromarray(rlds_batch["observation"]["image_wrist"])
            real_image_flags[1] = True
        else:
            if self.data_manip_cfg.image_pad_other_view:
                img_wrist = img_head
            else:
                img_wrist = Image.fromarray(np.zeros_like(img_head, dtype=np.uint8))
        if MANIP_DATASETS_BASIC_SETTINGS[dataset_name]["image_obs_keys"]["wrist_2"] is not None:
            img_wrist2 = Image.fromarray(rlds_batch["observation"]["image_wrist_2"])
            real_image_flags[2] = True
        else:
            if self.data_manip_cfg.image_pad_other_view:
                img_wrist2 = img_wrist
            else:
                img_wrist2 = Image.fromarray(np.zeros_like(img_head, dtype=np.uint8))

        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        action_tokens = ''
        task_lap_prefix = ''
        task_fast_prefix = ''
        fast_ids = []
        lap_img_t0 = None
        lap_img_tk = None
        if self.task_cfg.use_lap_aux:
            task_lap_prefix = LAP_TASK_TOKEN
            action_tokens += LAP_START_TOKEN

            num_lap_token = 4 if lam_source == "univla" else 8
            for j in range(num_lap_token):
                action_tokens += LAP_PAD_TOKEN

            action_tokens += LAP_END_TOKEN

            # 仅有纯推理时，不需要lap token
            if task != "infer":
                if lam_source == "univla":
                    lap_img_t0 = copy.copy(img_head)
                    lap_img_tk = Image.fromarray(rlds_batch["observation"]["image_future"])
                else:  # "lapa"
                    lap_img_t0 = img_head.resize((224, 224), Image.BICUBIC)
                    lap_img_tk = Image.fromarray(rlds_batch["observation"]["image_future"]).resize((224, 224), Image.BICUBIC)

        if self.task_cfg.use_fast_aux:
            task_fast_prefix += FAST_TASK_TOKEN
            fast_tokens = np.array(self.FAST_action_tokenizer(rlds_batch["action"]))
            fast_ids = (fast_tokens[0] + FAST_TOKEN_INDEX_OFFSET).tolist()
            action_tokens += FAST_START_TOKEN
            # NOTE: 因为100000开始的2048个token并不是special token，所以后续直接在input-id上操作
            action_tokens += FAST_END_TOKEN

        # TODO
        state_pad_tokens = ""
        state_pad_tokens += STATE_PAD_TOKEN

        domain_prefix = ""
        if self.task_cfg.domain_aware:
            for i in range(32):
                domain_prefix += DOMAIN_PAD_TOKEN

        conversation = [
            {"from": "human", "value": f"<image><image><image>What action should the robot take to {lang}?{task_lap_prefix}{domain_prefix}{state_pad_tokens}{task_fast_prefix}"},
            {"from": "gpt", "value": action_tokens},
        ]

        # img_head.save("/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/head.png")
        # img_wrist.save("/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/wrist.png")
        # img_wrist2.save("/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/wrist2.png")
        # Image.fromarray(rlds_batch["observation"]["image_future"]).save("/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/future.png")

        images = [img_head, img_wrist, img_wrist2]
        
        # 这里已经处理为tensor
        results = [self.process_image_unified(im) for im in images]
        image, grid_thw = map(list, zip(*results))

        # torch.cuda.synchronize()
        # mark4 = time.perf_counter()
        # delta_process_img = mark4 - mark3
        # print(f"process img cost time: {delta_process_img:.4f}s")

        grid_thw_merged = copy.deepcopy(grid_thw)

        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        # 重点按照qwen的处理流程
        chat_sources = [conversation]
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.base_tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
            real_image_flags=real_image_flags,
            task_cfg=self.task_cfg,
            fast_ids=fast_ids,
            mask_virtual_view=self.data_manip_cfg.mask_virtual_view,
        )

        position_ids, _ = self.get_rope_index(
            self.image_processor.merge_size,
            data_dict['input_ids'],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict['action_chunk'] = torch.from_numpy(rlds_batch['action'])
        data_dict['state'] = torch.from_numpy(rlds_batch['state'])

        data_dict['position_ids'] = position_ids

        data_dict['pixel_values'] = torch.cat(image, dim=0)
        data_dict['image_grid_thw'] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thw], dim=0
        )


        data_dict["lap_im0"] = lap_img_t0
        data_dict["lap_imk"] = lap_img_tk

        if self.task_cfg.domain_aware:
            data_dict["domain_id"] = dataset_domain_ids[dataset_name]

        return data_dict
    

class UniversalRLDSDataset(IterableDataset):
    def __init__(
        self,
        cfg: OverallArguments,
        batch_transform: UniversalActionBatchTransform,
        image_aug: bool = False,
        normalize_only: bool = False,
    ) -> None:
        
        manip_data_cfg = cfg.datasets.manip_data
        shuffle_buffer_size = manip_data_cfg.rlds_shuffle_buffer_size
        resize_resolution = manip_data_cfg.resize_resolution
        self.batch_transform = batch_transform

        # If applicable, enable image augmentations
        image_aug_kwargs = None
        if image_aug:
            image_aug_kwargs = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )

        # Initialize RLDS Dataset
        datasets_list = MANIP_DATASETS_USE[manip_data_cfg.dataset_use]
        self.dataset, self.dataset_length, self.dataset_statistics = make_interleaved_rlds_dataset(
            weighted_datasets_list=datasets_list,
            cfg=cfg,
            shuffle_buffer_size=shuffle_buffer_size,
            normalize_only=normalize_only,
            stats_files_root=DEFAULT_STATS_FILES_ROOT,
            train= (cfg.task.train == "train"),
            resize_size=resize_resolution,
            batch_size=None,
            balance_weights=True,
            image_aug_kwargs=image_aug_kwargs,
            traj_transform_threads=len(datasets_list),
            traj_read_threads=len(datasets_list),
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


@dataclass
class DataCollatorForRobotDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    latent_action_model: None
    lam_image_transform: None
    cfg: OverallArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, lap_im0, lap_imk = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "lap_im0", "lap_imk")
        )

        # 如果确实有lap任务
        if lap_im0[0] is not None:
            lam_source = self.cfg.task.lam_source
            batch_lap_video = []
            for idx in range(len(lap_im0)):
                initial_pixel_values = self.lam_image_transform(lap_im0[idx])
                target_pixel_values = self.lam_image_transform(lap_imk[idx])
                if lam_source == "univla":
                    video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0)
                    batch_lap_video.append(video)
                else:
                    video = torch.stack([initial_pixel_values, target_pixel_values], dim=1).unsqueeze(0)
                    batch_lap_video.append(video)

            batch_lap_video = torch.cat(batch_lap_video, dim=0).to(self.latent_action_model.device)
            # 这里 1）拼出文本，encode出token-id；2）直接根据Index对应写出id， 目前选择1
            addition_lap_token_ids = []
            if lam_source == "univla":
                batch_la_ids = self.latent_action_model.vq_encode(batch_lap_video)['indices']

                for act_indices in batch_la_ids:
                    action_vocab = [f'<ACT_{x.item()}>' for x in act_indices]
                    action_text = ""
                    for word in action_vocab:
                        action_text += word
                    action_tokens = self.tokenizer.encode(action_text)
                    addition_lap_token_ids.append(action_tokens)
            else:
                _, batch_la_ids, _ = self.latent_action_model.inference(batch_lap_video)
                batch_la_ids = batch_la_ids.tolist()
           
                for act_indices in batch_la_ids:
                    action_vocab = [f'<ACT_{x}>' for x in act_indices]
                    action_text = ""
                    for word in action_vocab:
                        action_text += word
                    action_tokens = self.tokenizer.encode(action_text)
                    addition_lap_token_ids.append(action_tokens)
            
            complete_input_ids = []
            for i, ids in enumerate(input_ids):
                ids = ids.squeeze(0)
                add_lap_tokens = torch.tensor(addition_lap_token_ids[i])

                # 找到 开始和结束 的位置
                pos_start = torch.argmax((ids == LAP_START_TOKEN_INDEX).int()).item()
                pos_end = torch.argmax((ids == LAP_END_TOKEN_INDEX).int()).item()

                if pos_start > 0 and pos_end > pos_start:
                    new_ids = torch.cat([ids[:pos_start+1], add_lap_tokens, ids[pos_end:]])
                    new_ids.unsqueeze(0)
                else:
                    # 如果没找到，就不插
                    new_ids = ids
                complete_input_ids.append(new_ids)
            input_ids = complete_input_ids

            complete_labels = []
            for i, tgts in enumerate(labels):
                tgts = tgts.squeeze(0)
                add_lap_tokens = torch.tensor(addition_lap_token_ids[i])

                # 找到 开始和结束 的位置
                pos_start = torch.argmax((tgts == LAP_START_TOKEN_INDEX).int()).item()
                pos_end = torch.argmax((tgts == LAP_END_TOKEN_INDEX).int()).item()

                if pos_start > 0 and pos_end > pos_start:
                    new_tgts = torch.cat([tgts[:pos_start+1], add_lap_tokens, tgts[pos_end:]])
                    new_tgts.unsqueeze(0)
                else:
                    # 如果没找到，就不插
                    new_tgts = tgts
                complete_labels.append(new_tgts)
            labels = complete_labels

        act_chunk_str = 'action_chunk'
        act_chunk = None
        if act_chunk_str in instances[0]:
            act_chunk = [inst[act_chunk_str] for inst in instances]

        state_str = 'state'
        state = None
        if state_str in instances[0]:
            state = [inst[state_str] for inst in instances]

        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]

        if self.cfg.datasets.manip_data.mask_virtual_view:
            # NOTE: 这里mask  pad  0
            attention_masks = [inst["attention_mask"].squeeze(0) for inst in instances]
            attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            attention_masks = attention_masks[:, : self.tokenizer.model_max_length]
        else:
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id)

        # action chunk 先用 stack
        if act_chunk is not None:
            act_chunk = torch.stack(act_chunk, dim=0)
        if state is not None:
            state = torch.stack(state, dim=0)

        

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            attention_mask=attention_masks,
            action_chunk=act_chunk,
            state=state,
            
        )
        if self.cfg.task.domain_aware:
            domain_id = [instance["domain_id"] for instance in instances]
            domain_id = torch.tensor(domain_id, dtype=torch.long)
            batch["domain_id"] = domain_id

        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["position_ids"] = position_ids

        return batch


class EmbodiedLazyQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer, 
        image_processor,
        cfg: OverallArguments,
    ):
        super(EmbodiedLazyQADataset, self).__init__()

        dataset_list = data_list(data_args.vqa_dataset_use.split(","))
        rank0_print(f"Loading datasets: {dataset_list}")
        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_25

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_processor.max_pixels = cfg.datasets.manip_data.max_pixels
        self.image_processor.min_pixels = cfg.datasets.manip_data.min_pixels
        self.image_processor.size["longest_edge"] = cfg.datasets.manip_data.max_pixels
        self.image_processor.size["shortest_edge"] = cfg.datasets.manip_data.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = self.image_processor
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e


    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        # sources为列表，len（固定?）为1 ；  sources[0]: dict
        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )

        # logger.info(f'new qa-entry:\nconversation: {chat_sources[0]}\ndata-dict: {data_dict}')

        return data_dict    
    
    
@dataclass
class VQADataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["position_ids"] = position_ids
        return batch


def make_embodied_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    image_processor, 
    cfg: OverallArguments,
    normalize_only: bool = False,
) -> Dict:
    
    task_cfg = cfg.task
    
    latent_action_model = None
    lam_source = task_cfg.lam_source
    # if use_lap_aux is True, then load lam
    if task_cfg.use_lap_aux:
        print(f"Loading LAM, source-{lam_source}......")
        # get the local rank of the current process
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    
        if lam_source == "univla":
            from tasks.latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
            # TODO: write in cfg
            lam_path = task_cfg.univla_lam_path
            codebook_size: int = 16
            lam_model_dim: int = 768
            lam_latent_dim: int = 128
            lam_patch_size: int = 14
            lam_enc_blocks: int = 12
            lam_dec_blocks: int = 12
            lam_num_heads: int = 12
            latent_action_model = ControllableDINOLatentActionModel(
                in_dim=3,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                num_latents=codebook_size,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
                dropout=0.,
            )

            lam_ckpt = torch.load(lam_path, map_location="cpu")['state_dict']
            new_ckpt = {}
            for key in lam_ckpt.keys():
                new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

            latent_action_model.load_state_dict(new_ckpt, strict=True)
            latent_action_model = latent_action_model.to(device).eval()

        elif lam_source == "lapa":
            from tasks.lapa_model.genie.lapa import LatentActionQuantization
            lam_path = task_cfg.lapa_lam_path
            latent_action_model = LatentActionQuantization(
                dim = 1024,
                quant_dim = 32,
                codebook_size = 32,
                image_size = 224,
                patch_size = 32,
                spatial_depth = 8,
                temporal_depth = 8,
                dim_head = 64,
                heads = 16,
                code_seq_len = 8,
            )
            
            lam_ckpt = torch.load(lam_path, map_location="cpu")['state_dict']
            latent_action_model.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=False)
            latent_action_model = latent_action_model.to(device).eval()
            
        print(f'LAM-{lam_source} Loaded')


    FAST_tokenizer = None
    if cfg.task.use_fast_aux:
        print("Loading FAST tokenizer ...... ")
        fast_path = "/wx-mix01/sppro/permanent/wjxu22/model/FAST"
        FAST_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)
        print("FAST tokenizer loaded")

    batch_trans = UniversalActionBatchTransform(
        base_tokenizer=tokenizer,
        image_processor=image_processor,
        lam_image_transform=transforms.ToTensor(),
        get_rope_index=get_rope_index_25,
        data_manip_cfg=cfg.datasets.manip_data,
        task_cfg=cfg.task,
        latent_action_tokenizer=latent_action_model,
        FAST_action_tokenizer=FAST_tokenizer,
    )
    train_dataset = UniversalRLDSDataset(
        batch_transform=batch_trans,
        cfg=cfg,
        image_aug=False,
        normalize_only=normalize_only,
    )

    data_collator = DataCollatorForRobotDataset(
        tokenizer=tokenizer,
        lam_image_transform=transforms.ToTensor(),
        latent_action_model=latent_action_model,
        cfg=cfg,
    )

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def make_vqa_data_module(tokenizer: transformers.PreTrainedTokenizer, image_processor, cfg):
    vqa_dataset = EmbodiedLazyQADataset(tokenizer=tokenizer, image_processor=image_processor, cfg=cfg)

    data_collator = VQADataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=vqa_dataset, eval_dataset=None, data_collator=data_collator
    )
    