import copy
import inspect
import json
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
tf.config.set_visible_devices([], "GPU")

import iFlyBotVLA.datasets.rlds.data_utils as data_utils
from iFlyBotVLA.config.robot_dataset_basic import *
from config.base_configs import OverallArguments, DatasetsArguments, ManipDataArguments, TaskArguments
from datasets.rlds.standardize_transforms import STANDARDIZE_SE3_TRANSFORMS, STANDARDIZE_JOINT_TRANSFORMS, STANDARDIZE_DELTA_EE_TRANSFORMS

def make_universal_trajectory_from_rlds(
    dataset_name: str,
    manip_cfg: ManipDataArguments,
    dataset_statistics = None,
    normalize_only: bool = False,
    train: bool = True,
    shuffle: bool = False,
    num_parallel_reads: int = 8,  # tf.data.AUTOTUNE
    num_parallel_calls: int = 8,  # tf.data.AUTOTUNE
):  
    """
    Restructure a single RLDS dataset into a unified format.
    """
    
    LANGUAGE_KEY = "language_instruction"
    
    action_chunk_size = manip_cfg.action_chunk_size
    duration = manip_cfg.action_duration
    style = manip_cfg.action_style
    control_reference = manip_cfg.control_reference
    rotation_representation = manip_cfg.rotation_representation
    
    info = MANIP_DATASETS_BASIC_SETTINGS[dataset_name]
    is_bimanual = info["is_bimanual"]
    frequency = info["frequency"]
    
    standardize_fn = None
    REQUIRED_KEYS = {"observation", "action", LANGUAGE_KEY}
    if style == "eef_pose":
        standardize_fn = STANDARDIZE_SE3_TRANSFORMS[dataset_name]
        REQUIRED_KEYS = {"observation", LANGUAGE_KEY, "left_eef_se3", "right_eef_se3", "left_gripper_action", "right_gripper_action"} \
            if is_bimanual else {"observation", LANGUAGE_KEY, "eef_se3", "gripper_action"}
    elif style == "joint":
        standardize_fn = STANDARDIZE_JOINT_TRANSFORMS[dataset_name]
        REQUIRED_KEYS = {"observation", LANGUAGE_KEY, "left_action", "right_action", "left_state", "right_state"} \
            if is_bimanual else {"observation", LANGUAGE_KEY, "action", "state"}
    elif style == "delta_eef":
        standardize_fn = STANDARDIZE_DELTA_EE_TRANSFORMS[dataset_name]
    
    if standardize_fn is None:
        raise ValueError(f"No {style}-standardize-fn impl for {dataset_name}")
    
    def restructure_eef_pose(traj):
        traj = standardize_fn(traj)
        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(f"Style-{style} Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Check standardize_fn")
        
        traj_len = tf.shape(traj[LANGUAGE_KEY])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in info["image_obs_keys"].items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]
        
        # TODO:
        iflytek_cut = True if "cut_mark" in traj else False
        
        # 预设时间段 的 原数据集频率下 action个数
        window_size = duration * frequency
        action_offset = tf.range(window_size+1)[None, :]
        traj_start_idx, traj_end_idx = 0, traj_len
        if iflytek_cut:
            traj_start_idx, traj_end_idx = 0, traj_len-30
        start_indices = tf.range(traj_start_idx, traj_end_idx)[:, None]
        
        action_chunk_indices = start_indices + action_offset
        bounded_action_chunk_indices = tf.maximum(tf.minimum(action_chunk_indices, traj_len-1), 0)
        
        traj = data_utils.make_se3_action_chunk(traj, bounded_action_chunk_indices, action_chunk_size, is_bimanual)
        traj = data_utils.make_se3_current_state(traj, is_bimanual)
        traj = data_utils.make_relative_se3_state_action(traj, control_reference, is_bimanual)
        traj = data_utils.flatten_relative_se3_state_action(traj, control_reference, rotation_representation, is_bimanual)
        
        # prepare for multi-level task instructions: e.g. main_task --> sub_task
        task = {}
        task[LANGUAGE_KEY] = traj.pop(LANGUAGE_KEY)
        
        dataset_name_entry = tf.repeat(dataset_name, traj_len)
        state = traj["state"]
        if iflytek_cut:
            new_obs = {k: v[traj_start_idx : traj_end_idx] for k, v in new_obs.items()}
            task = {k: v[traj_start_idx : traj_end_idx] for k, v in task.items()}
            dataset_name_entry = dataset_name_entry[traj_start_idx : traj_end_idx]
            state = state[traj_start_idx: traj_end_idx]
        
        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "state": tf.cast(state, tf.float32),
            "dataset_name": dataset_name_entry,
        }
        
        return traj
    
    def restructure_joint(traj):
        traj = standardize_fn(traj)
        
        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(f"Style-{style} Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Check standardize_fn")
        
        traj_len = tf.shape(traj[LANGUAGE_KEY])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in info["image_obs_keys"].items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]
        
        # TODO:
        iflytek_cut = True if "cut_mark" in traj else False
        
        window_size = duration * frequency
        action_offset = tf.range(window_size+1)[None, :]
        traj_start_idx, traj_end_idx = 0, traj_len
        if iflytek_cut:
            traj_start_idx, traj_end_idx = 0, traj_len-30
        start_indices = tf.range(traj_start_idx, traj_end_idx)[:, None]
        
        action_chunk_indices = start_indices + action_offset
        bounded_action_chunk_indices = tf.maximum(tf.minimum(action_chunk_indices, traj_len-1), 0)

        traj = data_utils.make_joint_action_state(traj, control_reference, bounded_action_chunk_indices, action_chunk_size, is_bimanual)
        
        task = {}
        task[LANGUAGE_KEY] = traj.pop(LANGUAGE_KEY)
        
        dataset_name_entry = tf.repeat(dataset_name, traj_len)
        state = traj["state"]
        if iflytek_cut:
            new_obs = {k: v[traj_start_idx : traj_end_idx] for k, v in new_obs.items()}
            task = {k: v[traj_start_idx : traj_end_idx] for k, v in task.items()}
            dataset_name_entry = dataset_name_entry[traj_start_idx : traj_end_idx]
            state = state[traj_start_idx: traj_end_idx]
        
        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "state": tf.cast(state, tf.float32),
            "dataset_name": dataset_name_entry,
        }
        
        return traj

    def restructure_delta_eef(traj):
        traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(f"Style-{style} Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Check standardize_fn")
        
        traj_len = tf.shape(traj[LANGUAGE_KEY])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in info["image_obs_keys"].items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]
                
        traj = data_utils.make_delta_eef_action_chunk(traj, action_chunk_size)
        
        task = {}
        task[LANGUAGE_KEY] = traj.pop(LANGUAGE_KEY)
        dataset_name_entry = tf.repeat(dataset_name, traj_len)
        
        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "state": tf.cast(traj["state"], tf.float32),
            "dataset_name": dataset_name_entry,
        }
        
        return traj
    
    is_union = info["union"]
    dataset = None
    split = "train"
    if normalize_only: shuffle = False
    
    if is_union:
        dataset = data_utils.build_union_rlds_dataset(
            dataset_name, info["path"], num_parallel_reads=num_parallel_reads)        
    else:

        builder = tfds.builder(dataset_name, data_dir=os.path.dirname(info["path"]))
        
        val_split = next((s for s in ("val", "test", "eval") if s in builder.info.splits), None)
        if val_split:
            split = "train" if train else val_split
            logging.info(f"\n################\n{dataset_name} has val split-{val_split}##############")
        else:
            split = "train[:95%]" if train else "train[95%:]"
            logging.info(f"\n################\n{dataset_name} doesn't have the split in (val, test, eval) ##############")
        if normalize_only: split = "all"
        
        dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
        
    # restructure
    if style == "eef_pose":
        dataset = dataset.traj_map(restructure_eef_pose, num_parallel_calls)
    elif style == "joint":
        dataset = dataset.traj_map(restructure_joint, num_parallel_calls)
        # for traj in dataset:
        #     traj = restructure_joint(traj)
    elif style == "delta_eef":
        dataset = dataset.traj_map(restructure_delta_eef, num_parallel_calls)
    else:
        raise ValueError(f"Style-{style} is not supported.")
    
    if normalize_only:
        dataset_statistics = data_utils.make_statistics_unified(
            dataset=dataset,
            dataset_name=dataset_name,
            style=style,
            control_reference=control_reference,
            timestep_conditioned=manip_cfg.timestep_conditioned_norm,
            mark=manip_cfg.norm_unique_mark,
        )

        return dataset, dataset_statistics
    
    dataset = dataset.traj_map(
        partial(
            data_utils.normalize_unified,
            metadata=dataset_statistics,
            style=style,
            rotation_representation=rotation_representation,
            timestep_conditioned=manip_cfg.timestep_conditioned_norm,
            control_reference=control_reference,
            is_bimanual=is_bimanual,
        ),
        num_parallel_calls,
    )
    # debug
    # for traj in dataset:
    #     traj = data_utils.normalize_unified(
    #         traj, metadata=dataset_statistics, style=style, control_reference=control_reference, is_bimanual=is_bimanual,)
    
    return dataset, dataset_statistics


def apply_universal_frame_transforms(
    dataset: dl.DLataset,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    train: bool = True,
    num_parallel_calls: int = 8,
) -> dl.DLataset:
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    def decode_images(traj):
        obs = traj["observation"]
        image_names = {key[6:] for key in obs if key.startswith("image_")}
        for name in image_names:
            image = obs[f"image_{name}"]
            if image.dtype == tf.string:
                if tf.strings.length(image) == 0:
                    # this is a padding image
                    image = tf.zeros((*resize_size, 3), dtype=tf.uint8)
                else:
                    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
            elif image.dtype != tf.uint8:
                raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
            
            if resize_size is not None:
                image = dl.transforms.resize_image(image, size=resize_size)

            obs[f"image_{name}"] = image

        return traj

    decode_transform = decode_images
    dataset = dataset.frame_map(decode_transform, num_parallel_calls)
    
    # TODO: images augmentation ?

    return dataset
 
        
def make_interleaved_rlds_dataset(
    weighted_datasets_list: List[Tuple],
    cfg: OverallArguments,
    shuffle_buffer_size: int,
    normalize_only: bool = False,
    stats_files_root: Optional[str] = None,
    train: bool = True,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = True,
    image_aug_kwargs: Optional[Dict[str, Any]] = None,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    
    # 把weighted_datasets_list中的dataset, weights 拆成两个独立的list
    datasets_list, sample_weights = zip(*weighted_datasets_list)
    
    dataset_sizes, all_dataset_statistics = [], {}
    # 强制分支  归一化功能 和 训练功能
    if normalize_only:
        for name in datasets_list:
            _, dataset_stats = make_universal_trajectory_from_rlds(
                name, cfg.datasets.manip_data, normalize_only=normalize_only, train=train, shuffle=False
            ) 
        # TODO
        return None, None, None
    
    manip_cfg = cfg.datasets.manip_data
    dataset_sizes, all_dataset_statistics = data_utils.get_stats_from_files(
        stats_files_root,
        datasets_list,
        manip_cfg.action_style,
        manip_cfg.control_reference,
        manip_cfg.timestep_conditioned_norm,
        manip_cfg.norm_unique_mark,
    )
        
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])

    # Balance and Normalize Weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    data_utils.pprint_data_mixture(datasets_list, sample_weights)
    dataset_length = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    
    # Allocate Threads based on Weights
    threads_per_dataset = data_utils.allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = data_utils.allocate_threads(traj_read_threads, sample_weights)
    logging.info("Threads per Dataset: %s", threads_per_dataset)
    logging.info("Reads per Dataset: %s", reads_per_dataset)
    # Construct Datasets
    logging.info("Constructing datasets...")
    
    valid_datasets = []
    valid_indices = []
    for idx, (dataset_name, statistics, threads, reads) in enumerate(
        zip(datasets_list, all_dataset_statistics, threads_per_dataset, reads_per_dataset)
    ):
        try:
            statistics = data_utils.tree_map(np.array, statistics)
            dataset, _ = make_universal_trajectory_from_rlds(
                dataset_name, cfg.datasets.manip_data, statistics, normalize_only=normalize_only, train=train, shuffle=False, num_parallel_calls=threads, num_parallel_reads=reads)
            
            info = MANIP_DATASETS_BASIC_SETTINGS[dataset_name]
            duration = manip_cfg.action_duration
            frequency = info["frequency"]
            window_size = int(duration * frequency)
    
            # dataset = dataset.repeat()

            if cfg.task.use_lap_aux:
                dataset = dataset.traj_map(partial(data_utils.chunk_obs_head_only, window_size=window_size))

            dataset = dataset.flatten(num_parallel_calls=threads)  
            
            valid_datasets.append(dataset)
            valid_indices.append(idx)
        except Exception as e:
            logging.warning(
                "Dataset %s failed to construct and will be skipped. Error: %s",
                dataset_name,
                str(e),
            )
    
    if len(valid_indices) < len(datasets_list):
        # 有被跳过的，重新子集并归一化
        valid_sample_weights = np.array([sample_weights[i] for i in valid_indices], dtype=float)
        valid_sample_weights = valid_sample_weights / valid_sample_weights.sum()
        filtered_dataset_kwargs_list = [datasets_list[i] for i in valid_indices]
        print("Some datasets were skipped. Re-normalized sample weights for remaining datasets:")
        data_utils.pprint_data_mixture(filtered_dataset_kwargs_list, valid_sample_weights)
    else:
        # 没有跳过，沿用之前已经归一化好的 sample_weights
        valid_sample_weights = sample_weights
        
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(valid_datasets, valid_sample_weights, seed=42)

    dataset = apply_universal_frame_transforms(dataset, resize_size=resize_size, train=train)

    dataset = dataset.shuffle(shuffle_buffer_size)

    # [Contract] When training VLA Policies, we let the Collator handle Batching!
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Note =>> Seems to reduce memory usage without affecting speed?
    dataset = dataset.with_ram_budget(1)

    return dataset, dataset_length, all_dataset_statistics

