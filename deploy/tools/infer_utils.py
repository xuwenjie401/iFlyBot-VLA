import sys
from enum import Enum
import math
import os
import time
from typing import Union, Tuple, List, Dict, Optional

import imageio
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import torch
import transformers
from transformers import PreTrainedTokenizer

from src.iFlyBotVLA.config.token_configs import *
from src.iFlyBotVLA.config.base_configs import TaskArguments

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
VIDEO_STR="sppro_video"

from PIL import Image
def resize_image_np_version(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    assert isinstance(resize_size, (int, tuple))
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # PIL expects (W,H)
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize(resize_size, resample=Image.LANCZOS)  # lanczos3 ~ Pillow.LANCZOS

    # Back to numpy
    img_resized = np.array(pil_img, dtype=np.uint8)
    return img_resized


def state_qwxyz_to_qxyzw(state: np.ndarray, is_bimanual: bool) -> np.ndarray:
    """
    Convert quaternion from wxyz to xyzw format.
    state 16维, left-(x,y,z,qw,qx,qy,qz,grip), right-(x,y,z,qw,qx,qy,qz,grip)
    """
    m_state = state.copy()

    left_qwxyz = m_state[..., 3:7]
    left_qxyzw = np.roll(left_qwxyz, -1, axis=-1)
    m_state[..., 3:7] = left_qxyzw 

    if is_bimanual:
        right_qwxyz = m_state[..., 11:15]
        right_qxyzw = np.roll(right_qwxyz, -1, axis=-1)
        m_state[..., 11:15] = right_qxyzw

    state = m_state
    return state


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def interpolate_state(last_state, cur_state, chunk_size, obs_horizon, relative):
    """
    state: [n, 8]
    """
    
    last_pos = last_state[:3]
    last_quat = last_state[3:7]
    last_gripper = last_state[-1]
    cur_pos = cur_state[:3]
    cur_quat = cur_state[3:7]
    cur_gripper = cur_state[-1]
    
    # 端点时间：0 -> last, 1 -> cur
    key_times = np.array([0.0, 1.0], dtype=float)
    # 全部等分的时间点（包含两端）
    all_ts = np.linspace(0.0, 1.0, chunk_size + 1, dtype=float)

    if relative:
        # 取“靠近 cur 的最后 obs_horizon 个点（不含 1.0 本身）”
        # 例：chunk_size=5, all_ts=[0,0.2,0.4,0.6,0.8,1.0]
        # 取 [0.2,0.4,0.6,0.8] 中最后 obs_horizon 个（比如 3 个就是 [0.4,0.6,0.8]）
        query_ts = all_ts[-(obs_horizon + 1):-1]
    else:
        # 绝对位姿情况下，会包含current
        query_ts = all_ts[-obs_horizon:]

    # 位置线性插值
    pos_query = (1 - query_ts[:, None]) * last_pos[None, :] + query_ts[:, None] * cur_pos[None, :]

    # 四元数 Slerp（注意 SciPy 四元数格式是 [x, y, z, w]）
    rot_key = Rotation.from_quat(np.vstack([last_quat, cur_quat]))
    slerp = Slerp(key_times, rot_key)
    rot_query = slerp(query_ts)
    quat_query = rot_query.as_quat()  # 形状 (K, 4)

    # 第8维标量线性插值（若你不想插值而是取 last/cur，可在此改策略）
    gripper_query = (1 - query_ts) * last_gripper + query_ts * cur_gripper
    gripper_query = gripper_query.reshape(-1, 1)

    # 组装 [pos(3), quat(4), scalar(1)] -> (obs_horizon, 8)
    out = np.concatenate([pos_query, quat_query, gripper_query], axis=1)
    return out


def interpolate_joint_state(last_state, cur_state, chunk_size, obs_horizon, relative):
    """
    state: [n, 8]
    """
    
    # 端点时间：0 -> last, 1 -> cur
    key_times = np.array([0.0, 1.0], dtype=float)
    # 全部等分的时间点（包含两端）
    all_ts = np.linspace(0.0, 1.0, chunk_size + 1, dtype=float)

    if relative:
        # 取“靠近 cur 的最后 obs_horizon 个点（不含 1.0 本身）”
        # 例：chunk_size=5, all_ts=[0,0.2,0.4,0.6,0.8,1.0]
        # 取 [0.2,0.4,0.6,0.8] 中最后 obs_horizon 个（比如 3 个就是 [0.4,0.6,0.8]）
        query_ts = all_ts[-(obs_horizon + 1):-1]
    else:
        # 绝对位姿情况下，会包含current
        query_ts = all_ts[-obs_horizon:]

    # 关节角 直接 线性插值
    state_query = (1 - query_ts[:, None]) * last_state[None, :] + query_ts[:, None] * cur_state[None, :]

    return state_query


def make_relative(seq, ref):
    n = seq.shape[0]

    # [n, dim]  T_base_end(i)
    pos = seq[:, :3]
    quat = seq[:, 3:7]
    gripper = seq[:, -1:]

    # T_base_end0
    ref_pos = ref[:3]
    ref_quat = ref[3:7]

    R_seq = Rotation.from_quat(quat).as_matrix()  # [n, 3, 3]
    R_ref = Rotation.from_quat(ref_quat).as_matrix()  # [3, 3]
    R_ref_inv = R_ref.T
    t_ref_inv = -R_ref_inv @ ref_pos

    # T_end-0_end-i
    rel_R = R_ref_inv @ R_seq
    rel_pos = (R_ref_inv @ pos.T).T + t_ref_inv

    rel_quat = Rotation.from_matrix(rel_R).as_quat() 
    
    # 测试用, 单位变换
    # rel_pos = np.zeros_like(rel_pos)
    # rel_quat = np.zeros_like(rel_quat)
    # rel_quat[..., -1] = np.ones_like(rel_quat[..., -1])

    relative = np.concatenate([rel_pos, rel_quat, gripper], axis=-1)  # [n, 8]
    return relative


def normalize_pose_quat_state(data, norm_stats, is_bimanual: bool, pad_for_20=True):
    high = norm_stats["proprio"]["q99"]
    low = norm_stats["proprio"]["q01"]

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    data_normalized = np.clip(2*(data - low) / (high - low + 1e-8) - 1, -1.5, 1.5)

    mask = np.ones(data.shape[1], dtype=bool)
    skip_range = (3, 7)
    mask[skip_range[0]: skip_range[1]] = False

    if is_bimanual:
        skip_range = (11, 15)
        mask[skip_range[0]: skip_range[1]] = False

    # mask的位置，取原始值
    data_normalized[:, ~mask] = data[:, ~mask]

    # state为8x2=16维，pad为10x2=20维
    if pad_for_20:
        left_state = data_normalized[:, :8]
        left_padded = np.pad(left_state, ((0, 0), (0, 2)), mode='constant', constant_values=0)

        if is_bimanual:
            right_state = data_normalized[:, 8:]
            right_padded = np.pad(right_state, ((0, 0), (0, 2)), mode='constant', constant_values=0)
            data_normalized = np.concatenate([left_padded, right_padded], axis=1)
        else:
            data_normalized = np.pad(left_padded, ((0, 0), (0, 10)), mode='constant', constant_values=0)

    return data_normalized


def normalize_joint_state(data, norm_stats, is_bimanual: bool, pad_for_20=True):
    high = norm_stats["proprio"]["q99"]
    low = norm_stats["proprio"]["q01"]

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    data_normalized = np.clip(2*(data - low) / (high - low + 1e-8) - 1, -1.5, 1.5)

    # state为8x2=16维，pad为10x2=20维
    if pad_for_20:
        left_state = data_normalized[:, :8]
        left_padded = np.pad(left_state, ((0, 0), (0, 2)), mode='constant', constant_values=0)

        if is_bimanual:
            right_state = data_normalized[:, 8:]
            right_padded = np.pad(right_state, ((0, 0), (0, 2)), mode='constant', constant_values=0)
            data_normalized = np.concatenate([left_padded, right_padded], axis=1)
        else:
            data_normalized = np.pad(left_padded, ((0, 0), (0, 10)), mode='constant', constant_values=0)

    return data_normalized


def normalize_libero_delta_eef_state(data, norm_stats, pad_for_20=True, openvla_style=True):
    high = norm_stats["proprio"]["q99"]
    low = norm_stats["proprio"]["q01"]
    skip_range = (3, 9)

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    if openvla_style:
        data = np.insert(data, data.shape[1]-2, 0, axis=1)
        data_normalized = np.clip(2*(data - low) / (high - low + 1e-8) - 1, -1.5, 1.5)
        # 找到倒数第三维的下标
        mask_idx = data.shape[1] - 3
        # 覆盖掉那一列，用原始值替代
        data_normalized[:, mask_idx] = data[:, mask_idx]

    else: 
        # TODO: 这里对旋转那6维可能有nan，但有mask就先不管了
        data_normalized = np.clip(2*(data - low) / (high - low + 1e-8) - 1, -1.5, 1.5)

        mask = np.ones(data.shape[1], dtype=bool)
        mask[skip_range[0]: skip_range[1]] = False

        # mask的位置，取原始值
        data_normalized[:, ~mask] = data[:, ~mask]

    if pad_for_20:
        # pad到20
        data_normalized = np.concatenate(
            (data_normalized, np.zeros((data_normalized.shape[0], 20 - data_normalized.shape[1]))), axis=1)

    return data_normalized


def denormalize_pose_quat_action(data, norm_stats, is_bimanual: bool):

    high = norm_stats["action"]["q99"]
    low = norm_stats["action"]["q01"]

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    # 创建与归一化时相同的 mask
    if is_bimanual:
        skip_range = list(range(3,7)) + list(range(11,15))
    else:
        skip_range = list(range(3,7))
    d = data.shape[1]
    mask = np.ones(d, dtype=bool)
    mask[skip_range] = False

    # 原始归一化公式: y = 2 * (x - low) / (high - low) - 1
    # 反解 x: x = ((y + 1) / 2) * (high - low) + low
    denorm_range = high - low + 1e-8  # 使用与归一化时相同的范围和 epsilon
    
    # 首先对所有列应用反归一化公式
    # 注意：这里的 `data` 是已经归一化后的输入
    data_denormalized = (data + 1) / 2 * denorm_range + low
    
    # 使用 mask 将未被归一化的列恢复为其原始值
    data_denormalized[:, ~mask] = data[:, ~mask]
    return data_denormalized


def denormalize_joint_action(data, norm_stats, is_bimanual: bool):

    high = norm_stats["action"]["q99"]
    low = norm_stats["action"]["q01"]

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    # 原始归一化公式: y = 2 * (x - low) / (high - low) - 1
    # 反解 x: x = ((y + 1) / 2) * (high - low) + low
    denorm_range = high - low + 1e-8  # 使用与归一化时相同的范围和 epsilon
    
    # 首先对所有列应用反归一化公式
    # 注意：这里的 `data` 是已经归一化后的输入
    data_denormalized = (data + 1) / 2 * denorm_range + low
    
    return data_denormalized
    

def rot6d_to_rmat(rmat6d):
    n = rmat6d.shape[0]

    # 提取前两列
    col0 = rmat6d[:, [0, 2, 4]]  # shape [n,3]
    col1 = rmat6d[:, [1, 3, 5]]  # shape [n,3]

    # 归一化 col0
    col0_norm = np.linalg.norm(col0, axis=1, keepdims=True)
    col0 = col0 / col0_norm

    # Gram-Schmidt正交化 对 col1
    proj = np.sum(col1 * col0, axis=1, keepdims=True) * col0
    col1 = col1 - proj
    col1 = col1 / np.linalg.norm(col1, axis=1, keepdims=True)

    # 第三列 col2 = cross(col0, col1)
    col2 = np.cross(col0, col1, axis=1)

    # 拼成 [n,3,3]
    R_batch = np.stack([col0, col1, col2], axis=2)

    # r = Rotation.from_matrix(R_batch)
    # axis_angle = r.as_rotvec()
    # return axis_angle

    return R_batch

def invert_poses(arr: np.ndarray) -> np.ndarray:
    """
    arr: shape (..., 8)
      0:3 -> (x, y, z)
      3:7 -> quaternion (qx, qy, qz, qw)  [xyzw 顺序]
      7   -> 其他量（如 gripper), 原样保留

    返回与输入同形状的数组，前 7 项为逆位姿 (t_inv, q_inv), 第 8 列原样拷贝。
    """
    a = np.asarray(arr)
    assert a.shape[-1] >= 7, "最后一维至少包含 x,y,z,qx,qy,qz,qw"

    # 处理 1D 输入的情况
    squeezed = False
    if a.ndim == 1:
        a = a[None, ...]
        squeezed = True

    dtype = a.dtype
    t = a[..., 0:3].astype(np.float64)
    q = a[..., 3:7].astype(np.float64)  # [x,y,z,w]

    # 归一化四元数以防数值误差
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    qn = q / norm

    rot = Rotation.from_quat(qn)        # xyzw
    rot_inv = rot.inv()
    q_inv = rot_inv.as_quat()    # 仍是 xyzw

    # t_inv = -R^T * t = -rot_inv.apply(t)
    t_inv = -rot_inv.apply(t)

    out = np.empty_like(a, dtype=dtype)
    out[..., 0:3] = t_inv.astype(dtype)
    out[..., 3:7] = q_inv.astype(dtype)

    # 保留第 8 列及其后面的内容（如果有的话）
    if a.shape[-1] > 7:
        out[..., 7:] = a[..., 7:]

    if squeezed:
        out = out[0]
    return out


def standardize_quaternion(quat):
    # 计算每个四元数的模长，形状 [n, 1]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)

    # 归一化为单位四元数
    quat_normed = quat / (norms + 1e-8)

    # 保证 w >= 0
    # mask = quat_normed[:, 3] < 0
    # quat_normed[mask] = -quat_normed[mask]

    return quat_normed


def poses_absolute_to_relative(pos, quat, ref_pos, ref_quat) -> np.ndarray:
    
    N = pos.shape[0]
    
    R_ref = Rotation.from_quat(ref_quat).as_matrix()
    se3_ref = np.eye(4)
    se3_ref[:3, :3] = R_ref
    se3_ref[:3, 3] = ref_pos
    
    seq_rmat = Rotation.from_quat(quat).as_matrix()
    seq_se3 = np.eye(4)[None, ...].repeat(N, axis=0)
    seq_se3[:, :3, :3] = seq_rmat
    seq_se3[:, :3, 3] = pos
    
    se3_ref_inv = np.linalg.inv(se3_ref)
    relative_se3 = se3_ref_inv[None, ...] @ seq_se3
    
    return relative_se3


def poses_relative_to_absolute(rel_pos, rel_quat, cur_pos, cur_quat) -> np.ndarray:
    N = rel_pos.shape[0]
    
    R_cur = Rotation.from_quat(cur_quat).as_matrix()
    se3_cur = np.eye(4)
    se3_cur[:3, :3] = R_cur
    se3_cur[:3, 3] = cur_pos
    
    rel_rmat = Rotation.from_quat(rel_quat).as_matrix()
    rel_se3 = np.eye(4)[None, ...].repeat(N, axis=0)
    rel_se3[:, :3, :3] = rel_rmat
    rel_se3[:, :3, 3] = rel_pos
    
    absolute_se3 = se3_cur[None, ...] @ rel_se3
    abs_quat = Rotation.from_matrix(absolute_se3[:, :3, :3]).as_quat()
    abs_pos = absolute_se3[:, :3, 3]
    
    return abs_pos, abs_quat


def process_pose_quat_action(raw_actions, norm_stats, cur_state, is_bimanual, relative_pose):
    # raw_actions 为 [chunk_size, 20], 左右各取前8
    if raw_actions.ndim == 3:
        left_shrink = raw_actions[0, :, :8]
        right_shrink = raw_actions[0, :, 10:18]
    elif raw_actions.ndim == 2:
        left_shrink = raw_actions[:, :8]
        right_shrink = raw_actions[:, 10:18]
    else:
        raise ValueError(f"check raw_actions dim")
    
    if is_bimanual:
        shrink_action = np.concatenate([left_shrink, right_shrink], axis=1)
    else:
        shrink_action = left_shrink

    denormed_actions = denormalize_pose_quat_action(shrink_action, norm_stats, is_bimanual)
    
    cur_pos = cur_state[:3]
    cur_quat = cur_state[3:7]

    # xyzw 转 wxyz
    left_qxyzw = denormed_actions[..., 3:7]
    left_qxyzw = standardize_quaternion(left_qxyzw)
    
    if relative_pose:
        # 转为绝对位姿
        left_pos = denormed_actions[..., :3]
        left_abs_pos, left_qxyzw = poses_relative_to_absolute(left_pos, left_qxyzw, cur_pos, cur_quat)
        denormed_actions[:, :3] = left_abs_pos
        
    
    left_qwxyz = np.roll(left_qxyzw, 1, axis=-1)
    left_gripper = denormed_actions[..., 7:8]
    left_gripper = (left_gripper > 0.5).astype(int)
    
    denormed_actions[:, 3:7] = left_qwxyz 
    denormed_actions[:, 7:8] = left_gripper

    if is_bimanual:
        right_qxyzw = denormed_actions[..., 11:15]
        right_qxyzw = standardize_quaternion(right_qxyzw)
        
        if relative_pose:
            right_pos = denormed_actions[..., 8:11]
            right_abs_pos, right_qxyzw = poses_relative_to_absolute(right_pos, right_qxyzw, cur_pos, cur_quat)
            denormed_actions[:, 8:11] = right_abs_pos
        
        right_qwxyz = np.roll(right_qxyzw, 1, axis=-1)
        right_gripper = denormed_actions[..., 15:]
        right_gripper = (right_gripper > 0.5).astype(int)

        denormed_actions[:, 11:15] = right_qwxyz
        denormed_actions[:, 15:] = right_gripper

    return denormed_actions


def process_joint_action(raw_actions, norm_stats, cur_state, is_bimanual, relative_joint):
    # raw_actions 为 [chunk_size, 20], 左右各取前8
    if raw_actions.ndim == 3:
        left_shrink = raw_actions[0, :, :8]
        right_shrink = raw_actions[0, :, 10:18]
    elif raw_actions.ndim == 2:
        left_shrink = raw_actions[:, :8]
        right_shrink = raw_actions[:, 10:18]
    else:
        raise ValueError(f"check raw_actions dim")
    
    if is_bimanual:
        shrink_action = np.concatenate([left_shrink, right_shrink], axis=1)
    else:
        shrink_action = left_shrink

    denormed_actions = denormalize_joint_action(shrink_action, norm_stats, is_bimanual)
    
    left_joints = denormed_actions[..., :7]
    left_gripper = denormed_actions[..., 7:8]
    
    # 把相对关节角 加回成 绝对关节角
    if relative_joint:
        left_joints = left_joints + cur_state[:7]
        
    absolute_joint_action = np.concatenate([left_joints, left_gripper], axis=-1)
    
    # TODO: 双臂逻辑
    
    return absolute_joint_action


def denormalize_libero_delta_eef_action(data, norm_stats):

    high = norm_stats["action"]["q99"]
    low = norm_stats["action"]["q01"]

    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)

    # 原始归一化公式: y = 2 * (x - low) / (high - low) - 1
    # 反解 x: x = ((y + 1) / 2) * (high - low) + low
    denorm_range = high - low + 1e-8
    data_denormalized = (data + 1) / 2 * denorm_range + low
    
    return data_denormalized


def process_libero_delta_eef_action(actions, norm_stats):
    if actions.ndim == 3:
        actions = actions[0, :, :7]
    elif actions.ndim == 2:
        actions = actions[:, :7]
    else:
        raise ValueError(f"actions ndim is not right")
    denorm_actions = denormalize_libero_delta_eef_action(actions, norm_stats)

    # 夹爪
    denorm_actions[..., -1] = 1 - 2 * denorm_actions[..., -1]
    # Binarize to -1 or +1.
    denorm_actions[..., -1] = np.sign(denorm_actions[..., -1])

    return denorm_actions


def normalize_gripper_action(action, binarize=True):

    # 将最后一列二值化
    # 大于0.5 → 1，小于等于0.5 → -1
    action[..., -1] = 2.0 * (action[..., -1] > 0.5) - 1.0

    return action


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
            # 对于FAST的token-ids，在这里补齐
            if (task_cfg.train == "train" or task_cfg.train == "check_gt") and (task_cfg.use_fast_aux) and (role not in ["user", "system"]):
                modified_id = []
                for id in encode_id:
                    modified_id.append(id)
                    if id == FAST_START_TOKEN_INDEX:
                        # 把fast_id列表逐个插入
                        modified_id.extend(fast_ids)
                encode_id = modified_id

            # 如果是推理模式
            if task_cfg.train == "infer" and role not in ["user", "system"]:
                # 把encode_id中 ASSISTANT_TOKEN_INDEX + MAGIC_TOKEN_INDEX 后面的删去
                modified_id = []
                mark = False
                for id in encode_id:
                    modified_id.append(id)
                    if id == ASSISTANT_TOKEN_INDEX:
                        mark = True
                    if id == MAGIC_TOKEN_INDEX and mark == True:
                        break
                encode_id = modified_id
            # check gt 模式  去掉 eos+\n
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

        # 现在 根据 input_ids 制作 attention_mask
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
                            # 寻找对应的 vision_end
                            while pos2 < L and cur_id[pos2] != VISION_END_TOKEN_INDEX:
                                pos2 += 1
                            if pos2 < L and cur_id[pos2] == VISION_END_TOKEN_INDEX:
                                block_end = pos2
                                
                                # 这是一次 <image> 出现
                                if img_flag_cursor < len(real_image_flags) and (not bool(real_image_flags[img_flag_cursor])):
                                    # mask掉：整个区块 mask 置 0（含 vision_start / vision_end）
                                    for k in range(block_start, block_end + 1):
                                        attn_mask[k] = 0
                                img_flag_cursor += 1
                                # 跳过该区块
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


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas



def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"/home/agxi/sppro/robobrain2/rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def dummy_policy(index):
    # 一直给同一个值，测试delta概念
    base_action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.3, 1.0])

    action = base_action

    return action
