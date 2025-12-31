import json
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math
from scipy.interpolate import interp1d

import dlimp as dl
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_trans
import tensorflow_graphics.math.interpolation.slerp as tfg_slerp

from config.robot_dataset_basic import DEFAULT_STATS_FILES_ROOT


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: Dict) -> Dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


def galaxea_gripper_action_transfer(gripper: tf.Tensor) -> tf.Tensor:
    gripper_binary = tf.where(gripper >= 50.0, 1.0, 0.0)
    return gripper_binary

def galaxea_gripper_state_transfer(gripper: tf.Tensor) -> tf.Tensor:
    gripper_sacled = gripper / 100.0
    return gripper_sacled


def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


def tf_pose_to_mat(
    pos,     # [n, 3]
    rmat,    # [n, 3, 3]
):
    pos_col = tf.expand_dims(pos, axis=-1)                   # [n, 3, 1]
    mat_3x4 = tf.concat([rmat, pos_col], axis=-1)

    # create [0, 0, 0, 1] as the last row
    last_row = tf.constant([0, 0, 0, 1], dtype=mat_3x4.dtype)
    last_row = tf.reshape(last_row, (1, 1, 4))  # [1, 1, 4]
    last_row = tf.tile(last_row, [tf.shape(mat_3x4)[0], 1, 1])  # [B, 1, 4]

    # concatenate the last row to the matrix, get [B, 4, 4]
    mat_4x4 = tf.concat([mat_3x4, last_row], axis=1)
    return mat_4x4


def gripper_2d_to_1d(gripper: tf.Tensor) -> tf.Tensor:
    gripper_1d = tf.abs(gripper[:, 1] - gripper[:, 0])
    gripper_1d = tf.expand_dims(gripper_1d, axis=-1)  # shape (n,1)
    return gripper_1d


def sppro_fold_gripper_action_convert(gripper: tf.Tensor) -> tf.Tensor:
    ans = tf.cast(gripper >= 0.03, tf.float32)
    return ans


def euler_to_rmat(euler):
    return tfg_trans.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    return tfg_trans.rotation_matrix_3d.inverse(rot_mat)


def rotmat_to_rot6d(mat):
    """
    Converts rotation matrix to R6 rotation representation (first two rows in rotation matrix).
    Args:
        mat: rotation matrix

    Returns: 6d vector (first two rows of rotation matrix)

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Translates velocity actions (translation + rotation) from base frame of the robot to wrist frame.
    Args:
        velocity: 6d velocity action (3 x translation, 3 x rotation)
        wrist_in_robot_frame: 6d pose of the end-effector in robot base frame

    Returns: 9d velocity action in robot wrist frame (3 x translation, 6 x rotation as R6)

    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_list: List[str], dataset_weights: List[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_name, weight in zip(dataset_list, dataset_weights):
        pad = 80 - len(dataset_name)
        print(f"# {dataset_name}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def interpolate_action_atomic(
    R, t, gripper, traj_len, act_num, chunk_size,
):
    quat = tfg_trans.quaternion.from_rotation_matrix(R)

    # 0.0, 1.0, 2.0, ..., act_num
    src_idx = tf.linspace(0.0, tf.cast(act_num-1, tf.float32), act_num)
    # 0.0, act_num/chunk_size, ..., act_num
    tgt_idx = tf.linspace(0.0, tf.cast(act_num-1, tf.float32), chunk_size)  # chunk_size+1

    # floor/ceil索引
    idx_floor = tf.floor(tgt_idx)
    idx_ceil = tf.minimum(idx_floor+1, tf.cast(act_num-1, tf.float32))
    # 每个相对左边ref的比例
    ratio=tgt_idx-idx_floor
    idx_floor = tf.cast(idx_floor, tf.int32)
    idx_ceil = tf.cast(idx_ceil, tf.int32)

    # 扩展成traj_len
    idx_floor_b = tf.tile(idx_floor[None, :], [traj_len, 1])     # [traj_len, chunk_size]
    idx_ceil_b = tf.tile(idx_ceil[None, :], [traj_len, 1])       # [traj_len, chunk_size]
    ratio_b = tf.tile(ratio[None, :], [traj_len, 1])    # [traj_len, chunk_size]

    # 取出两侧四元数， slerp
    quat_floor = tf.gather(quat, idx_floor_b, batch_dims=1)  # [traj_len, chunk_size, 4]
    quat_ceil = tf.gather(quat, idx_ceil_b, batch_dims=1)

    quat_floor_flat = tf.reshape(quat_floor, [-1, 4])
    quat_ceil_flat = tf.reshape(quat_ceil, [-1, 4])
    ratio_flat = tf.reshape(ratio_b, [-1])
    ratio_exp = tf.expand_dims(ratio_flat, axis=-1)

    # tensorflow_graphics的实现不安全，会出现nan
    quat_interp = tfg_slerp.interpolate(quat_floor_flat, quat_ceil_flat, ratio_exp)  # [traj_len * chunk_size, 4]
    nan_mask = tf.math.reduce_any(tf.math.is_nan(quat_interp), axis=-1)  # shape [n], True 表示该行 NaN
    quat_interp_safe = tf.where(
        tf.expand_dims(nan_mask, -1),  # broadcast 到 [n,1]
        quat_floor_flat,
        quat_interp
    )

    quat_interp_safe = tf.reshape(quat_interp_safe, [traj_len, -1, 4])
    R_interp = tfg_trans.quaternion.rotation_matrix_3d.from_quaternion(quat_interp_safe)

    # 取出平移和夹爪  线性插值
    pos_floor = tf.gather(t, idx_floor_b, batch_dims=1)
    pos_ceil = tf.gather(t, idx_ceil_b, batch_dims=1)
    ratio_b_exp = tf.expand_dims(ratio_b, axis=-1)
    pos_interp = pos_floor + ratio_b_exp * (pos_ceil - pos_floor)
    pos_interp_expanded = tf.expand_dims(pos_interp, axis=-1)   # (n, chunk_size, 3, 1)

    grip_floor = tf.gather(gripper, idx_floor_b, batch_dims=1)
    # grip_ceil = tf.gather(gripper, idx_ceil_b, batch_dims=1)
    # grip_interp = grip_floor + ratio_b_exp * (grip_ceil - grip_floor)
    # grip_debug = tf.concat((grip_floor, grip_ceil, grip_interp), axis=-1)

    grip_interp = grip_floor

    # 重组回SE3
    bottom_row = tf.constant([0., 0., 0., 1.], shape=(1, 1, 1, 4))
    bottom_row = tf.tile(bottom_row, [traj_len, chunk_size, 1, 1])  # (n, chunk_size, 1, 4)

    Rt_3x4 = tf.concat([R_interp, pos_interp_expanded], axis=-1)  # (n, chunk_size, 3, 4)
    se3_interp = tf.concat([Rt_3x4, bottom_row], axis=2)  # (n, chunk_size, 4, 4)

    return se3_interp, grip_interp


def make_se3_action_chunk(
    traj: Dict[str, Any],
    bounded_indices,
    chunk_size,
    is_bimanual,
):
    if is_bimanual:
        left_action = tf.gather(traj['left_eef_se3'], bounded_indices, axis=0)  # [traj_len, win_size]
        left_gripper_act = tf.gather(traj['left_gripper_action'], bounded_indices, axis=0)

        right_action = tf.gather(traj['right_eef_se3'], bounded_indices, axis=0)  # [traj_len, win_size]
        right_gripper_act = tf.gather(traj['right_gripper_action'], bounded_indices, axis=0)

        traj["action"] = tf.stack([left_action, right_action], axis=2)
        traj["gripper_act"] = tf.stack([left_gripper_act, right_gripper_act], axis=2)
    
    else:
        traj['action'] = tf.gather(traj['eef_se3'], bounded_indices, axis=0)  # [traj_len, win_size]
        traj['gripper_act'] = tf.gather(traj['gripper_action'], bounded_indices, axis=0)
        
    traj_len = tf.shape(traj['action'])[0]
    raw_act_num = tf.shape(traj['action'])[1]
        
    if is_bimanual:
        left_R = traj['action'][:, :, 0, :3, :3]
        right_R = traj['action'][:, :, 1, :3, :3]
        left_t = traj['action'][:, :, 0, :3, 3]
        right_t = traj["action"][:, :, 1, :3, 3]

        left_gripper = traj["gripper_act"][..., 0]
        right_gripper = traj["gripper_act"][..., -1]

        left_se3_interp, left_grip_interp = interpolate_action_atomic(left_R, left_t, left_gripper, traj_len, raw_act_num, chunk_size+1)
        right_se3_interp, right_grip_interp = interpolate_action_atomic(right_R, right_t, right_gripper, traj_len, raw_act_num, chunk_size+1)

        traj["action"] = tf.stack([left_se3_interp, right_se3_interp], axis=2)                # [traj_len, chunk, 2, 4, 4]
        traj["gripper_act"] = tf.stack([left_grip_interp, right_grip_interp], axis=2)         # [traj_len, chunk, 2]
    else:
        R = traj['action'][..., :3, :3]
        t = traj['action'][..., :3, 3]
        gripper = traj['gripper_act']

        se3_interp, grip_interp = interpolate_action_atomic(R, t, gripper, traj_len, raw_act_num, chunk_size+1)

        traj['action'] = se3_interp
        traj['gripper_act'] = grip_interp
    
    return traj
    
    
def make_se3_current_state(
    traj: Dict[str, Any],
    is_bimanual,
):

    if is_bimanual:
        left_state = traj["left_eef_se3"]
        right_state = traj["right_eef_se3"]
        left_gripper = traj['left_gripper_state']
        right_gripper = traj['right_gripper_state']

        traj["state"] = tf.stack([left_state, right_state], axis=1)
        traj["gripper_obs"] = tf.stack([left_gripper, right_gripper], axis=1)

        del traj['left_eef_se3']
        del traj['right_eef_se3']
        del traj['left_gripper_state']
        del traj['right_gripper_state']
        
    else:
        traj['state'] = traj["eef_se3"]
        traj['gripper_obs'] = traj['observation']['gripper_state']
            
        del traj['eef_se3']
        del traj['observation']['gripper_state']

    return traj
    
    
def make_relative_se3_state_action(traj, control_reference, is_bimanual):
    if control_reference == "absolute":
        return traj
    
    if is_bimanual:
        left_pose0 = traj["action"][:, 0, :, :]
        right_pose0 = traj["action"][:, 1, :, :]
        
        left_pose0_inv = tf.linalg.inv(left_pose0)     # [n, 4, 4]
        left_pose0_inv_exp = tf.expand_dims(left_pose0_inv, axis=1)            # [n, 1, 4, 4]
        left_pose0_inv_exp = tf.expand_dims(left_pose0_inv_exp, axis=1)            # [n, 1, 1, 4, 4]
        right_pose0_inv = tf.linalg.inv(right_pose0)     # [n, 4, 4]
        right_pose0_inv_exp = tf.expand_dims(right_pose0_inv, axis=1)            # [n, 1, 4, 4]
        right_pose0_inv_exp = tf.expand_dims(right_pose0_inv_exp, axis=1)            # [n, 1, 1, 4, 4]
        
        if control_reference == "relative":
            left_updated_state = tf.linalg.matmul(left_pose0_inv_exp, traj["state"][:, 0:1, :, :])
            right_updated_state = tf.linalg.matmul(right_pose0_inv_exp, traj["state"][:, 1:, :, :])
            traj["state"] = tf.concat([left_updated_state, right_updated_state], axis=2)
        
        left_updated_action = tf.linalg.matmul(left_pose0_inv_exp, traj["action"][:, 0:1, :, :])
        right_updated_action = tf.linalg.matmul(right_pose0_inv_exp, traj["action"][:, 1:, :, :])
        traj["action"] = tf.concat([left_updated_action, right_updated_action], axis=2)
    
    else:
        pose0 = traj['action'][:, 0, :, :]    # [n, 4, 4]
        pose0_inv = tf.linalg.inv(pose0)     # [n, 4, 4]
        pose0_inv_exp = tf.expand_dims(pose0_inv, axis=1)            # [n, 1, 4, 4]

        if control_reference == "relative":
            traj['state'] = tf.linalg.matmul(pose0_inv_exp, traj['state'])    # [n, chunk_size, 4, 4]

        traj['action'] = tf.linalg.matmul(pose0_inv_exp, traj['action'])    # [n, chunk_size, 4, 4]
        
    return traj


def flatten_relative_se3_state_action(traj, control_reference, rotation_representation, is_bimanual):
    if is_bimanual:
        # 平移  (traj_len, horizon, 3)
        left_pos_state = traj['state'][:, 0, :3, 3]
        right_pos_state = traj["state"][:, 1, :3, 3]
        left_pos_act = traj['action'][:, :, 0, :3, 3]
        right_pos_act = traj['action'][:, :, 1, :3, 3]
        left_gripper_state = traj["gripper_obs"][:, 0]
        right_gripper_state = traj["gripper_obs"][:, 1]
        left_gripper_action = traj["gripper_act"][:, :, 0:1]
        right_gripper_action = traj["gripper_act"][:, :, 1:2]

        left_rotation_state, right_rotation_state = None, None
        left_rotation_act, right_rotation_act = None, None
        if rotation_representation == "quat":
            left_rmat_state = traj['state'][:, 0, :3, :3]
            right_rmat_state = traj['state'][:, 1, :3, :3]
            left_rotation_state = tfg_trans.quaternion.from_rotation_matrix(left_rmat_state)
            right_rotation_state = tfg_trans.quaternion.from_rotation_matrix(right_rmat_state)

            left_rmat_act = traj['action'][:, :, 0, :3, :3]
            right_rmat_act = traj['action'][:, :, 1, :3, :3]
            left_rotation_act = tfg_trans.quaternion.from_rotation_matrix(left_rmat_act)
            right_rotation_act = tfg_trans.quaternion.from_rotation_matrix(right_rmat_act)
        elif rotation_representation == "rmat6d":
            # 旋转  (traj_len, horizon, 6)
            left_rmat_state_cols = traj['state'][:, 0, :3, :2]
            right_rmat_state_cols = traj['state'][:, 1, :3, :2]
            shape_2d = tf.shape(left_rmat_state_cols)[:2]
            left_rotation_state = tf.reshape(left_rmat_state_cols, tf.concat([shape_2d, [6]], axis=0))
            right_rotation_state = tf.reshape(right_rmat_state_cols, tf.concat([shape_2d, [6]], axis=0))

            left_rmat_act_cols = traj['action'][:, :, 0, :3, :2]
            right_rmat_act_cols = traj['action'][:, :, 1, :3, :2]
            shape_2d = tf.shape(left_rmat_act_cols)[:2]
            left_rotation_act =  tf.reshape(left_rmat_act_cols, tf.concat([shape_2d, [6]], axis=0))  
            right_rotation_act =  tf.reshape(right_rmat_act_cols, tf.concat([shape_2d, [6]], axis=0)) 

        # 拼接
        final_state = tf.concat([left_pos_state, left_rotation_state, left_gripper_state, right_pos_state, right_rotation_state, right_gripper_state], axis=-1)
        final_action = tf.concat([left_pos_act, left_rotation_act, left_gripper_action, right_pos_act, right_rotation_act, right_gripper_action], axis=-1)

        del traj["left_gripper_action"]
        del traj["right_gripper_action"]
        del traj["left_gripper_state"]
        del traj["right_gripper_state"]
        del traj['gripper_act']
        del traj['gripper_obs']

    else:
        # 平移  (traj_len, horizon, 3)
        pos_state = traj['state'][:, :3, 3]
        pos_act = traj['action'][:, :, :3, 3]

        rotation_state = None
        rotation_act = None
        if rotation_representation == "quat":
            rmat_state = traj['state'][:, :3, :3]
            rotation_state = tfg_trans.quaternion.from_rotation_matrix(rmat_state)

            rmat_act = traj['action'][:, :, :3, :3]
            rotation_act = tfg_trans.quaternion.from_rotation_matrix(rmat_act)
        elif rotation_representation == "rmat6d":
            # 旋转  (traj_len, horizon, 6)
            rmat_state_cols = traj['state'][:, :3, :2]
            shape_2d = tf.shape(rmat_state_cols)[:2]
            rotation_state = tf.reshape(rmat_state_cols, tf.concat([shape_2d, [6]], axis=0))

            rmat_act_cols = traj['action'][:, :, :3, :2]
            shape_2d = tf.shape(rmat_act_cols)[:2]
            rotation_act =  tf.reshape(rmat_act_cols, tf.concat([shape_2d, [6]], axis=0))   

        # 拼接
        final_state = tf.concat([pos_state, rotation_state, traj['gripper_obs']], axis=-1)
        final_action = tf.concat([pos_act, rotation_act, traj['gripper_act']], axis=-1)

        del traj["gripper_action"]
        del traj['gripper_act']
        del traj['gripper_obs']
        
    # action去掉0位
    traj['action'] = final_action[:, 1:, :]
    traj['state'] = final_state
        
    # TODO:
    if control_reference == "relative":
        traj['state'] = tf.zeros_like(traj['state'])
    
    return traj
    
    
def interpolate_joint_atomic(
    joint, gripper, traj_len, act_num, chunk_size, style,
):
    # 0.0, act_num/chunk_size, ..., act_num
    tgt_idx = tf.linspace(0.0, tf.cast(act_num-1, tf.float32), chunk_size)  # chunk_size+1

    # floor/ceil索引
    idx_floor = tf.maximum(tf.floor(tgt_idx), tf.cast(0, tf.float32))
    idx_ceil = tf.minimum(idx_floor+1, tf.cast(act_num-1, tf.float32))
    # 每个相对左边ref的比例
    ratio=tgt_idx-idx_floor
    idx_floor = tf.cast(idx_floor, tf.int32)
    idx_ceil = tf.cast(idx_ceil, tf.int32)

    # 扩展成traj_len
    idx_floor_b = tf.tile(idx_floor[None, :], [traj_len, 1])     # [traj_len, chunk_size]
    idx_ceil_b = tf.tile(idx_ceil[None, :], [traj_len, 1])       # [traj_len, chunk_size]
    ratio_b = tf.tile(ratio[None, :], [traj_len, 1])    # [traj_len, chunk_size]

    # 关节 线性插值
    joint_floor = tf.gather(joint, idx_floor_b, batch_dims=1)
    joint_ceil = tf.gather(joint, idx_ceil_b, batch_dims=1)
    ratio_b_exp = tf.expand_dims(ratio_b, axis=-1)
    joint_interp = joint_floor + ratio_b_exp * (joint_ceil - joint_floor)

    grip_floor = tf.gather(gripper, idx_floor_b, batch_dims=1)

    if style == "action":
        grip_interp = grip_floor
    else:
        grip_ceil = tf.gather(gripper, idx_ceil_b, batch_dims=1)
        grip_interp = grip_floor + ratio_b_exp * (grip_ceil - grip_floor)
        # grip_debug = tf.concat((grip_floor, grip_ceil, grip_interp), axis=-1)

    action = tf.concat([joint_interp, grip_interp], axis=-1)
    return action


def make_joint_action_state(traj, control_reference, bounded_indices, chunk_size, is_bimanual):
    if is_bimanual:
        left_action = tf.gather(traj['left_action'], bounded_indices, axis=0)    # [traj_len, win_size]
        right_action = tf.gather(traj['right_action'], bounded_indices, axis=0)    # [traj_len, win_size]
        left_state = traj['left_state']
        right_state = traj['right_state']
        
        traj_len = tf.shape(left_action)[0]
        if control_reference == "relative" or control_reference == "anchored_relative":
            relative_left_action_joint = left_action - left_state[:traj_len, None, :]
            left_gripper = left_action[:, :, -1:]
            left_action = tf.concat((relative_left_action_joint[..., :-1], left_gripper), axis=-1)

            relative_right_action_joint = right_action - right_state[:traj_len, None, :]
            right_gripper = right_action[:, :, -1:]
            right_action = tf.concat((relative_right_action_joint[..., :-1], right_gripper), axis=-1)
        
        traj["action"] = tf.concat([left_action, right_action], axis=-1)
        traj["state"] = tf.concat([left_state, right_state], axis=-1)
    else:
        traj['action'] = tf.gather(traj['action'], bounded_indices, axis=0)
        
    traj_len = tf.shape(traj['action'])[0]
    act_num = tf.shape(traj['action'])[1]
    
    if is_bimanual:
        half = tf.shape(traj['action'])[2] // 2
        left_joint = traj["action"][..., :half-1]
        left_gripper = traj["action"][..., half-1:half]
        right_joint = traj["action"][..., half:-1]
        right_gripper = traj["action"][..., -1:]
        
        left_action_interp = interpolate_joint_atomic(left_joint, left_gripper, traj_len, act_num, chunk_size+1, "action")
        right_action_interp = interpolate_joint_atomic(right_joint, right_gripper, traj_len, act_num, chunk_size+1, "action")
        
        # 去0位
        traj["action"] = tf.concat([left_action_interp, right_action_interp], axis=-1)[:, 1:, :]
    else:
        joint = traj["action"][..., :-1]
        gripper = traj["action"][..., -1:]
        traj["action"] = interpolate_joint_atomic(joint, gripper, traj_len, act_num, chunk_size+1, "action")[:, 1:, :]
        
        if control_reference == "relative" or control_reference == "anchored_relative":
            relative_action_joint = traj["action"] - traj["state"][:traj_len, None, :]
            action_gripper = traj["action"][..., -1:]
            traj["action"] = tf.concat((relative_action_joint[..., :-1], action_gripper), axis=-1)
        
    return traj


def make_delta_eef_action_chunk(traj, chunk_size):
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    
    start_indices = tf.range(traj_len)[:, None]
    action_chunk_indices = tf.broadcast_to(
        tf.range(0, chunk_size),
        [traj_len, chunk_size]
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, chunk_size]
    )
    
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), traj_len-1)
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    
    # absolute_action_mask = tf.zeros([traj_len, action_dim], dtype=tf.bool)
    # neutral_actions = tf.where(
    #     absolute_action_mask[:, None, :],
    #     traj["action"],
    #     tf.zeros_like(traj["action"]),
    # )
    
    # for delta-eef, neutral is zero; ignore gripper
    neutral_actions = tf.zeros_like(traj["action"])
    
    action_past_goal = action_chunk_indices > traj_len-1
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    
    return traj


allowed_sppro_folding_prefix = [
    # "folding_1018_part1"
]

def is_sppro_fold_dir_allowed(dir_name: str):
    if len(allowed_sppro_folding_prefix) == 0:
        return True

    for prefix in allowed_sppro_folding_prefix:
        if dir_name.startswith(prefix):
            return True
    
    return False

sppro_pick_place_prefix = {
    "sppro_pick_place_v4": [
        "general_pick_place_",
    ]
}

def check_sppro_pick_place_prefix(dirname, prefixes):
    for prefix in prefixes:
        if dirname.startswith(prefix):
            return True
    
    return False

def find_general_pick_place_all_levels(root_dir, name):
    """
    递归遍历 root_dir 下的所有子目录，
    查找以 sppro_pick_place_prefix 开头的文件夹。
    如果某个文件夹被匹配，则不再深入其子目录。
    """
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 用列表拷贝是为了安全地修改 dirnames
        for dirname in list(dirnames):
            prefixes = sppro_pick_place_prefix[name]
            if check_sppro_pick_place_prefix(dirname, prefixes):
                full_path = os.path.join(dirpath, dirname)
                info = {
                    "name": dirname[len(sppro_pick_place_prefix[name][0]):].lstrip("_"),
                    "path": full_path,
                }
                result.append(info)
                # 阻止 os.walk 继续进入这个子文件夹
                dirnames.remove(dirname)
    return result


def get_part_info_from_union(root_dir, name, distributed=True):
    import torch.distributed as dist
    
    all_parts_info = []
    hyper_name = ""
    if name == "agibot":
        hyper_name = "rlds_agibot_dataset"
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for dirname in dirnames:
                part_info = {}
                part_info["part_name"] = f"{name}_{dirname}"
                part_info["real_path"] = os.path.join(dirpath, dirname)
                part_info["hyper_name"] = hyper_name
                all_parts_info.append(part_info)
            break
    elif name == "galaxea":
        for i in range(1, 6):
            hyper_name = f"part{i}_r1_lite"
            part_info = {}
            part_info["part_name"] = f"{name}_{hyper_name}"
            part_info["real_path"] = root_dir
            part_info["hyper_name"] = hyper_name
            all_parts_info.append(part_info)
    elif name == "libero_union":
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            part_info = {}
            part_info["part_name"] = f"{name}_{entry}"
            part_info["real_path"] = root_dir
            part_info["hyper_name"] = entry  # 直接用子目录名

            all_parts_info.append(part_info)
    elif name == "sppro_fold_v7":
        hyper_name = "rlds_example_dataset"
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for dirname in dirnames:
                if is_sppro_fold_dir_allowed(dirname):
                    part_info = {}
                    part_info["part_name"] = f"{name}_{dirname}"
                    part_info["real_path"] = os.path.join(dirpath, dirname)
                    part_info["hyper_name"] = hyper_name
                    # 检查文件夹权限
                    try:
                        check_dir = os.path.join(part_info["real_path"], hyper_name, "1.0.0")
                        if not os.path.isdir(check_dir):
                            # print(f"no {hyper_name} dir in {part_info['real_path']}")
                            continue
                    except PermissionError:
                        print(f"no access permission in {part_info['real_path']}")
                        continue
                    all_parts_info.append(part_info)
                    # 阻止 os.walk 继续进入这个子文件夹
                    dirnames.remove(dirname)
    elif name == "sppro_pick_place_v4":
        hyper_name = "rlds_example_dataset"
        
        results = find_general_pick_place_all_levels(root_dir, name)
        if distributed:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            
            results.sort(key=lambda x : x["name"])
            shard_size = len(results) // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size if rank < world_size - 1 else len(results)
            results = results[start_idx:end_idx]

        for res in results:
            part_info = {}
            part_info["part_name"] = res["name"]
            part_info["real_path"] = res["path"]
            part_info["hyper_name"] = hyper_name
            # 检查文件夹权限
            try:
                check_dir = os.path.join(part_info["real_path"], hyper_name, "1.0.0")
                if not os.path.isdir(check_dir):
                    # print(f"no {hyper_name} dir in {part_info['real_path']}")
                    continue
            except PermissionError:
                print(f"no access permission in {part_info['real_path']}")
                continue
            all_parts_info.append(part_info)
    else:
        raise ValueError(f"Other union dataset not implemented")
    
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!! len(find dirs)  {len(all_parts_info)}")
    return all_parts_info



def build_union_rlds_dataset(
    name,
    root_dir,
    num_parallel_reads=8, 
):
    import tensorflow_datasets as tfds
    
    parts_info = get_part_info_from_union(root_dir, name=name)
    
    part_datasets = []
    for ds_id, ds_info in enumerate(parts_info):
        try:
            builder = tfds.builder(name=ds_info["hyper_name"], data_dir=ds_info["real_path"], version="1.0.0")
            part_dataset = dl.DLataset.from_rlds(
                builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads,
            )
            part_datasets.append(part_dataset)
        except Exception as e:
            print(f"sth wrong with {ds_info['real_path']}")
    
    union_dataset = part_datasets[0]
    print(f"dataset -- {name}, find {len(part_datasets)}, merging as one ......")
    for i in range(1, len(part_datasets)):
        union_dataset = union_dataset.concatenate(part_datasets[i])
    
    return union_dataset
            
            
def make_statistics_unified(
    dataset,
    dataset_name,
    style,
    control_reference,
    timestep_conditioned,
    mark,
):
    # 当前分时间步归一化，仅对相对位姿情况生效
    do_timestep_normalize = timestep_conditioned and style == "eef_pose" and (control_reference == "relative" or control_reference == "anchored_relative")
    ts_mark = ""
    if do_timestep_normalize:
        ts_mark = "timestep_"

    file_name = f"{dataset_name}_{style}_{control_reference}_{ts_mark}{mark}.json"
    dataset_dir = os.path.join(DEFAULT_STATS_FILES_ROOT, dataset_name)
    primary_path = os.path.join(dataset_dir, file_name)
    local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", dataset_name, file_name))
    
    # 已经存在则跳过
    if os.path.exists(primary_path) or os.path.exists(local_path):
        print(f"statistics file {file_name} already exists, skip")
        return None
    
    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": traj["state"],
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")
    
    # 正常模式
    if not do_timestep_normalize :
        actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
        for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
            # 由于universal模式，已经把action和state chunk化了，此时不分时间步归一化统计，则只取chunk-0位
            single_action = traj["action"][:, 0, :]
            single_proprio = traj["proprio"][:, :]
            actions.append(single_action)
            proprios.append(single_proprio)
            num_transitions += traj["action"].shape[0]
            num_trajectories += 1

        actions, proprios = np.concatenate(actions), np.concatenate(proprios)
        metadata = {
            "action": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "max": actions.max(0).tolist(),
                "min": actions.min(0).tolist(),
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            },
            "proprio": {
                "mean": proprios.mean(0).tolist(),
                "std": proprios.std(0).tolist(),
                "max": proprios.max(0).tolist(),
                "min": proprios.min(0).tolist(),
                "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
            },
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories,
        }
    
        try: 
            os.makedirs(DEFAULT_STATS_FILES_ROOT, exist_ok=True)
            os.makedirs(dataset_dir, exist_ok=True)
            
            with open(primary_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except PermissionError:
            print(f"Could not write dataset statistics to {primary_path}. Writing to {local_path} instead.")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

        return metadata
    
    # 分时间步归一化
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    record_shape = True
    act_shape = None
    
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        if record_shape:
            act_shape = (traj['action'].shape[1], traj['action'].shape[2])
            record_shape = False

        if traj != "not_ok":
            action = traj['action']      # [traj_len, chunk_size, d]
            proprio = traj["proprio"][:, :]
            proprios.append(proprio)

            assert action.ndim == 3, f"action has to be 3-dimensional. Got shape={action.shape}"
            assert proprio.ndim == 2, f"proprio has to be 2-dimensional. Got shape={proprio.shape}"

            action_flat = action.reshape(action.shape[0], -1)
            actions.append(action_flat)

            num_transitions += traj["action"].shape[0]
            num_trajectories += 1

    # [n, chunk_size*d]  ,  [n, d]
    actions, proprios = np.concatenate(actions), np.concatenate(proprios)

    action_mean = actions.mean(0)
    action_std = actions.std(0)
    action_max = actions.max(0)
    action_min = actions.min(0)
    action_q01 = np.quantile(actions, 0.01, axis=0)
    action_q99 = np.quantile(actions, 0.99, axis=0)

    # 恢复2dim  (chunk_size, d)
    action_mean = action_mean.reshape(act_shape)
    action_std = action_std.reshape(act_shape)
    action_max = action_max.reshape(act_shape)
    action_min = action_min.reshape(act_shape)
    action_q01 = action_q01.reshape(act_shape)
    action_q99 = action_q99.reshape(act_shape)

    metadata = {
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "max": action_max.tolist(),
            "min": action_mean.tolist(),
            "q01": action_q01.tolist(),
            "q99": action_q99.tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try: 
        os.makedirs(DEFAULT_STATS_FILES_ROOT, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        
        with open(primary_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except PermissionError:
        print(f"Could not write dataset statistics to {primary_path}. Writing to {local_path} instead.")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    return metadata


def rotation_related_norm_mask(mask, rotation_representation, control_reference, timestep_conditioned, is_bimanual):
    do_timestep_normalize = (control_reference == "relative" or control_reference == "anchored_relative") and timestep_conditioned

    rotation_dims = {"quat": 4, "rmat6d": 6}
    gripper_start_dims = {"quat": 7, "rmat6d": 9}
    num_rota_dim = rotation_dims[rotation_representation]
    gripper_dim_idx = gripper_start_dims[rotation_representation]

    if do_timestep_normalize:
        if is_bimanual:
            mask = tf.concat([mask[:, :3],tf.zeros([tf.shape(mask)[0], num_rota_dim], tf.bool), mask[:, gripper_dim_idx:gripper_dim_idx+1]], axis=1)
            mask = tf.concat([mask, mask], axis=1)
        else:
            mask = tf.concat([mask[:, :3],tf.zeros([tf.shape(mask)[0], num_rota_dim], tf.bool), mask[:, gripper_dim_idx:]], axis=1)
    else:
        if is_bimanual:
            mask = tf.concat([mask[:3],tf.zeros([num_rota_dim], tf.bool), mask[gripper_dim_idx:gripper_dim_idx+1]], axis=0)
            mask = tf.concat([mask, mask], axis=0)
        else:
            mask = tf.concat([mask[:3],tf.zeros([num_rota_dim], tf.bool), mask[gripper_dim_idx:]], axis=0)
        mask = mask[None, :]

    return mask


def normalize_unified(
    traj,
    metadata,
    style,
    rotation_representation,
    timestep_conditioned,
    control_reference,
    is_bimanual,
    norm_mode: str = "quantile",
):
    keys_to_normalize = {"action": "action", "proprio": "state"}

    if norm_mode == "normal":
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            if style == "eef_pose" and key == "action":
                mask = rotation_related_norm_mask(mask, rotation_representation, control_reference, timestep_conditioned, is_bimanual)
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj
    
    elif norm_mode in ["bounds", "quantile"]:
        for key, traj_key in keys_to_normalize.items():
            if norm_mode == "bounds":
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif norm_mode == "quantile":
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            if style == "eef_pose" and key == "action":
                mask = rotation_related_norm_mask(mask, rotation_representation, control_reference, timestep_conditioned, is_bimanual)
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1.5, 1.5),
                    x,
                ),
            )

            # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        return traj

    raise ValueError(f"Unknown Normalization Type {norm_mode}")


def get_stats_from_files(
    stats_files_root,
    dataset_names,
    style, 
    control_reference,
    timestep_conditioned,
    mark,
):
    datasets_sizes = []
    datasets_statistics = []

    # 当前分时间步归一化，仅对相对位姿情况生效
    do_timestep_normalize = timestep_conditioned and style == "eef_pose" and (control_reference == "relative" or control_reference == "anchored_relative")
    ts_mark = ""
    if do_timestep_normalize:
        ts_mark = "timestep_"
    
    for dataset_name in dataset_names:
        file_name = f"{dataset_name}_{style}_{control_reference}_{ts_mark}{mark}.json"
        dataset_dir = os.path.join(DEFAULT_STATS_FILES_ROOT, dataset_name)
        primary_path = os.path.join(dataset_dir, file_name)
        local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", dataset_name, file_name))
        
        if os.path.exists(primary_path):
            # with open(primary_path, "r", encoding="utf-8") as f:
            #     metadata = json.load(f)
            with tf.io.gfile.GFile(primary_path, 'r') as f:
                metadata = json.load(f)
            datasets_sizes.append(metadata["num_transitions"])
            datasets_statistics.append(metadata)
            print(f"{dataset_name} stats file found at {primary_path}")
        elif os.path.exists(local_path):
            # with open(local_path, "r", encoding="utf-8") as f:
            #     metadata = json.load(f)
            with tf.io.gfile.GFile(primary_path, 'r') as f:
                metadata = json.load(f)
            datasets_sizes.append(metadata["num_transitions"])
            datasets_statistics.append(metadata)
            print(f"{dataset_name} stats file found at {local_path}")
        else:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n{dataset_name} statistics file {file_name} not found\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
    
    return datasets_sizes, datasets_statistics


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation


def chunk_obs_head_only(traj, window_size):
    traj_len = tf.shape(traj["action"])[0]

    # Create indices for the first and last elements within the window size
    first_indices = tf.range(traj_len)[:, None]  # First index is the current timestep
    last_indices = tf.maximum(first_indices + (window_size - 1), 0)  # Last index is the end of the window

    floored_last_indices = tf.maximum(tf.minimum(last_indices, traj_len - 1), 0)
    future_pair = tf.gather(traj["observation"]["image_primary"], floored_last_indices)
    future_pair = tf.squeeze(future_pair, axis=1)

    new_obs = {
        "image_primary": traj["observation"]["image_primary"],
        "image_wrist": traj["observation"]["image_wrist"],
        "image_wrist_2": traj["observation"]["image_wrist_2"],
        "image_future": future_pair,
    }
    traj["observation"] = new_obs

    return traj
