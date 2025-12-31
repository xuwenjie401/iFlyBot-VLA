"""

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "action_trunk": Action in 1s
        "language_instruction": str
    }
"""

from typing import Any, Dict
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_trans

from datasets.rlds.data_utils import(
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
    tf_pose_to_mat,
    gripper_2d_to_1d,
    sppro_fold_gripper_action_convert,
    galaxea_gripper_state_transfer,
    galaxea_gripper_action_transfer,
    velocity_act_to_wrist_frame,
)


# ==========================================================================================================
# Bridge
# ==========================================================================================================

def bridge_oxe_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # print(trajectory.keys(), trajectory['observation'].keys())
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    
    state = {}
    state["EEF_state"] = trajectory["observation"]["state"][:, :6]
    state["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(state[key], tf.float32)
            )
            for key in ["EEF_state", None, "gripper_state"]
        ],
        axis=1
    )

    return trajectory


def bridge_oxe_se3_trajectory_transform(
    trajectory: Dict[str, Any],
    rotation_form: str = 'rmat_6d',
) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]
    
    euler = trajectory['observation']['state'][:, 3:6]
    pos = trajectory['observation']['state'][:, 0:3]         # [n, 3]

    rmat = tfg_trans.rotation_matrix_3d.from_euler(euler)    # [n, 3, 3]
    trajectory['eef_se3'] = tf_pose_to_mat(pos, rmat)        # [n, 4, 4]

    trajectory["gripper_action"] = tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32)
    
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    return trajectory


def bridge_orig_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # print(trajectory.keys(), trajectory['observation'].keys())
    trajectory = relabel_bridge_actions(trajectory)
    
    state = {}
    state["EEF_state"] = trajectory["observation"]["state"][:, :6]
    state["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(state[key], tf.float32)
            )
            for key in ["EEF_state", None, "gripper_state"]
        ],
        axis=1
    )
    return trajectory


def bridge_orig_se3_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]
    
    euler = trajectory['observation']['state'][:, 3:6]
    pos = trajectory['observation']['state'][:, 0:3]         # [n, 3]
    # TODO: 旋转矩阵可以最后再转，初始保留四元数会比较方便
    rmat = tfg_trans.rotation_matrix_3d.from_euler(euler)    # [n, 3, 3]
    trajectory['eef_se3'] = tf_pose_to_mat(pos, rmat)        # [n, 4, 4]

    gripper_act = binarize_gripper_actions(trajectory["action"][:, -1])
    trajectory["gripper_action"] = tf.cast(gripper_act[:, None], tf.float32)

    # Note: 注意在OpenVLA原版的transform函数里，由于action定义成state_k+1 - state_k，所以在relabel里去掉了最后一条
    # 这里由于我不make action，只把状态弄出来，所以相当于seq_length会比默认方法+1
    
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    # print(f"[check-transform-traj]: {trajectory.keys()}\neef-se3: {trajectory['eef_se3']}")
    return trajectory
    

# ==========================================================================================================
# RT1 oxe
# ==========================================================================================================
def rt1_oxe_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(trajectory["observation"][key], tf.float32)
            )
            for key in ["base_pose_tool_reached", "gripper_closed"]
        ],
        axis=1
    )
    
    return trajectory


def rt1_oxe_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    trajectory["gripper_action"] = rel2abs_gripper_actions(gripper_action)[:, None]

    pos = trajectory['observation']['base_pose_tool_reached'][:, 0:3]
    quat = trajectory["observation"]["base_pose_tool_reached"][:, 3:7]
    rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quat)
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_closedness_commanded"]
    return trajectory


# ==========================================================================================================
# droid
# ==========================================================================================================
def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))


def droid_joint_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    state_joint = tf.cast(trajectory["observation"]["joint_position"], tf.float32)
    state_gripper = tf.cast(trajectory["observation"]["gripper_position"], tf.float32)
    action_joint = tf.cast(trajectory["action_dict"]["joint_position"], tf.float32)
    action_gripper = tf.cast(trajectory["action_dict"]["gripper_position"], tf.float32)

    trajectory["state"] = tf.concat(
        (
            state_joint,
            state_gripper,
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            action_joint,
            action_gripper,
        ),
        axis=-1,
    )

    return trajectory


def droid_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    euler = tf.cast(trajectory['observation']['cartesian_position'][:, 3:6], tf.float32)
    pos = tf.cast(trajectory['observation']['cartesian_position'][:, 0:3], tf.float32)

    rmat = tfg_trans.rotation_matrix_3d.from_euler(euler)    # [n, 3, 3]
    trajectory['eef_se3'] = tf_pose_to_mat(pos, rmat)        # [n, 4, 4]

    trajectory["gripper_action"] = tf.cast(1 - trajectory["action_dict"]["gripper_position"], tf.float32)
    # trajectory["gripper_action"] = 1 - trajectory["action_dict"]["gripper_position"]

    trajectory["observation"]["gripper_state"] = tf.cast(trajectory["observation"]["gripper_position"], tf.float32)

    return trajectory


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]

    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["state"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    
    # print(trajectory['observation'].keys())
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *wrist* frame of the robot.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"], trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


# ==========================================================================================================
# OXE others
# ==========================================================================================================
def bc_z_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)),
        ),
        axis=-1,
    )
    
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(trajectory["observation"][key], tf.float32)
            )
            for key in ["present/xyz", "present/axis_angle", None, "present/sensed_close"]
        ],
        axis=1
    )
    
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory

def bc_z_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    axis_angle = trajectory['observation']['present/axis_angle']
    angle = tf.expand_dims(tf.norm(axis_angle, axis=-1), axis=-1)
    axis = axis_angle / (angle + 1e-8)

    pos = trajectory['observation']['present/xyz'][:, 0:3]         # [n, 3]
    rmat = tfg_trans.rotation_matrix_3d.from_axis_angle(axis=axis, angle=angle)
    
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)

    trajectory["gripper_action"] = invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32))

    # TODO: gripper state
    trajectory["observation"]["gripper_state"] = invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32))
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # x,y,z,  qx, qy, qz, qw,  gripper
    # trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    pos = trajectory["observation"]["robot_state"][:, 6:9]
    quat = trajectory["observation"]["robot_state"][:, 9:13]

    # TODO: 夹爪用哪个？ 
    # trajectory["observation"]["gripper_state"] = trajectory["observation"]["robot_state"][:, 13]    # 二值，gripper_is_closed，意义相反？
    trajectory["observation"]["gripper_state"] = rel2abs_gripper_actions(trajectory["action"]["gripper_closedness_action"])[:, None]

    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)
    trajectory["gripper_action"] = gripper_action[:, None]

    rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quat)
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    return trajectory


def jaco_play_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    
    state = {}
    state["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    state["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(state[key], tf.float32)
            )
            for key in ["state_eef", None, "state_gripper"]
        ],
        axis=1
    )
    
    return trajectory


def jaco_play_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: 这里我发现 原版的transform代码state取前6维有问题，虽然说openvla也不用state吧
    # 来自https://github.com/clvrai/clvr_jaco_play_dataset的说明————
    # ee_cartesian_pos_ob : end effector cartesian position. ee_cartesian_pos_ob[0:3] corresponds to position and ee_cartesian_pos_ob[3:7] corresponds to orientation in quarternian format

    pos = trajectory["observation"]["end_effector_cartesian_pos"][:, :3]
    quat = trajectory["observation"]["end_effector_cartesian_pos"][:, 3:]
    rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quat)
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)

    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)
    trajectory["gripper_action"] = gripper_action[:, None]

    # TODO: gripper state

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def furniture_bench_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory

def furniture_bench_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    pos = trajectory["observation"]["state"][:, :3]
    quat = trajectory["observation"]["state"][:, 3:7]
    rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quat)
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)

    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))
    trajectory["gripper_action"] = gripper_action

    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1][:, None]
    return trajectory


def stanford_hydra_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    
    state = {}
    state["eef_state"] = tf.concat((trajectory["observation"]["state"][:, :3], trajectory["observation"]["state"][:, 7:10],),axis=-1)
    state["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(state[key], tf.float32)
            )
            for key in ["eef_state", None, "gripper_state"]
        ],
        axis=1,
    )
    
    return trajectory

def stanford_hydra_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    pos = trajectory["observation"]["state"][:, :3]
    quat = trajectory["observation"]["state"][:, 3:7]
    rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quat)
    trajectory["eef_se3"] = tf_pose_to_mat(pos, rmat)

    trajectory["gripper_action"] = invert_gripper_actions(trajectory["action"][:, -1:])

    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    return trajectory


def viola_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(trajectory["observation"][key], tf.float32)
            )
            for key in ["joint_states", "gripper_states"]
        ],
        axis=1
    )
    
    return trajectory

def viola_se3_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    mat44 = tf.reshape(trajectory["observation"]["ee_states"], (-1, 4, 4))
    trajectory["eef_se3"] = tf.transpose(mat44, perm=[0, 2, 1])

    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)
    trajectory["gripper_action"] = gripper_action

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_states"]

    return trajectory


# ==========================================================================================================
# LIBERO
# ==========================================================================================================
def libero_delta_ee_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )

    state = {}
    state["EEF_state"] = trajectory["observation"]["state"][:, :6]
    state["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["state"] = tf.concat(
        [
            (
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(state[key], tf.float32)
            )
            for key in ["EEF_state", None, "gripper_state"]
        ],
        axis=1
    )
    
    return trajectory

def libero_se3_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:

    # euler = trajectory['observation']['state'][:, 3:6]
    # rmat = tfg_trans.rotation_matrix_3d.from_euler(euler)    # [n, 3, 3]
    axis_angle = trajectory['observation']['state'][:, 3:6]
    angle = tf.expand_dims(tf.norm(axis_angle, axis=-1), axis=-1)
    axis = axis_angle / (angle + 1e-8)
    rmat = tfg_trans.rotation_matrix_3d.from_axis_angle(axis=axis, angle=angle)    # [n, 3, 3]

    pos = trajectory['observation']['state'][:, 0:3]         # [n, 3]
    trajectory['eef_se3'] = tf_pose_to_mat(pos, rmat)        # [n, 4, 4]

    trajectory["gripper_action"] = invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))
    
    trajectory["language_instruction"] = trajectory["language_instruction"]
    trajectory["observation"]["gripper_state"] = gripper_2d_to_1d(trajectory["observation"]["state"][:, -2:])

    # print(f"[check-transform-traj]: {trajectory.keys()}\neef-se3: {trajectory['eef_se3']}")
    return trajectory


def libero_pad_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:

    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))
    zero_vector = tf.zeros_like(gripper_action)

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            zero_vector,
            gripper_action,
        ],
        axis=1,
    )
    gripper_state = gripper_2d_to_1d(trajectory["observation"]["state"][:, -2:])  # 2D gripper state

    trajectory["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            gripper_state,
        ),
        axis=-1,
    )

    return trajectory


# ==========================================================================================================
# Organizations Open-Source
# ==========================================================================================================
def agibot_se3_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    left_quat = trajectory["state_quat"][:, 0, :]
    left_pos = trajectory["state_pos"][:, 0, :]
    right_quat = trajectory["state_quat"][:, 1, :]
    right_pos = trajectory["state_pos"][:, 1, :]

    left_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=left_quat)
    right_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=right_quat)

    trajectory["left_eef_se3"] = tf_pose_to_mat(left_pos, left_rmat)
    trajectory["right_eef_se3"] = tf_pose_to_mat(right_pos, right_rmat)

    trajectory["left_gripper_state"] = trajectory["state_gripper"][:, 0]
    trajectory["right_gripper_state"] = trajectory["state_gripper"][:, 1]
    trajectory["left_gripper_action"] = trajectory["action_gripper"][:, 0]
    trajectory["right_gripper_action"] = trajectory["action_gripper"][:, 1]

    # trajectory["left_gripper_state"] = trajectory["state_joint"][:, 6]
    # trajectory["right_gripper_state"] = trajectory["state_joint"][:, 13]
    # trajectory["left_gripper_action"] = trajectory["action_joint"][:, 6]
    # trajectory["right_gripper_action"] = trajectory["action_joint"][: 13]

    return trajectory


def agibot_joint_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["left_state"] = tf.concat(
        (
            trajectory["state_joint"][:, :7],
            trajectory["state_gripper"][:, 0:1],
        ),
        axis=-1,
    )
    trajectory["right_state"] = tf.concat(
        (
            trajectory["state_joint"][:, 7:14],
            trajectory["state_gripper"][:, 1:],
        ),
        axis=-1,
    )
    trajectory["left_action"] = tf.concat(
        (
            trajectory["action_joint"][:, :7],
            trajectory["action_gripper"][:, 0:1],
        ),
        axis=-1,
    )
    trajectory["right_action"] = tf.concat(
        (
            trajectory["action_joint"][:, 7:14],
            trajectory["action_gripper"][:, 1:],
        ),
        axis=-1,
    )

    return trajectory


def galaxea_joint_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    gripper_state_left = galaxea_gripper_state_transfer(trajectory["observation"]["gripper_state_left"])
    gripper_state_right = galaxea_gripper_state_transfer(trajectory["observation"]["gripper_state_right"])

    gripper_action_left = galaxea_gripper_action_transfer(trajectory["action"][:, 6])[:, None]
    gripper_action_right = galaxea_gripper_action_transfer(trajectory["action"][:, 13])[:, None]

    zeros_vector = tf.zeros_like(gripper_action_left)
    
    trajectory["left_state"] = tf.concat(
        (
            trajectory["observation"]["joint_position_arm_left"],
            zeros_vector,
            gripper_state_left,
        ),
        axis=-1,
    )
    trajectory["right_state"] = tf.concat(
        (
            trajectory["observation"]["joint_position_arm_right"],
            zeros_vector,
            gripper_state_right,
        ),
        axis=-1,
    )
    trajectory["left_action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            zeros_vector,
            gripper_action_left,
        ),
        axis=-1,
    )
    trajectory["right_action"] = tf.concat(
        (
            trajectory["action"][:, 7:13],
            zeros_vector,
            gripper_action_right,
        ),
        axis=-1,
    )
    return trajectory


# ==========================================================================================================
# iFlyTek self-made
# ==========================================================================================================
def sppro_fold_se3_trajectory_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    
    left_pos = trajectory["observation"]["state_pose"][:, 0:3]
    left_qw = trajectory["observation"]["state_pose"][:, 3:4]
    left_qxyz = trajectory["observation"]["state_pose"][:, 4:7]
    trajectory["left_gripper_state"] = trajectory["observation"]["state_pose"][:, 7]
    right_pos = trajectory["observation"]["state_pose"][:, 8:11]
    right_qw = trajectory["observation"]["state_pose"][:, 11:12]
    right_qxyz = trajectory["observation"]["state_pose"][:, 12:15]
    trajectory["right_gripper_state"] = trajectory["observation"]["state_pose"][:, 15]

    left_quat = tf.concat((left_qxyz, left_qw), axis=-1)
    right_quat = tf.concat((right_qxyz, right_qw), axis=-1)
    left_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=left_quat)
    right_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=right_quat)

    trajectory["left_eef_se3"] = tf_pose_to_mat(left_pos, left_rmat)
    trajectory["right_eef_se3"] = tf_pose_to_mat(right_pos, right_rmat)

    trajectory["left_gripper_action"] = sppro_fold_gripper_action_convert(trajectory["action_pose"][:, 7])
    trajectory["right_gripper_action"] = sppro_fold_gripper_action_convert(trajectory["action_pose"][:, 15])

    return trajectory


def sppro_pick_place_se3_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    
    left_pos = trajectory["observation"]["state_pose"][:, 0:3]
    left_qw = trajectory["observation"]["state_pose"][:, 3:4]
    left_qxyz = trajectory["observation"]["state_pose"][:, 4:7]
    trajectory["left_gripper_state"] = trajectory["observation"]["state_pose"][:, 7]
    right_pos = trajectory["observation"]["state_pose"][:, 8:11]
    right_qw = trajectory["observation"]["state_pose"][:, 11:12]
    right_qxyz = trajectory["observation"]["state_pose"][:, 12:15]
    trajectory["right_gripper_state"] = trajectory["observation"]["state_pose"][:, 15]

    left_quat = tf.concat((left_qxyz, left_qw), axis=-1)
    right_quat = tf.concat((right_qxyz, right_qw), axis=-1)
    left_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=left_quat)
    right_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=right_quat)

    trajectory["left_eef_se3"] = tf_pose_to_mat(left_pos, left_rmat)
    trajectory["right_eef_se3"] = tf_pose_to_mat(right_pos, right_rmat)

    # trajectory["left_gripper_state"] = trajectory["observation"]["state_joint"][:, 7]
    # trajectory["right_gripper_state"] = trajectory["observation"]["state_joint"][:, 15]
    trajectory["left_gripper_action"] = sppro_fold_gripper_action_convert(trajectory["action_pose"][:, 7])
    trajectory["right_gripper_action"] = sppro_fold_gripper_action_convert(trajectory["action_pose"][:, 15])
    
    return trajectory


def sppro_pick_place_singlearm_se3_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    
    left_pos = trajectory["observation"]["state_pose"][:, 0:3]
    left_qw = trajectory["observation"]["state_pose"][:, 3:4]
    left_qxyz = trajectory["observation"]["state_pose"][:, 4:7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state_pose"][:, 7][:, None]

    left_quat = tf.concat((left_qxyz, left_qw), axis=-1)
    left_rmat = tfg_trans.rotation_matrix_3d.from_quaternion(quaternion=left_quat)

    trajectory["eef_se3"] = tf_pose_to_mat(left_pos, left_rmat)
    trajectory["gripper_action"] = sppro_fold_gripper_action_convert(trajectory["action_pose"][:, 7][:, None])
    
    return trajectory


def sppro_pick_place_singlearm_joint_trajectory_transform(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    
    state = trajectory["observation"]["state_joint"][:, 0:8]

    trajectory["state"] = state

    # trajectory["action"] = trajectory["action_joint"][:, 0:8]
    # 关节角全用从动臂
    action_joint = trajectory["observation"]["state_joint"][:, 0:7]
    action_gripper = trajectory["action_joint"][:, 7:8]
    trajectory["action"] = tf.concat((action_joint, action_gripper), axis=-1)
    
    return trajectory


def sppro_folding_joint_trajectory_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    zeros_vector = tf.zeros_like(trajectory["observation"]["state_joint"][:, 6:7])
    trajectory["left_state"] = tf.concat(
        [
            trajectory["observation"]["state_joint"][:, :6],
            zeros_vector,
            trajectory["observation"]["state_joint"][:, 6:7],
        ],
        axis=-1,
    )
    trajectory["right_state"] = tf.concat(
        [
            trajectory["observation"]["state_joint"][:, 7:13],
            zeros_vector,
            trajectory["observation"]["state_joint"][:, 13:14],
        ],
        axis=-1,
    )
    trajectory["left_action"] = tf.concat(
        [
            trajectory["action_joint"][:, :6],
            zeros_vector,
            # trajectory["action_joint"][:, 6:7],
            sppro_fold_gripper_action_convert(trajectory["action_joint"][:, 6:7]),
        ],
        axis=-1,
    )
    trajectory["right_action"] = tf.concat(
        [
            trajectory["action_joint"][:, 7:13],
            zeros_vector,
            # trajectory["action_joint"][:, 13:14],
            sppro_fold_gripper_action_convert(trajectory["action_joint"][:, 13:14]),
        ],
        axis=-1,
    )

    return trajectory


# === Registry ===
STANDARDIZE_SE3_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_se3_trajectory_transform,
    "bridge_dataset": bridge_orig_se3_trajectory_transform,
    "fractal20220817_data": rt1_oxe_se3_trajectory_transform,   
    "droid": droid_se3_trajectory_transform,
    "bc_z": bc_z_se3_trajectory_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_se3_trajectory_transform,
    "viola": viola_se3_trajectory_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_se3_trajectory_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_se3_trajectory_transform,
    "jaco_play": jaco_play_se3_trajectory_transform,

    "libero_goal_no_noops": libero_se3_trajectory_transform,
    "libero_spatial_no_noops": libero_se3_trajectory_transform,
    "libero_object_no_noops": libero_se3_trajectory_transform,
    "libero_10_no_noops": libero_se3_trajectory_transform,
    "libero_union": libero_se3_trajectory_transform,

    "agibot": agibot_se3_trajectory_transform,
    "galaxea": None,
    "part1_r1_lite": None,
    "part2_r1_lite": None,
    "part3_r1_lite": None,
    "part4_r1_lite": None,
    "part5_r1_lite": None,

    "sppro_fold_v7": sppro_fold_se3_trajectory_transform,
    "sppro_pick_place_v4": sppro_pick_place_singlearm_se3_trajectory_transform,
}


STANDARDIZE_JOINT_TRANSFORMS = {
    "bridge_oxe": None,
    "bridge_dataset": None,
    "fractal20220817_data": None,
    "droid": droid_joint_trajectory_transform,
    "bc_z": None,
    "berkeley_autolab_ur5": None,
    "viola": None,
    "stanford_hydra_dataset_converted_externally_to_rlds": None,
    "furniture_bench_dataset_converted_externally_to_rlds": None,
    "jaco_play": None,

    # seems not right
    # "libero_goal_no_noops": libero_pad_trajectory_transform,
    # "libero_spatial_no_noops": libero_pad_trajectory_transform,
    # "libero_object_no_noops": libero_pad_trajectory_transform,
    # "libero_10_no_noops": libero_pad_trajectory_transform,
    # "libero_union": libero_pad_trajectory_transform,
    "libero_goal_no_noops": None,
    "libero_spatial_no_noops": None,
    "libero_object_no_noops": None,
    "libero_10_no_noops": None,
    "libero_union": None,

    "agibot": agibot_joint_trajectory_transform,
    "galaxea": galaxea_joint_trajectory_transform,
    "part1_r1_lite": galaxea_joint_trajectory_transform,
    "part2_r1_lite": galaxea_joint_trajectory_transform,
    "part3_r1_lite": galaxea_joint_trajectory_transform,
    "part4_r1_lite": galaxea_joint_trajectory_transform,
    "part5_r1_lite": galaxea_joint_trajectory_transform,

    "sppro_fold_v7": sppro_folding_joint_trajectory_transform,
    "sppro_pick_place_v4": sppro_pick_place_singlearm_joint_trajectory_transform,
}


STANDARDIZE_DELTA_EE_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_delta_ee_transform,
    "bridge_dataset": bridge_orig_delta_ee_transform,
    "fractal20220817_data": rt1_oxe_delta_ee_transform,
    "droid": droid_baseact_transform,
    "bc_z": bc_z_delta_ee_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_delta_ee_transform,
    "viola": viola_delta_ee_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_delta_ee_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_delta_ee_transform,
    "jaco_play": jaco_play_delta_ee_transform,

    "libero_goal_no_noops": libero_delta_ee_transform,
    "libero_spatial_no_noops": libero_delta_ee_transform,
    "libero_object_no_noops": libero_delta_ee_transform,
    "libero_10_no_noops": libero_delta_ee_transform,
    "libero_union": libero_delta_ee_transform,

    "agibot": None,
    "galaxea": None,
    "part1_r1_lite": None,
    "part2_r1_lite": None,
    "part3_r1_lite": None,
    "part4_r1_lite": None,
    "part5_r1_lite": None,

    "sppro_fold_v7": None,
    "sppro_pick_place_v4": None,
}

