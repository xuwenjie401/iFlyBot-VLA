from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Sequence, Any

from config.base_configs import OverallArguments
from config.robot_dataset_basic import MANIP_DATASETS_BASIC_SETTINGS, MANIP_DATASETS_USE


def make_datastyle_mask(
    cfg: OverallArguments,
    singlearm_action_dim: int = 10,
):
    if not cfg.task.use_action_mask:
        return None
    
    single_valid_dim = 7
    
    style = cfg.datasets.manip_data.action_style
    max_joint_dof = 0
    is_bimanual = False
    # 如果是joint, 选择使用数据集中 最高的valid_dim
    if style == "joint":
        dataset_list = MANIP_DATASETS_USE[cfg.datasets.manip_data.dataset_use]
        
        for ds_name, ratio in dataset_list:
            ds_setting = MANIP_DATASETS_BASIC_SETTINGS[ds_name]
            
            joint_dof = ds_setting["joint_dof"]
            if joint_dof > max_joint_dof:
                max_joint_dof = joint_dof
            
            if ds_setting["is_bimanual"]:
                is_bimanual = True
        
        single_valid_dim = max_joint_dof + 1
        
    elif style == "delta_eef":
        single_valid_dim = 7
    elif style == "eef_pose":
        rotation_representation = cfg.datasets.manip_data.rotation_representation
        if rotation_representation == "quat":
            single_valid_dim = 8
        elif rotation_representation == "euler":
            single_valid_dim = 7
        elif rotation_representation == "rmat6d":
            single_valid_dim = 10
        
    indices = list(range(single_valid_dim))
    if is_bimanual:
        arm2_offset = singlearm_action_dim
        indices += [i + arm2_offset for i in range(single_valid_dim)]

    return indices




