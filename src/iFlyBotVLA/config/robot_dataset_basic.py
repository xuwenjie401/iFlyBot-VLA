from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Sequence, Any


EEF_POSE = "eef_pose"
DELTA_EE = "delta_eef"
JOINT = "joint"

# 
MANIP_DATASETS_BASIC_SETTINGS = {
    # OXE ----------------------------------------------------------------------------------------------------------------------
    "bridge_dataset": {
        "frequency": 5,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image_0", "wrist": None, "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/bridge_dataset",
        "union": False,
    },
    "droid": {
        "frequency": 15,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE, JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "exterior_image_1_left", "wrist": "wrist_image_left", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/droid",
        "union": False,
    },
    "fractal20220817_data": {
        "frequency": 3,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": None, "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE-part2/tensorflow_datasets/fractal20220817_data",
        "union": False,
    },
    # "jaco_play": {
    #     "frequency": 10,
    #     "is_bimanual": False,
    #     "support_format": [EEF_POSE, DELTA_EE],
    #     "image_obs_keys": {"primary": "image", "wrist": "image_wrist", "wrist_2": None},
    #     "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/jaco_play",
    #     "union": False,
    # },
    "berkeley_autolab_ur5": {
        "frequency": 5,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "hand_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE-part2/tensorflow_datasets/berkeley_autolab_ur5",
        "union": False,
    },
    "bc_z": {
        "frequency": 10,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": None, "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/bc_z",
        "union": False,
    },
    "viola": {
        "frequency": 20,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "agentview_rgb", "wrist": "eye_in_hand_rgb", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/viola",
        "union": False,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "frequency": 10,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE-part2/tensorflow_datasets/furniture_bench_dataset_converted_externally_to_rlds",
        "union": False,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "frequency": 10,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/stanford_hydra_dataset_converted_externally_to_rlds",
        "union": False,
    },
    "bridge_oxe": {
        "frequency": 5,
        "is_bimanual": False,
        "support_format": [EEF_POSE, DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": None, "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/OXE/2025.6.5-gs-2.8T/bridge_oxe",
        "union": False,
    },
    
    # LIBERO ----------------------------------------------------------------------------------------------------------------------
    "libero_goal_no_noops": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_libero_rlds/libero_goal_no_noops",
        "union": False,
    },
    "libero_object_no_noops": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_libero_rlds/libero_object_no_noops",    
        "union": False,
    },
    "libero_spatial_no_noops": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_libero_rlds/libero_spatial_no_noops",
        "union": False,
    },
    "libero_10_no_noops": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_libero_rlds/libero_10_no_noops",
        "union": False,
    },
    "libero_union": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [DELTA_EE],
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_libero_rlds",    
        "union": True,
    },
    
    # StartUps Open-Source ----------------------------------------------------------------------------------------------------------------------
    "agibot": {
        "frequency": 30,
        "is_bimanual": True,
        "support_format": [EEF_POSE, JOINT],
        "joint_dof": 7,
        "image_obs_keys": {"primary": "image_head", "wrist": "image_left_wrist", "wrist_2": "image_right_wrist"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/datasets/modified_agibot_rlds",
        "union": True,
    },
    "galaxea": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds",
        "union": True,
    },
    "part1_r1_lite": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds/part1_r1_lite",
        "union": False,
    },
    "part2_r1_lite": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds/part2_r1_lite",
        "union": False,
    },
    "part3_r1_lite": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds/part3_r1_lite",
        "union": False,
    },
    "part4_r1_lite": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds/part4_r1_lite",
        "union": False,
    },
    "part5_r1_lite": {
        "frequency": 15,
        "is_bimanual": True,
        "support_format": [JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_camera_head", "wrist": "image_camera_wrist_left", "wrist_2": "image_camera_wrist_right"},
        "path": "/wx-mix01/sppro/permanent/wjxu22/Galaxea-Open-World-Dataset/rlds/part5_r1_lite",
        "union": False,
    },
    
    # iFlyTek Self-Made ----------------------------------------------------------------------------------------------------------------------
    "sppro_fold_v7": {
        "frequency": 30,
        "is_bimanual": True,
        "support_format": [EEF_POSE, JOINT],
        "joint_dof": 6,
        "image_obs_keys": {"primary": "image_head", "wrist": "left_wrist_image", "wrist_2": "right_wrist_image"},
        # "image_obs_keys": {"primary": "image_head", "wrist": None, "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/sppro_fold_v7",
        "union": True,
    },
    "sppro_pick_place_v4": {
        "frequency": 30,
        "is_bimanual": False,
        "support_format": [EEF_POSE, JOINT],
        "joint_dof": 7,
        "image_obs_keys": {"primary": "image_head", "wrist": "left_wrist_image", "wrist_2": None},
        "path": "/wx-mix01/sppro/permanent/jfni3/sppro_pick_place_v4",
        "union": True,
    },
    
}


MANIP_DATASETS_USE = {
    "libero_all": [
        ("libero_goal_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_spatial_no_noops", 1.0),
        ("libero_10_no_noops", 1.0),
    ],

    "libero_union": [
        ("libero_union", 1.0)
    ],
    
    # 可用 universal 位姿 的数据集合
    "pretrain_eef_pose_all": [
        ("sppro_fold_v7", 0.5),
        ("sppro_pick_place_v4", 0.5),
        
        ("agibot", 1.0),
        ("droid", 0.45),
        
        ("bridge_dataset", 1.0),
        ("fractal20220817_data", 0.5),
        ("berkeley_autolab_ur5", 1.0),
        ("bc_z", 0.15),
        ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ],
    
    # 可用 关节角 的数据集合
    "pretrain_joint_all": [
        ("sppro_fold_v7", 0.5),
        ("sppro_pick_place_v4", 0.5),
        
        ("agibot", 1.0),
        ("droid", 0.45),
        ("galaxea", 0.5),
    ],
    
    "pretrain_mixed_all": [
        ("sppro_fold_v7", 0.5),
        ("sppro_pick_place_v4", 0.5),
        
        ("agibot", 1.0),
        ("droid", 0.45),
        ("galaxea", 0.5),
        
        ("bridge_dataset", 1.0),
        ("fractal20220817_data", 0.5),
        ("berkeley_autolab_ur5", 1.0),
        ("bc_z", 0.15),
        ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        # ("viola", 1.0),
    ],


    "pretrain_joint_1219": [
        ("sppro_fold_v7", 1.0),
        ("sppro_pick_place_v4", 0.5),
        
        ("agibot", 1.0),
        ("droid", 0.75),
        ("galaxea", 0.5),
    ],
    
    
    "test": [
        # ("libero_goal_no_noops", 1.0),
        # ("libero_object_no_noops", 1.0),
        # ("libero_spatial_no_noops", 1.0),
        # ("libero_10_no_noops", 1.0),
        # ("libero_union", 1.0),

        # ("sppro_fold_v7", 1.0),
        # ("sppro_pick_place_v4", 1.0),
        
        # ("agibot", 1.0),
        # ("droid", 1.0),
        ("galaxea", 1.0),
        # ("part1_r1_lite", 1.0),
        # ("part2_r1_lite", 1.0),
        # ("part3_r1_lite", 1.0),
        # ("part4_r1_lite", 1.0),
        # ("part5_r1_lite", 1.0),
        
        # ("bridge_dataset", 1.0),
        # ("fractal20220817_data", 1.0),
        # ("berkeley_autolab_ur5", 1.0),
        # ("bc_z", 1.0),
        # ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        # ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        # ("viola", 1.0),
    ],

    "normalize": [
        # ("libero_goal_no_noops", 1.0),
        # ("libero_object_no_noops", 1.0),
        # ("libero_spatial_no_noops", 1.0),
        # ("libero_10_no_noops", 1.0),
        ("libero_union", 1.0),

        # ("sppro_fold_v7", 1.0),
        # ("sppro_pick_place_v4", 1.0),
        
        # ("agibot", 1.0),
        # ("droid", 1.0),
        # ("galaxea", 1.0),
        # ("part1_r1_lite", 1.0),
        # ("part2_r1_lite", 1.0),
        # ("part3_r1_lite", 1.0),
        # ("part4_r1_lite", 1.0),
        # ("part5_r1_lite", 1.0),
        
        # ("bridge_dataset", 1.0),
        # ("fractal20220817_data", 1.0),
        # ("berkeley_autolab_ur5", 1.0),
        # ("bc_z", 1.0),
        # ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        # ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        # ("viola", 1.0),
    ],
    
}


# DOMAIN settings
VOID_DOMAIN_ID = 0
SPPRO_PICK_PLACE_DOMAIN_ID = 1
SPPRO_FOLD_DOMAIN_ID = 2
BRIDGE_DOMAIN_ID = 3
RT1_DOMAIN_ID = 4
AGIBOT_DOMAIN_ID = 5
GALAXEA_DOMAIN_ID = 6
DROID_DOMAIN_ID = 7
LIBERO_DOMAIN_ID = 8
BERKELEY_AUTOLAB_DOMAIN_ID = 9
JACO_PLAY_DOMAIN_ID = 10
STANDFORD_HYDRA_DOMAIN_ID = 11
FURNITURE_BENCH_DOMAIN_ID = 12
BC_Z_DOMAIN_ID = 13
UTOKYO_XARM_DOMAIN_ID = 14
VIOLA_DOMAIN_ID = 15
LANGUAGE_TABLE_DOMAIN_ID = 16
dataset_domain_ids = {
    'fractal20220817_data' : RT1_DOMAIN_ID,
    'bridge_oxe' : BRIDGE_DOMAIN_ID,
    'bridge_dataset' : BRIDGE_DOMAIN_ID,
    'berkeley_autolab_ur5' : BERKELEY_AUTOLAB_DOMAIN_ID,
    'jaco_play' : JACO_PLAY_DOMAIN_ID,
    'language_table' : LANGUAGE_TABLE_DOMAIN_ID,
    'stanford_hydra_dataset_converted_externally_to_rlds' : STANDFORD_HYDRA_DOMAIN_ID,
    'furniture_bench_dataset_converted_externally_to_rlds' : FURNITURE_BENCH_DOMAIN_ID,
    'bc_z' : BC_Z_DOMAIN_ID,
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds' : UTOKYO_XARM_DOMAIN_ID,
    'droid' : DROID_DOMAIN_ID,
    'viola' : VIOLA_DOMAIN_ID,

    "agibot" : AGIBOT_DOMAIN_ID,
    "galaxea" : GALAXEA_DOMAIN_ID,
    "part1_r1_lite": GALAXEA_DOMAIN_ID,
    "part2_r1_lite": GALAXEA_DOMAIN_ID,
    "part3_r1_lite": GALAXEA_DOMAIN_ID,
    "part4_r1_lite": GALAXEA_DOMAIN_ID,
    "part5_r1_lite": GALAXEA_DOMAIN_ID,
    "sppro_fold_v7": SPPRO_FOLD_DOMAIN_ID,
    "sppro_pick_place_v4": SPPRO_PICK_PLACE_DOMAIN_ID,
    "sppro_package": SPPRO_PICK_PLACE_DOMAIN_ID,

    "libero_goal_no_noops": LIBERO_DOMAIN_ID,
    "libero_spatial_no_noops": LIBERO_DOMAIN_ID,
    "libero_object_no_noops": LIBERO_DOMAIN_ID,
    "libero_10_no_noops": LIBERO_DOMAIN_ID,
}


DEFAULT_STATS_FILES_ROOT = "/wx-mix01/sppro/permanent/wjxu22/datasets/statistics"


