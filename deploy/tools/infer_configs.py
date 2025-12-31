from dataclasses import dataclass, field
from typing import Optional, List, Dict, Sequence, Union, Tuple, Any
from pathlib import Path

@dataclass
class InferArguments:
    # 
    checkpoint_path: str = field(default="")
    load_only_vlm: bool = field(default=False)

    # norm file
    norm_stats_file: str = field(default="")

    embodiment: str=field(default="")
    
    # server
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)

    # 是否忽视state，目前会变全0 
    ignore_states: bool = field(default=False)

    # 图像字段
    image_head: str = field(default="base_0_rgb")
    image_wrist_left: str = field(default="left_wrist_0_rgb")
    image_wrist_right: str = field(default="right_wrist_0_rgb")

    # 注意这里将决定 模型推理的action 如何使用，和基础配置参数里的有区别
    is_bimanual: bool = field(default=False)

    # 历史遗留参数
    state_horizon: Optional[int] = field(default=None)
    old_version_ckpt: bool = field(default=False)
    old_version_prompt: Optional[bool] = field(default=False)


