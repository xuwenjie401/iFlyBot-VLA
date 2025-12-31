import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generic, TypeVar

import numpy as np
import safetensors
import torch

from config.base_configs import ModelArguments


class BaseModel(torch.nn.Module, abc.ABC):
    """
    Abstract VLA-Framework
    """
    def __init__(self,) -> None:
        super().__init__()
        
    # @abc.abstractmethod
    # def from_pretrained(
    #     cls, 
    #     pretrained_checkpoint: str, 
    #     # train_config,
    # ) -> None:
    #     pass
    

@dataclasses.dataclass
class BaseModelConfig(abc.ABC):
    
    proprio_dim: int
    action_dim: int
    action_horizon: int
    max_token_len: int
    
    @abc.abstractmethod
    def create(self, ) -> "BaseModel":
        pass
    
    def load(self, model_args: ModelArguments):
        pass



    
    
    
    