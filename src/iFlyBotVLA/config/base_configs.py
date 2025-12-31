from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Sequence, Union, Tuple, Any
from pathlib import Path

@dataclass
class CommonArguments:
    run_id: str = field(default="wjxu22_test_1218")
    is_debug: bool = field(default=False)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints


@dataclass
class ActionExpertArguments:
    action_dim: int = 20
    proprio_dim: int = 20
    hidden_size: int = 1024
    num_heads: int = 16
    dropout: float = 0.1
    gated: bool = False
    action_loss_type: str = "l1"   # 或 Literal["l1","l2",...]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/RoboBrain2.0-7B")
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_vision: bool = field(default=True)
    
    action_expert_cfg: ActionExpertArguments = field(default_factory=ActionExpertArguments)
    
    # Resume Run Parameters
    load_from_checkpoint: bool = False
    load_only_vlm: bool = False
    model_checkpoint: str= ""                                       # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = 0                                  # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = 0                                 # Epoch to Resume (should match checkpoint)
    special_training_stage: str=field(default="")                   # 目前仅对 pretrain_stage_2生效
    
    vlm_model_path: str = "/wx-mix01/sppro/permanent/wjxu22/model/robobrain2_3b"
    vlm_style: str = 'state'

    # Run Arguments
    save_interval: int = 10000                                      # Interval for saving checkpoints (in steps)
    image_aug: bool = True                                          # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    use_lora: bool = False
    freeze_vlm: bool = False
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0       
    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = ''                

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl",)                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "latent-action-pretrain"                   # Name of W&B project to log to (use default!)
    wandb_entity: str = "iFlyTek"                           # Name of entity to log under
    
    truncate_action_grad_to_vlm: bool = True       # action数据的梯度是否在传给vlm时截断 (预训练True, 微调False)
    skew_timesteps: bool = False                   # flow的训练，噪声采样是否更偏向0端(小t)
    
    enable_multi_noise_sample: bool = True         # flow-matching中，是否对每个样本采样多个噪声时间点
    multi_noise_sample_num: int = 5                # 每个样本采样的噪声时间点数量


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    dispatch_batches: Optional[bool] = False       # 重要 
    
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.03
    batch_size: int = 4

    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True

    max_steps = 100000


@dataclass
class VQADataArguments:
    vqa_dataset_use: str = field(default="")
    vqa_data_flatten: bool = field(default=False)
    vqa_data_packing: bool = field(default=False)
    vqa_base_interval: int = field(default=2)
    vqa_max_pixels: int = field(default=28 * 28 * 576)
    
    vqa_batch_size: int = field(default=2)


@dataclass
class ManipDataArguments:
    dataset_use: str = field(default="")
    data_packing: bool = field(default=False)
    
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    resize_resolution: int = field(default=448)
    action_style: str = field(default="eef_pose")                   # eef_pose  /  delta_eef  /  joint
    # absolute: state&action都是绝对值 ; relative: state&action都是相对cur_state的相对值 ; anchored_relative: state绝对，action相对
    control_reference: str = field(default="anchored_relative")    
    image_pad_other_view: bool = field(default=False)       # 是否pad其他视角的图像tokens
    mask_virtual_view: bool = field(default=True)          # 是否attention-mask构造视角的图像tokens

    norm_unique_mark: str = field(default="v0")                # 用于强行控制走新的归一化
    norm_mode: str = field(default="quantile")                 # 归一化模式 normal, quantile, bounds
    timestep_conditioned_norm: bool = field(default=False)     # 分时间步归一化，仅对 action-relative 生效
    
    # 以下参数仅限于universal模式，由于旋转可配置，所以单臂action都pad到最长情况10（3+6+1），双臂*2
    # --------------------------------------------------------------------------------
    bimanual: bool = field(default=True)                    # 是否按照双臂的数据格式
    single_to_bimanual_random_pad: bool = field(default=False)    # 单臂数据pad 成 双臂数据是否随机换边 （不随机默认左臂）
    rotation_representation: str = field(default="quat")    # 旋转的表达方式 quat(4), rmat6d(6) 目前只支持这俩
    # --------------------------------------------------------------------------------

    # all: 把所有state归0  /  random: 只保留current state，且一半随机置零  /  remain_current: 将horizon>1时的 非current状态归零
    state_dropout: str = field(default="no")

    # 控制内存增长速度
    # rlds_num_parallel_calls: int = field(default=0)         # 控制tf.data的map并行度, 0为tf.data.AUTOTUNE
    union_distributed: bool = field(default=True)          # 自采合并数据，是否根据rank分配文件夹

    action_duration: int = field(default=1)                 # action chunk 的持续时间（秒）
    action_chunk_size: int = field(default=30)              # 

    rlds_shuffle_buffer_size: int = field(default=5000)      # oxe数据的shuffle_buffer_size


@dataclass
class DatasetsArguments:
    manip_data: ManipDataArguments = field(default_factory=ManipDataArguments)
    vqa_data: VQADataArguments = field(default_factory=VQADataArguments)
    enable_vqa_cotrain: bool = field(default=False)              # 是否启用vqa数据的cotrain

    
@dataclass
class TaskArguments:
    train: str = field(default="train")                           # "train", "check_gt", "infer"
    
    use_lap_aux: bool = field(default=False)                       # 是否启用latent action 辅助任务
    infer_predict_latent: bool = field(default=True)
    num_latent_tokens: int = field(default=8)
    
    use_fast_aux: bool = field(default=False)                      # 是否启用FAST token 辅助任务
    # 推理action默认不推FAST
    infer_predict_fast: bool = field(default=False)
    
    domain_aware: bool = field(default=False)                            # 是否启用domain-aware的soft prompt等
    domain_freeze: bool = field(default=False)
    
    lam_source: str = field(default="lapa")               # "univla" , "lapa"
    univla_lam_path: str = field(default="/wx-mix01/sppro/permanent/wjxu22/model/UniVLA/lam-stage-2.ckpt")
    lapa_lam_path: str = field(default="/wx-mix01/sppro/permanent/wjxu22/model/lapa/dual.ckpt")
    
    use_action_mask: bool = field(default=False)  
    

@dataclass
class OverallArguments:
    common: CommonArguments = field(default_factory=CommonArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    datasets: DatasetsArguments = field(default_factory=DatasetsArguments)
    task: TaskArguments = field(default_factory=TaskArguments)

