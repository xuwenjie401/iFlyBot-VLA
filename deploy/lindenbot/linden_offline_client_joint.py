import collections
import dataclasses
import math
from pathlib import Path
import sys
import json

import imageio
import numpy as np
import tqdm
import tyro
import argparse
import torch
import cv2
import einops
import time
import pandas as pd
import h5py
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
core_root = project_root / "src" / "iFlyBotVLA"
print(f"project-root: {project_root}")
print(f"Core-root: {core_root}")
sys.path.append(str(project_root))
sys.path.append(str(core_root))

from deploy.vla_websocket import websocket_client_policy as _websocket_client_policy

ISAAC_DUMMY_ACTION = [0.0] * 7 + [1.0]

is_bimanual = False

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 448
    replan_steps: int = 30
    
    num_trials_per_task: int = 1

output_file = "/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/sppro_vla_joint_check_1.json"

state_fields = ["l_s_0", "l_s_1", "l_s_2", "l_s_3", "l_s_4", "l_s_5", "l_s_6", "l_s_7",
                "r_s_0", "r_s_1", "r_s_2", "r_s_3", "r_s_4", "r_s_5", "r_s_6", "r_s_7"]
action_fields = ["l_s_0", "l_s_1", "l_s_2", "l_s_3", "l_s_4", "l_s_5", "l_s_6", "l_a_7",
                 "r_s_0", "r_s_1", "r_s_2", "r_s_3", "r_s_4", "r_s_5", "r_s_6", "r_a_7"]


axis_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'grip']
def paint(all_actions, all_labels, timesteps, eps_idx):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Episode {eps_idx} Left Arm Action Tracking", fontsize=20, y=0.98)
    axes = axes.flatten()
    
    for i, (ax, action_data, label_data) in enumerate(zip(axes, all_actions, all_labels)):
        # 用散点而不是连线绘制
        ax.scatter(timesteps, action_data, color='blue', s=8, label=f'Predicted {i}')
        ax.scatter(timesteps, label_data, color='red', s=8, alpha=0.7, label=f'Label {i}')
        
        ax.set_title(axis_names[i], fontsize=12)
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"outputs/left_arm_joint_eps{eps_idx}.png", dpi=200)
    plt.show()
    plt.close(fig)


def eval_video(args: Args) -> None:
    
    print("Connecting to policy server...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Connected to policy server.")
    
    data_dict = {}
    # episode_dir = "/wx-mix01/sppro/permanent/wjxu22/datasets/eval/sppro_pick_place_v4/episode_2025-10-03_09_22_28_970"    # 香蕉
    # task_description = "pick up the banana and put it on the plate"

    episode_dir = "/wx-mix01/sppro/permanent/wjxu22/datasets/eval/sppro_pick_place_v4/episode_2025-10-04_16_26_29_144"    # 可乐
    task_description = "pick up the cola and put it on the plate"

    episode_dir = Path(episode_dir)
    
    head_video_path = episode_dir / "cam_head" / "cam_head.mp4"
    left_video_path = episode_dir / "cam_left" / "cam_left.mp4"
    right_video_path = episode_dir / "cam_right" / "cam_right.mp4"
    csv_path = episode_dir / "robot_data.csv"
    
    df = pd.read_csv(
        csv_path,
        header=0,
        index_col=False,
        usecols=lambda col: not col.startswith("Unnamed")
    )
    # actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    # states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    
    act_dim = 16 if is_bimanual else 8
    actions = df[action_fields].values.astype(np.float32)
    actions = actions[:, :act_dim]
    csv_states = df[state_fields].values.astype(np.float32)

    states = np.zeros((csv_states.shape[0], act_dim))
    states[:, 0:7] = csv_states[:, 0:7]
    states[:, 7] = actions[:, 7]

    if is_bimanual:
        states[:, 8:15] = csv_states[:, 7:14]
        states[:, 15] = actions[:, 15]
    
    head_cap = cv2.VideoCapture(str(head_video_path))
    left_cap = cv2.VideoCapture(str(left_video_path))
    right_cap = cv2.VideoCapture(str(right_video_path))
    head_frames, left_frames, right_frames = [], [], []
    
    head_frame_count = 0
    while head_cap.isOpened():
        ret, frame = head_cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head_frames.append(rgb)
        head_frame_count += 1
    head_cap.release()
    
    left_frame_count = 0
    while left_cap.isOpened():
        ret, frame = left_cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_frames.append(rgb)
        left_frame_count += 1
    left_cap.release()
    
    right_frame_count = 0
    while right_cap.isOpened():
        ret, frame = right_cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        right_frames.append(rgb)
        right_frame_count += 1
    right_cap.release()
    
    if head_frame_count != left_frame_count or head_frame_count != right_frame_count:
        raise ValueError("Video frame counts do not match!")
    
    if head_frame_count != actions.shape[0] or head_frame_count != states.shape[0]:
        raise ValueError("Video frame count and csv data counts do not match!")
    
    data_dict['states'] = states
    data_dict['actions'] = states
    data_dict['image'] = head_frames
    data_dict['left_wrist_image'] = left_frames
    data_dict['right_wrist_image'] = right_frames
    
    record = []
    for eps_idx in range(args.num_trials_per_task):
        action_plan = collections.deque()
        
        t = 0
        err_list = []
        left_actions = [[] for _ in range(8)]
        right_actions = [[] for _ in range(8)]
        left_labels = [[] for _ in range(8)]
        right_labels = [[] for _ in range(8)]
        
        for i in tqdm.tqdm(range(data_dict["states"].shape[0])):
            try:
                img = data_dict["image"][i]
                left_wrist_img = data_dict["left_wrist_image"][i]
                right_wrist_img = data_dict["right_wrist_image"][i]
                
                img = einops.rearrange(img, "h w c -> c h w")
                left_wrist_img = einops.rearrange(left_wrist_img, "h w c -> c h w")
                right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
                
                if not action_plan:
                    element = {
                        "images": {
                            "cam_head": img,
                            "cam_left": left_wrist_img,
                            "cam_right": right_wrist_img,
                        },
                        "state": data_dict["states"][i],
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]

                    cur_rec = {}
                    cur_rec["index"] = i
                    cur_rec["state"] = data_dict["states"][i].tolist()  
                    cur_rec["action_chunk"] = action_chunk.tolist()
                    record.append(cur_rec)

                    assert(
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but the model only returns {len(action_chunk)} steps!"
                    action_plan.extend(action_chunk[:args.replan_steps])
                    
                action = action_plan.popleft().tolist()
                action_error = np.mean(np.abs(action[:8] - data_dict["actions"][i][:8]))
                err_list.append(action_error)
                for j in range(8):
                    left_actions[j].append(action[j])
                    left_labels[j].append(data_dict["actions"][i][j])

                    if is_bimanual:
                        right_actions[j].append(action[j+8])
                        right_labels[j].append(data_dict["actions"][i][j+8])   
            
            except Exception as e:
                print(f"Exception at step {i}: {e}")
                break
        
        with open(output_file, "w") as f:
            json.dump(record, f, indent=2)

        all_actions = [*left_actions]
        all_labels = [*left_labels]
        timesteps = np.arange(len(all_actions[0]))   

        paint(all_actions, all_labels, timesteps, eps_idx)
        

if __name__ == "__main__":
    tyro.cli(eval_video)
        
        