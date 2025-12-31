import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, List, Sequence
import copy
import numpy as np
from PIL import Image
import imageio
import argparse

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import yaml
import tqdm
from collections import deque
from enum import Enum
import logging
import time
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

import sys
project_root = Path(__file__).parent.parent.parent
core_root = project_root / "src" / "iFlyBotVLA"
print(f"project-root: {project_root}")
print(f"Core-root: {core_root}")
sys.path.append(str(project_root))
sys.path.append(str(core_root))

sys.path.append("/wx-mix01/sppro/permanent/wjxu22/opt/LIBERO")
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.iFlyBotVLA.config.token_configs import *

from deploy.vla_websocket import websocket_client_policy as _websocket_client_policy
from deploy.tools.infer_utils import quat2axisangle

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nccl_version = torch.cuda.nccl.version()
print(f"NCCL version: {nccl_version}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    # LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    # TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

@dataclass
class EvalLiberoConfig:
    task_suite_name: str = TaskSuite.LIBERO_GOAL     # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    # env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    env_img_res: int = 448

    replan_steps: int = 8

    host: str = "0.0.0.0"
    port: int = 8000

    local_log_dir: str = "./logs" 

    # sppro data-format
    obs_horizon: int = 2


class LiberoEvaluator:
    def __init__(
        self, 
        cfg: EvalLiberoConfig,
    ):
        self.cfg = cfg

        # 历史遗留
        obs_max_horizon = cfg.obs_horizon + 1
        self.history_states = deque(maxlen=obs_max_horizon)
        self.cur_state = None

        print("Connecting to policy server...")
        self.client = _websocket_client_policy.WebsocketClientPolicy(cfg.host, cfg.port)
        print("Connected to policy server.")


    def load_initial_states(self, cfg: EvalLiberoConfig, task_suite, task_id: int):
        """Load initial states for the given task."""
        # Get default initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # If using custom initial states, load them from file
        if cfg.initial_states_path != "DEFAULT":
            with open(cfg.initial_states_path, "r") as f:
                all_initial_states = json.load(f)
            logger.info(f"Using initial states from {cfg.initial_states_path}")
            return initial_states, all_initial_states
        else:
            logger.info("Using default initial states")
            return initial_states, None
        

    def run_episode(
        self,
        env,
        task_description: str,
        initial_state=None,
    ):
        """Run a single episode in the environment."""
        # Reset environment
        env.reset()
        # 清空历史状态
        self.history_states.clear()

        # Set initial state if provided
        if initial_state is not None:
            obs = env.set_init_state(initial_state)
        else:
            obs = env.get_observation()

        action_queue = deque(maxlen=30)

        # Setup
        t = 0
        replay_images = []
        max_steps = TASK_MAX_STEPS[self.cfg.task_suite_name]

        # Run episode
        success = False
        try:
            while t < max_steps + self.cfg.num_steps_wait:
                # Do nothing for the first few timesteps to let objects stabilize
                if t < self.cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action())
                    t += 1
                    continue

                # 注意这里旋转了180度
                img = get_libero_image(obs)
                wrist_img = get_libero_wrist_image(obs)
                # wrist_img_resized = resize_image_np_version(wrist_img, self.cfg.env_img_res)
                cur_state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))

                replay_images.append(img)

                if len(action_queue) == 0:
                    inputs = {
                        "image": {
                            "cam_head": img,
                            "cam_left": wrist_img,
                            "cam_right": None,
                        },
                        "state": cur_state,
                        "prompt": str(task_description),
                    }

                    action_chunk = self.client.infer(inputs)["actions"]
                    # if t % 7 == 0:
                        # print(f"Check actions: {actions}")
                    # breakpoint()
                    actions = action_chunk[:self.cfg.replan_steps]
                    action_queue.extend(actions)   

                action = action_queue.popleft()

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

        except Exception as e:
            logger.info(f"Episode error: {e}")
            exit()

        return success, replay_images


    def run_task(
        self,
        task_suite,
        task_id: int,
        total_episodes=0,
        total_successes=0
    ):
        """Run evaluation for a single task."""
        # Get task
        task = task_suite.get_task(task_id)

        # Get initial states
        initial_states, all_initial_states = self.load_initial_states(self.cfg, task_suite, task_id)

        # Initialize environment and get task description
        env, task_description = get_libero_env(task, resolution=self.cfg.env_img_res)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(self.cfg.num_trials_per_task)):
            # if episode_idx > 2:
            #     break
            logger.info(f"\nTask: {task_description}")

            # Handle initial state
            if self.cfg.initial_states_path == "DEFAULT":
                # Use default initial state
                initial_state = initial_states[episode_idx]
            else:
                # Get keys for fetching initial episode state from JSON
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episode if expert demonstration failed to complete the task
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    logger.info(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!")
                    continue

                # Get initial state
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            logger.info(f"Starting episode {task_episodes + 1}...")

            # Run episode
            success, replay_images = self.run_episode(
                env,
                task_description,
                initial_state,
            )

            # Update counters
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save replay video
            save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description)

            # Log results
            logger.info(f"Success: {success}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

        logger.info(f"Current task success rate: {task_success_rate}")
        logger.info(f"Current total success rate: {total_success_rate}")

        return total_episodes, total_successes


    def eval_libero(self) -> None:

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        print(f"task_suite we're using is {self.cfg.task_suite_name}")
        task_suite = benchmark_dict[self.cfg.task_suite_name]()
        num_tasks = task_suite.n_tasks

        # Start Evaluation
        total_episodes, total_successes = 0, 0
        count = 0
        for task_id in tqdm.tqdm(range(num_tasks)):
            
            # if count < 1:
            #     count += 1
            #     continue
            total_episodes, total_successes = self.run_task(
                task_suite,
                task_id,
                total_episodes=total_episodes,
                total_successes=total_successes,
            )
            count += 1

        # And... we're done!
        logger.info("... and that's all, folks!")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="libero_goal")
    args = parser.parse_args()

    config = EvalLiberoConfig(task_suite_name=args.env)

    evaluator = LiberoEvaluator(config)
    evaluator.eval_libero()
