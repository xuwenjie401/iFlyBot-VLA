import os
import sys
import argparse
from omegaconf import OmegaConf

import dataclasses
import enum
import logging
import socket

import tyro
from pathlib import Path

project_root = Path(__file__).parent.parent
core_root = project_root / "src" / "iFlyBotVLA"
print(f"project-root: {project_root}")
print(f"Core-root: {core_root}")
sys.path.append(str(project_root))
sys.path.append(str(core_root))

from deploy.vla_websocket import websocket_policy_server
from deploy.vla_websocket import iFlyBotVLA_policy as _policy

from src.iFlyBotVLA.config.base_configs import TaskArguments, ManipDataArguments, ModelArguments, OverallArguments
from deploy.tools.infer_configs import InferArguments

def main(basic_config: OverallArguments, infer_config: InferArguments) -> None:
    policy = _policy.create_trained_policy(basic_config, infer_config)
    # policy_metadata = policy.metadata

    # Record the policy's behavior.
    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=infer_config.host,
        port=infer_config.port,
        # metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="deploy/tools/settings/state_lap_fast_joint.json")
    parser.add_argument("--infer_json", type=str, default="deploy/tools/settings/state_lap_fast_joint_infer.json")
    args = parser.parse_args()
    
    base = OmegaConf.structured(OverallArguments)
    override_base = OmegaConf.load(args.config_json)
    merged_base = OmegaConf.merge(base, override_base)
    # 用dataclass   先转dict
    obj = OmegaConf.to_object(merged_base)
    basic_config = obj

    infer = OmegaConf.structured(InferArguments)
    override_infer = OmegaConf.load(args.infer_json)
    merged_infer = OmegaConf.merge(infer, override_infer)
    obj_infer = OmegaConf.to_object(merged_infer)
    infer_config = obj_infer
    
    # check configs
    print(f"cfg from loaded file: {args.config_json}:\nbasic_config:\n{basic_config}\ninfer_json:\n{infer_config}")
    
    main(basic_config, infer_config)
