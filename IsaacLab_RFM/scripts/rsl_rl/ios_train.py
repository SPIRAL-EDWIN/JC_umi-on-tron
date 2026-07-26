# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--asset_usd_dir",
    type=str,
    default=None,
    help="Per-process directory for converted robot USD assets.",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Enable synchronous multi-GPU training (launch with torchrun).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.distributed:
    # torchrun exposes one process per GPU through LOCAL_RANK.  Set the device
    # before AppLauncher starts Isaac Sim so every process owns the right GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args_cli.device = f"cuda:{local_rank}"

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import torch
from datetime import datetime

# Add rsl_rl to Python path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_rsl_rl_dir = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "rsl_rl")
if _rsl_rl_dir not in sys.path:
    sys.path.insert(0, _rsl_rl_dir)

import ext_loco.tasks  # noqa: F401
from rsl_rl.runners import ImplicitOneStageRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from ext_loco.tasks.loco_manipulation.EE_pose.config.sf_tron1_arm.agents.implicit_one_stage_cfg import (
    ImplicitOneStageRunnerCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: ImplicitOneStageRunnerCfg):
    """Train contactNet."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.asset_usd_dir is not None:
        asset_usd_dir = os.path.abspath(args_cli.asset_usd_dir)
        if args_cli.distributed:
            asset_usd_dir = f"{asset_usd_dir}_rank{os.environ.get('RANK', '0')}"
        env_cfg.scene.robot.spawn.usd_dir = asset_usd_dir

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    distributed = args_cli.distributed
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    if distributed:
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("NCCL is required for --distributed training.")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # Different seeds produce different rollouts on each rank.  Rank 0's model
    # parameters are broadcast by the runner before the first rollout.
    env_cfg.seed = agent_cfg.seed + rank
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    # Keep logs in the repository by default so the script is portable across
    # machines. Set WBC_LOG_ROOT to use a different location.
    default_log_root = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "logs", "rsl_rl")
    log_root_path = os.path.abspath(
        os.environ.get("WBC_LOG_ROOT", default_log_root)
    )
    if rank == 0:
        print(f"[INFO] Synchronous training with {world_size} rank(s).")
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    if rank == 0:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
    else:
        log_dir = None
    if distributed:
        shared_log_dir = [log_dir]
        torch.distributed.broadcast_object_list(shared_log_dir, src=0)
        log_dir = shared_log_dir[0]

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = ImplicitOneStageRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir if rank == 0 else None,
        device=agent_cfg.device,
        distributed=distributed,
    )
    # write git state to logs
    if rank == 0:
        runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if rank == 0:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    if rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)

    # close the simulator
    env.close()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
