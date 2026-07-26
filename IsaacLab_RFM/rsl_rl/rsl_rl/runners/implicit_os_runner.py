#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import Optional
import rsl_rl
from rsl_rl.algorithms import PPO_IOS
from rsl_rl.env import VecEnv

# from legged_loco.utils.wrapper import TwoCriticWrapperEnv
from rsl_rl.modules import ActorCritic, EmpiricalNormalization, SimplifiedContactNetModel, ActorCritic, GRUWrapper, FlowActorCritic
from rsl_rl.utils import store_code_state
from copy import deepcopy
import numpy as np

class ImplicitOneStageRunner:
    """ImplicitOneStage runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", distributed=False):
        self.cfg = train_cfg
        self.ppo_alg_cfg = train_cfg["ppo_algorithm"]
        # remove class name from config
        self.ppo_alg_cfg.pop("class_name")
        self.policy_cfg = train_cfg["policy"]
        self.gru_cfg = train_cfg["gru"]
        self.contactNet_cfg = train_cfg["contactNet"]
        self.device = device
        self.env = env
        self.distributed = distributed
        self.rank = torch.distributed.get_rank() if distributed else 0
        self.world_size = torch.distributed.get_world_size() if distributed else 1

        # The vectorized env returns (obs, extras)
        #   obs: tensor for policy observations
        #   extras["observations"]: dict with critic / contactNet / next_obs, etc.
        obs, extras = self.env.get_observations()

        policy_obs = obs
        critic_obs = extras["observations"]["critic"]
        contactnet_obs_hist = extras["observations"]["contactNet"]
        next_obs = extras["observations"]["next_obs"]

        num_obs = policy_obs.shape[1]
        self.num_obs = num_obs
        self.num_critic_obs = critic_obs.shape[1]
        self.num_contactNet_obs = contactnet_obs_hist.shape[2]
        self.next_obs_dim = next_obs.shape[1]

        contactNetClass = eval(self.contactNet_cfg.pop("class_name"))
        self.contactNet: SimplifiedContactNetModel = contactNetClass(
            input_dim=self.num_contactNet_obs,
            next_obs_decoder_output_dim=self.next_obs_dim,
            **self.contactNet_cfg,
        )

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCriticGru
        actor_policy_cfg = dict(self.policy_cfg)
        for key in (
            "parameterization",
            "solver_step_size",
            "prior_noise_std",
            "perturb_action_std",
            "sample_t_strategy",
            "p_mean",
            "p_std",
            "zero_action_input",
            "condition_drop_ratio",
        ):
            if key in self.ppo_alg_cfg:
                actor_policy_cfg[key] = self.ppo_alg_cfg[key]
        actor_critic: ActorCritic = actor_critic_class(
            num_obs + self.gru_cfg["gru_latent_dim"] - self.ppo_alg_cfg["next_obs_latent_dim"],
            self.num_critic_obs,
            self.env.num_actions,
            num_envs=self.env.num_envs,
            device=self.device,
            **actor_policy_cfg,
        ).to(self.device)

        gru = GRUWrapper(num_envs=self.env.num_envs, device=self.device, **self.gru_cfg)

        self.actor_critic = actor_critic
        self.gru = gru

        ppo_ios_cfg = dict(self.ppo_alg_cfg)
        for key in ("p_mean", "p_std", "zero_action_input", "condition_drop_ratio"):
            ppo_ios_cfg.pop(key, None)
        self.ppo_alg = PPO_IOS(actor_critic, gru, self.contactNet, device=self.device, **ppo_ios_cfg)
        self.ppo_alg.configure_distributed(distributed)

        self.cn_obs_hist_len = self.cfg["cn_obs_hist_len"]
        assert self.cn_obs_hist_len == contactnet_obs_hist.shape[1]
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.grad_penalty_scheme = self.cfg["grad_penalty_scheme"]
        self.adaptive_entropy = self.cfg["adaptive_entropy"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.num_critic_obs], until=1.0e8).to(
                self.device
            )
            self.contactNet_obs_normalizer = EmpiricalNormalization(shape=[self.num_contactNet_obs], until=1.0e8).to(
                self.device
            )
            self.next_obs_normalizer = EmpiricalNormalization(shape=[self.next_obs_dim], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.contactNet_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.next_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.ppo_alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            (num_obs,),
            (actor_critic.num_actor_obs,),
            (self.num_critic_obs,),
            (self.env.num_actions,),
            (self.cn_obs_hist_len, self.num_contactNet_obs),
            (self.next_obs_dim,),
            (self.gru_cfg["gru_latent_dim"],),
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        if self.distributed:
            self._broadcast_training_state()

    def _broadcast_training_state(self):
        """Start every rank from rank 0's identical model state."""
        modules = (self.actor_critic, self.ppo_alg.gru, self.ppo_alg.tf_encoder)
        for module in modules:
            for tensor in module.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def _normalize(self, normalizer, value):
        """Update empirical statistics from the global multi-rank batch."""
        if not self.distributed or not self.empirical_normalization or not normalizer.training:
            return normalizer(value)
        if normalizer.until is not None and normalizer.count >= normalizer.until:
            return (value - normalizer._mean) / (normalizer._std + normalizer.eps)

        dims = (0, 1) if value.dim() == 3 else (0,)
        local_mean = value.mean(dim=dims, keepdim=True).view_as(normalizer._mean)
        local_second_moment = value.square().mean(dim=dims, keepdim=True).view_as(normalizer._mean)
        torch.distributed.all_reduce(local_mean)
        torch.distributed.all_reduce(local_second_moment)
        global_mean = local_mean / self.world_size
        global_var = local_second_moment / self.world_size - global_mean.square()

        count_x = value.shape[0] * self.world_size
        normalizer.count += count_x
        rate = count_x / normalizer.count
        delta_mean = global_mean - normalizer._mean
        normalizer._mean += rate * delta_mean
        normalizer._var += rate * (
            global_var.clamp_min(0.0)
            - normalizer._var
            + delta_mean * (global_mean - normalizer._mean)
        )
        normalizer._std.copy_(torch.sqrt(normalizer._var.clamp_min(0.0)))
        return (value - normalizer._mean) / (normalizer._std + normalizer.eps)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.ppo_alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.ppo_alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.train_mode()  # switch to train mode (for dropout for example)

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", None)
        cn_obs_history = extras["observations"].get("contactNet", None)
        obs, critic_obs, cn_obs_history = (
            obs.to(self.device),
            critic_obs.to(self.device),
            cn_obs_history.to(self.device),
        )
        # perform normalization
        obs = self._normalize(self.obs_normalizer, obs)
        critic_obs = self._normalize(self.critic_obs_normalizer, critic_obs)
        cn_obs_history = self._normalize(self.contactNet_obs_normalizer, cn_obs_history)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self.tot_iter = tot_iter
        lr_udpate_count = 0
        for it in range(start_iter, tot_iter):
            self.it = it
            start = time.time()
            # gradient_penalty_scheme
            self._gradient_penalty_scheme(it)
            # adaptive_entropy
            self._adaptive_entropy_scheme(it)
            # Rollout
            dones = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    update_lr = i % 3 == 0
                    if update_lr:
                        lr_udpate_count += 1
                        self.ppo_alg.tf_gru_lr = self.get_learning_rate(
                            lr_udpate_count, self.contactNet_cfg["model_dim"]
                        )
                    actions = self.ppo_alg.act(obs, critic_obs, cn_obs_history)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # perform normalization
                    obs = self._normalize(self.obs_normalizer, obs)
                    critic_obs = self._normalize(
                        self.critic_obs_normalizer, infos["observations"]["critic"]
                    )
                    cn_obs_history = self._normalize(
                        self.contactNet_obs_normalizer, infos["observations"]["contactNet"]
                    )
                    next_obs = self._normalize(
                        self.next_obs_normalizer, infos["observations"]["next_obs"]
                    )
                    # move to the right device
                    obs, critic_obs, cn_obs_history, rewards, dones, next_obs = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        cn_obs_history.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_obs.to(self.device),
                    )
                    self.ppo_alg.process_env_step(rewards, dones, infos, next_obs)
                    if self.log_dir is not None:
                        # Book keeping
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        # The environment retains its last log dictionary on non-reset steps.  Only consume it
                        # when at least one episode actually ended, otherwise stale values are counted repeatedly.
                        if new_ids.numel() > 0:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                # Learning step
                start = stop
                self.ppo_alg.compute_returns(critic_obs)

            ppo_loss_dict = self.ppo_alg.update(it)
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if self.rank == 0 and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                # git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                # if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                #     for path in git_file_paths:
                #         self.writer.save_file(path)
                pass

        if self.rank == 0:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _gradient_penalty_scheme(self, it):
        if self.grad_penalty_scheme["is_used"]:
            if it >= self.grad_penalty_scheme["start_point"]:
                self.ppo_alg.grad_coef = self.grad_penalty_scheme["start_value"] + (
                    it - self.grad_penalty_scheme["start_point"]
                ) / (self.grad_penalty_scheme["end_point"] - self.grad_penalty_scheme["start_point"]) * (
                    self.grad_penalty_scheme["end_value"] - self.grad_penalty_scheme["start_value"]
                )
                self.ppo_alg.grad_coef = min(self.ppo_alg.grad_coef, self.grad_penalty_scheme["end_value"])
            else:
                self.ppo_alg.grad_coef = 0.0

    def _adaptive_entropy_scheme(self, it):
        if self.adaptive_entropy["is_used"]:
            if it >= self.adaptive_entropy["start_point"]:
                self.ppo_alg.entropy_coef = self.adaptive_entropy["start_value"] + (
                    it - self.adaptive_entropy["start_point"]
                ) / (self.adaptive_entropy["end_point"] - self.adaptive_entropy["start_point"]) * (
                    self.adaptive_entropy["end_value"] - self.adaptive_entropy["start_value"]
                )
                self.ppo_alg.entropy_coef = max(self.ppo_alg.entropy_coef, self.adaptive_entropy["end_value"])
            else:
                self.ppo_alg.entropy_coef = 0.0

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            weighted_sum_suffix = ".__weighted_sum"
            weighted_count_suffix = ".__weighted_count"

            # Preserve first-seen order while allowing metrics that were absent from the first reset event.
            info_keys = list(dict.fromkeys(key for ep_info in locs["ep_infos"] for key in ep_info))
            info_key_set = set(info_keys)
            weighted_metric_names = []
            private_weighted_keys = set()
            for key in info_keys:
                if not key.endswith(weighted_sum_suffix):
                    continue
                metric_name = key[: -len(weighted_sum_suffix)]
                count_key = metric_name + weighted_count_suffix
                if count_key in info_key_set:
                    weighted_metric_names.append(metric_name)
                    private_weighted_keys.update((key, count_key))

            def collect_info_values(key):
                values = []
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    value = torch.as_tensor(ep_info[key], device=self.device, dtype=torch.float32).reshape(-1)
                    values.append(value)
                if not values:
                    return torch.empty(0, device=self.device)
                return torch.cat(values)

            def log_episode_value(key, value):
                nonlocal ep_string
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            for key in info_keys:
                if key in private_weighted_keys:
                    continue
                infotensor = collect_info_values(key)
                if infotensor.numel() > 0:
                    log_episode_value(key, torch.mean(infotensor))

            # Weighted precision metrics retain their sample/command counts across asynchronous reset batches.
            for metric_name in weighted_metric_names:
                numerator = collect_info_values(metric_name + weighted_sum_suffix).sum()
                denominator = collect_info_values(metric_name + weighted_count_suffix).sum()
                if (
                    denominator.item() <= 0
                    or not torch.isfinite(numerator).item()
                    or not torch.isfinite(denominator).item()
                ):
                    continue
                log_episode_value(metric_name, numerator / denominator)
        mean_std = self.actor_critic.std.mean()

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("PPO_Loss/learning_rate", self.ppo_alg.learning_rate, locs["it"])
        self.writer.add_scalar("PPO_Loss/tf_gru_lr", self.ppo_alg.tf_gru_lr, locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_next_obs_est_loss", locs["ppo_loss_dict"]["next_obs"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_vel_est_loss", locs["ppo_loss_dict"]["lin_vel"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_value_loss", locs["ppo_loss_dict"]["value"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_surrogate_loss", locs["ppo_loss_dict"]["surrogate"], locs["it"])
        self.writer.add_scalar("PPO_Loss/beta_vae_loss", locs["ppo_loss_dict"]["beta_vae"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_kl", locs["ppo_loss_dict"]["kl"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_entropy", locs["ppo_loss_dict"]["entropy"], locs["it"])
        self.writer.add_scalar("PPO_Loss/mean_grad_penalty_loss", locs["ppo_loss_dict"]["grad_penalty"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            # if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
            #     self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            #     self.writer.add_scalar(
            #         "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
            #     )
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        str = f"""{'#' * width}\n""" f"""{str.center(width, ' ')}\n\n"""

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                # f"""{'#' * width}\n"""
                # f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value loss:':>{pad}} {locs["ppo_loss_dict"]['value']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["ppo_loss_dict"]['surrogate']:.4f}\n"""
                f"""{'action std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'kl:':>{pad}} {locs["ppo_loss_dict"]['kl']:.4f}\n"""
                f"""{'Mean grad penalty loss:':>{pad}} {locs["ppo_loss_dict"]['grad_penalty']:.4f}\n"""
                f"""{'reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'#' * width}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value loss:':>{pad}} {locs["ppo_loss_dict"]['value']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["ppo_loss_dict"]['surrogate']:.4f}\n"""
                f"""{'action std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'kl:':>{pad}} {locs["ppo_loss_dict"]['kl']:.4f}\n"""
                f"""{'#' * width}\n"""
                # f"""{'Mean grad penalty loss:':>{pad}} {locs['mean_grad_penalty_loss']:.4f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        # log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'mean vel est loss:':>{pad}} {locs["ppo_loss_dict"]['lin_vel']:.4f}\n"""
            f"""{'mean next obs est loss:':>{pad}} {locs["ppo_loss_dict"]['next_obs']:.4f}\n"""
            f"""{'beta_vae loss:':>{pad}} {locs["ppo_loss_dict"]['beta_vae']:.4f}\n"""
            f"""{'#' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(str + ep_string + log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "contactNet_state_dict": self.ppo_alg.tf_encoder.state_dict(),
            "gru_state_dict": self.ppo_alg.gru.state_dict(),
            "tf_gru_optimizer_state_dict": self.ppo_alg.tf_gru_optimizer.state_dict(),
            "ppo_optimizer_state_dict": self.ppo_alg.ppo_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
            saved_dict["contactNet_obs_norm_state_dict"] = self.contactNet_obs_normalizer.state_dict()
            saved_dict["next_obs_norm_state_dict"] = self.next_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):  # TODO: not complete
        loaded_dict = torch.load(path)
        self.ppo_alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        # allow loading checkpoints saved before GRUWrapper.hidden_state was registered as a buffer
        # filter out hidden_state from gru_state_dict since it depends on num_envs which may differ between training and play
        gru_state_dict = loaded_dict["gru_state_dict"].copy()
        if "hidden_state" in gru_state_dict:
            # Skip hidden_state as it depends on num_envs and will be reset during inference
            del gru_state_dict["hidden_state"]
        self.ppo_alg.gru.load_state_dict(gru_state_dict, strict=False)
        self.ppo_alg.tf_encoder.load_state_dict(loaded_dict["contactNet_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
            self.contactNet_obs_normalizer.load_state_dict(loaded_dict["contactNet_obs_norm_state_dict"])
            self.next_obs_normalizer.load_state_dict(loaded_dict["next_obs_norm_state_dict"])
            self.obs_normalizer.training = False
            self.critic_obs_normalizer.training = False
            self.contactNet_obs_normalizer.training = False
            self.next_obs_normalizer.training = False
        if load_optimizer:
            self.ppo_alg.tf_gru_optimizer.load_state_dict(loaded_dict["tf_gru_optimizer_state_dict"])
            self.ppo_alg.ppo_optimizer.load_state_dict(loaded_dict["ppo_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):  # TODO: not complete
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.ppo_alg.tf_encoder.to(device)
            self.ppo_alg.actor_critic.to(device)
            self.ppo_alg.gru.to(device)

        # return policy
        if self.empirical_normalization:
            contactNet_obs_normalizer = self.contactNet_obs_normalizer.to(device)
            obs_normalizer = self.obs_normalizer.to(device)

        policy = PolicyWrapper(
            self.ppo_alg.actor_critic,
            self.ppo_alg.gru,
            self.ppo_alg.tf_encoder,
            obs_normalizer,
            contactNet_obs_normalizer,
            self.env.num_envs,
            self.env.num_obs,
            self.gru_cfg["gru_latent_dim"],
            self.ppo_alg_cfg["next_obs_latent_dim"],
            device=self.device,
        )

        return policy

    def get_inference_vanilla_policy(self, device=None):  # TODO: not complete
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.ppo_alg.actor_critic.to(device)

        # return policy
        if self.empirical_normalization:
            obs_normalizer = self.obs_normalizer.to(device)

        policy = VanillaPolicyWrapper(
            self.ppo_alg.actor_critic,
            obs_normalizer,
            self.env.num_envs,
            self.env.num_obs,
            device=self.device,
        )

        return policy

    def inverse_bool(self, input, ids: list):
        for index in ids:
            lower, upper = index
            input[:, lower:upper] = torch.where(input[:, lower:upper] < 0.1, 0, 1)
        return input

    def get_learning_rate(self, step, d_model, warmup_steps=4000):
        # Calculate the learning rate using the formula from "Attention is All You Need"
        lr = 0.5 * (d_model**-0.5) * min((step + 1) ** (-0.5), step * (warmup_steps**-1.5))
        if self.tot_iter - self.it < 2000:
            lr = min(1e-4, max(1e-4 * (self.tot_iter - self.it) / 2000, 0.0))
        return lr

    # def get_learning_rate(self, step, d_model, warmup_steps=4000, total_steps=10000, min_lr=1e-5):
    #     # Calculate learning rate using cosine annealing after warmup
    #     if step < warmup_steps:
    #         lr = (d_model**-0.5) * (step * warmup_steps**-1.5)
    #     else:
    #         progress = (step - warmup_steps) / (total_steps - warmup_steps)
    #         lr = warmup_steps**-0.5 * (np.cos(np.pi / 2 * progress)) * (d_model**-0.5)
    #         lr = max(lr, min_lr)
    #     return lr

    def train_mode(self):  # TODO: not complete
        # self.actor_critic.eval()
        self.ppo_alg.train_mode()
        # if self.empirical_normalization:
        #     self.obs_normalizer.train()
        #     self.critic_obs_normalizer.train()
        #     self.contactNet_obs_normalizer.train()

    def eval_mode(self):  # TODO: not complete
        self.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()
            self.contactNet_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)


# class CNPolicyWrapper:
#     def __init__(self, actor, normalizer, num_envs, num_obs, device):
#         self.actor = actor.to(device)
#         self.actor.eval()
#         self.num_envs = num_envs
#         self.num_obs = num_obs
#         self.normalizer = None
#         if normalizer is not None:
#             self.normalizer = normalizer.to(device)
#             self.normalizer.eval()
#         self.first_obs = True
#         self.__name__ = "policy"

#     def __call__(self, obs, cn_output):
#         assert obs.dim() == 2, f"Expected obs to be 2D tensor, got {obs.dim()}"
#         num_envs, num_obs = obs.shape
#         assert num_obs == self.num_obs, f"Expected obs to have {self.num_obs} features, got {num_obs}"
#         assert num_envs == self.num_envs, f"Expected obs to have {self.num_envs} environments, got {num_envs}"
#         if self.normalizer is not None:
#             obs = self.normalizer(obs)
#         # latent = self.ts_model.privileged_encoder_forward(critic_obs)
#         return self.actor(torch.cat((obs, cn_output), dim=1))


class VanillaPolicyWrapper:
    def __init__(
        self,
        actor_critic,
        obs_normalizer,
        num_envs,
        num_obs,
        device,
    ):
        self.actor_critic: ActorCritic = actor_critic.to(device)
        self.actor_critic.eval()
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.obs_normalizer = obs_normalizer.to(device)
        self.obs_normalizer.eval()
        self.__name__ = "policy"

    def __call__(self, obs):
        assert obs.dim() == 2, f"Expected obs to be 2D tensor, got {obs.dim()}"
        num_envs, num_obs = obs.shape
        assert num_obs == self.num_obs, f"Expected obs to have {self.num_obs} features, got {num_obs}"
        assert num_envs == self.num_envs, f"Expected obs to have {self.num_envs} environments, got {num_envs}"
        obs = self.obs_normalizer(obs)
        return self.actor_critic.act_inference(obs)


class PolicyWrapper:
    def __init__(
        self,
        actor_critic,
        gru,
        tf_encoder,
        obs_normalizer,
        cn_obs_normlizer,
        num_envs,
        num_obs,
        num_gru_latent,
        next_obs_latent_dim,
        device,
    ):
        self.actor_critic: ActorCritic = actor_critic.to(device)
        self.actor_critic.eval()
        self.tf_encoder: SimplifiedContactNetModel = tf_encoder.to(device)
        self.tf_encoder.eval()
        self.gru: GRUWrapper = gru.to(device)
        self.gru.eval()
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_gru_latent = num_gru_latent
        self.next_obs_latent_dim = next_obs_latent_dim
        self.obs_normalizer = obs_normalizer.to(device)
        self.cn_obs_normalizer = cn_obs_normlizer.to(device)
        self.standard_gaussian = None
        self.obs_normalizer.eval()
        self.cn_obs_normalizer.eval()
        self.__name__ = "policy"

    def __call__(self, obs, cn_obs_hist):
        assert obs.dim() == 2, f"Expected obs to be 2D tensor, got {obs.dim()}"
        assert cn_obs_hist.dim() == 3
        num_envs, num_obs = obs.shape
        assert num_obs == self.num_obs, f"Expected obs to have {self.num_obs} features, got {num_obs}"
        assert num_envs == self.num_envs, f"Expected obs to have {self.num_envs} environments, got {num_envs}"
        cn_obs_hist = self.cn_obs_normalizer(cn_obs_hist)
        cn_output = self.tf_encoder(cn_obs_hist)
        obs = self.obs_normalizer(obs)
        gru_latent = self.gru.gru_forward(cn_output, hx=self.gru.hidden_state)
        mu = gru_latent[:, 3 : 3 + self.next_obs_latent_dim]
        logvar = gru_latent[:, 3 + self.next_obs_latent_dim :]
        self.standard_gaussian = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        next_obs_latent = self.standard_gaussian.sample() * (logvar.exp() + 1e-4).sqrt() + mu
        concat_gru_latent = (
            torch.cat(
                [
                    gru_latent[:, :3],
                    next_obs_latent,
                    gru_latent[:, 3 + 2 * self.next_obs_latent_dim :],
                ],
                dim=-1,
            )
            .clone()
            .detach()
        )
        return self.actor_critic.act_inference(torch.cat((obs, concat_gru_latent), dim=-1))

class TestOnnxPolicyWrapper:
    """
    Test onnx policy wrapper
    """
    def __init__(
        self,
        actor_session,
        gru_session,
        tf_encoder_session,
        num_envs,
        num_obs,
        num_gru_latent,
        next_obs_latent_dim,
        device,
    ):
        self.actor_critic = actor_session
        self.tf_encoder = tf_encoder_session
        self.gru = gru_session
        self.hidden_state = np.zeros((1, num_envs, num_gru_latent), dtype=np.float32)
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_gru_latent = num_gru_latent
        self.next_obs_latent_dim = next_obs_latent_dim
        self.standard_gaussian = None
        self.__name__ = "test_onnx_policy"

    def __call__(self, obs, cn_obs_hist):
        assert obs.dim() == 2, f"Expected obs to be 2D tensor, got {obs.dim()}"
        assert cn_obs_hist.dim() == 3
        num_envs, num_obs = obs.shape
        assert num_obs == self.num_obs, f"Expected obs to have {self.num_obs} features, got {num_obs}"
        assert num_envs == self.num_envs, f"Expected obs to have {self.num_envs} environments, got {num_envs}"

        cn_obs_hist_tensor = np.concatenate([cn_obs_hist.cpu().numpy()], axis=0)
        cn_obs_hist_tensor = cn_obs_hist_tensor.astype(np.float32)
        cn_obs_hist_tensor = {self.tf_encoder.get_inputs()[0].name: cn_obs_hist_tensor}
        cn_output_tensor = self.tf_encoder.run([self.tf_encoder.get_outputs()[0].name], cn_obs_hist_tensor)
        
        gru_input_tensor = {self.gru.get_inputs()[0].name: cn_output_tensor[0], self.gru.get_inputs()[1].name: self.hidden_state}
        gru_output_tensor = self.gru.run([self.gru.get_outputs()[0].name, self.gru.get_outputs()[1].name], gru_input_tensor)
        self.hidden_state = gru_output_tensor[1] # update hidden state
        gru_latent = gru_output_tensor[0]
        gru_latent = torch.tensor(gru_latent)
        mu = gru_latent[:, 3 : 3 + self.next_obs_latent_dim]
        logvar = gru_latent[:, 3 + self.next_obs_latent_dim :]
        self.standard_gaussian = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        next_obs_latent = self.standard_gaussian.sample() * (logvar.exp() + 1e-4).sqrt() + mu
        concat_gru_latent = (
            torch.cat(
                [
                    gru_latent[:, :3],
                    next_obs_latent,
                    gru_latent[:, 3 + 2 * self.next_obs_latent_dim :],
                ],
                dim=-1,
            )
        )
        
        obs = obs.to("cpu")
        obs = np.concatenate([obs], axis=0)
        obs = obs.astype(np.float32)
        concat_gru_latent = np.concatenate([concat_gru_latent], axis=0)
        concat_gru_latent= concat_gru_latent.astype(np.float32)
        actor_inputs = self.actor_critic.get_inputs()
        actor_input_tensor = {
            actor_inputs[0].name: obs,
            actor_inputs[1].name: concat_gru_latent,
        }
        if len(actor_inputs) >= 3:
            action_dim = self.actor_critic.get_outputs()[0].shape[-1]
            noise = np.random.randn(num_envs, action_dim).astype(np.float32)
            actor_input_tensor[actor_inputs[2].name] = noise
        if len(actor_inputs) == 4:
            condition_mask = np.ones((num_envs, self.num_obs + self.num_gru_latent - self.next_obs_latent_dim), dtype=np.float32)
            actor_input_tensor[actor_inputs[3].name] = condition_mask
        actor_output_tensor = self.actor_critic.run([self.actor_critic.get_outputs()[0].name], actor_input_tensor)
        return actor_output_tensor[0]
        
        
        


# class ContactNetWrapper:
#     def __init__(self, contactNet, normalizer, batch_first, num_envs, num_obs, device):
#         self.contactNet = contactNet.to(device)
#         self.device = device
#         self.contactNet.eval()
#         self.num_envs = num_envs
#         self.batch_first = batch_first
#         self.num_obs = num_obs
#         self.normalizer = None
#         if normalizer is not None:
#             self.normalizer = normalizer.to(device)
#             self.normalizer.eval()
#         self.__name__ = "contactNet"

#     def __call__(self, cn_obs):
#         assert cn_obs.dim() == 3, f"Expected obs to be 2D tensor, got {cn_obs.dim()}"
#         if self.batch_first:
#             num_envs, seq_len, num_obs = cn_obs.shape
#         assert num_obs == self.num_obs, f"Expected obs to have {self.num_obs} features, got {num_obs}"
#         # assert seq_len == 10, f"Expected obs to have 10 time steps, got {seq_len}"
#         assert num_envs == self.num_envs, f"Expected obs to have {self.num_envs} environments, got {num_envs}"
#         if self.normalizer is not None:
#             obs = self.normalizer(cn_obs)
#         # latent = self.ts_model.privileged_encoder_forward(critic_obs)
#         return self.contactNet.inference(obs, device=self.device)
