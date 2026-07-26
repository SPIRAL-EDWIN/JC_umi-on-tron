#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, FlowActorCritic, SimplifiedContactNetModel, GRUWrapper
from .ppo import PPO
from rsl_rl.storage import RolloutBuffer
from copy import deepcopy


class PPO_IOS(PPO):
    actor_critic: ActorCritic
    tf_encoder: SimplifiedContactNetModel
    gru: GRUWrapper

    def __init__(
        self,
        actor_critic,
        gru,
        tf_encoder,
        grad_coef=0.001,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        next_obs_latent_dim=32,
        beta=0.1,
        flow_matching=False,
        parameterization="velocity",
        solver_step_size=0.1,
        prior_noise_std=1.0,
        perturb_action_std=0.03,
        sample_t_strategy="uniform",
        device="cpu",
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.tf_gru_lr = learning_rate
        self.next_obs_latent_dim = next_obs_latent_dim
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.tf_encoder = tf_encoder
        self.tf_encoder.to(self.device)
        self.gru = gru
        self.gru.to(self.device)
        self.storage = None  # initialized in OneStageRunner
        self.standard_gaussian = torch.distributions.Normal(0, 1)  # initialized in _compute_losses
        self.beta = beta
        self.flow_matching = flow_matching
        self.parameterization = parameterization
        self.solver_step_size = solver_step_size
        self.prior_noise_std = prior_noise_std
        self.perturb_action_std = perturb_action_std
        self.sample_t_strategy = sample_t_strategy
        self.ppo_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.tf_gru_optimizer = optim.Adam(
            [
                {"params": self.gru.parameters()},
                {"params": self.tf_encoder.parameters()},
            ],
            lr=learning_rate,
        )
        self.ce_criterion_0 = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.mse_criterion = nn.MSELoss(reduction="mean")

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.grad_coef = grad_coef
        self.transition = None  # not used
        self.distributed = False
        self.world_size = 1

    def configure_distributed(self, enabled: bool):
        self.distributed = enabled
        self.world_size = torch.distributed.get_world_size() if enabled else 1

    def _distributed_mean(self, value: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return value
        value = value.clone()
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value / self.world_size

    def _average_gradients(self, parameters):
        if not self.distributed:
            return
        for parameter in parameters:
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        actor_input_shape,
        critic_obs_shape,
        action_shape,
        cn_obs_hist_shape,
        next_obs_shape,
        gru_latent_shape,
    ):
        self.storage = RolloutBuffer(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            actor_input_shape,
            critic_obs_shape,
            action_shape,
            cn_obs_hist_shape,
            next_obs_shape,
            gru_latent_shape,
            self.device,
        )

    def _prepare_actor_input(self, obs, old_gru_latent):
        cn_output = self.tf_encoder(obs["cn_obs_hist"])
        gru_latent = self.gru.gru_forward_without_memory(cn_output, old_gru_latent.unsqueeze(0))
        base_lin_vel_gt = obs["critic_obs"][:, obs["observations"].shape[1] : obs["observations"].shape[1] + 3]
        losses = {
            "lin_vel": (gru_latent[:, :3] - base_lin_vel_gt).abs().mean(),
        }

        mu = gru_latent[:, 3 : 3 + self.next_obs_latent_dim]
        logvar = gru_latent[:, 3 + self.next_obs_latent_dim :]
        losses["beta_vae"] = self._calc_kld_normal(mu, logvar)
        standard_gaussian = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        next_obs_latent = standard_gaussian.sample() * (logvar.exp() + 1e-4).sqrt() + mu
        next_obs_est = self.tf_encoder.next_obs_decoder(next_obs_latent)
        losses["next_obs"] = (next_obs_est - obs["next_obs_gt"]).abs().mean()

        concat_gru_latent = torch.cat(
            [
                gru_latent[:, :3],
                next_obs_latent,
                gru_latent[:, 3 + 2 * self.next_obs_latent_dim :],
            ],
            dim=-1,
        ).detach()
        actor_input = torch.cat([obs["observations"], concat_gru_latent], dim=-1)
        return actor_input, losses

    def _refresh_flow_rollout_stats(self):
        action_buffer = self.storage.buffers["actions"]
        actor_input_buffer = self.storage.buffers["actor_inputs"]
        flat_actions = action_buffer.flatten(0, 1)
        flat_actor_inputs = actor_input_buffer.flatten(0, 1)

        with torch.no_grad():
            old_logprob, _, noise, t, condition_mask = self.actor_critic.flow_matching_loss(
                flat_actions,
                flat_actor_inputs,
                use_condition_dropout=True,
                return_noise_t=True,
            )

        old_logprob = old_logprob.view_as(self.storage.buffers["fm_old_logprob"])
        t = t.unsqueeze(-1).view_as(self.storage.buffers["fm_t"])
        noise = noise.view_as(self.storage.buffers["fm_noise"])
        condition_mask = condition_mask.view_as(self.storage.buffers["fm_condition_mask"])
        self.storage.buffers["fm_old_logprob"].copy_(old_logprob)
        self.storage.buffers["fm_t"].copy_(t)
        self.storage.buffers["fm_noise"].copy_(noise)
        self.storage.buffers["fm_condition_mask"].copy_(condition_mask)

    def test_mode(self):
        self.actor_critic.eval()
        self.gru.eval()
        self.tf_encoder.eval()

    def train_mode(self):
        self.actor_critic.train()
        self.gru.train()
        self.tf_encoder.train()

    def _calc_grad_penalty(self, obs_batch_list: list[torch.Tensor], actions_log_prob_batch):
        gradient_penalty_loss = 0
        for obs_batch in obs_batch_list:
            grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
            gradient_penalty_loss += torch.sum(torch.abs(grad_log_prob), dim=-1).mean()
        gradient_penalty_loss /= len(obs_batch_list)
        return gradient_penalty_loss

    def act(self, obs, critic_obs, cn_obs_hist):
        self.storage.transition["observations"] = obs
        self.storage.transition["privileged_observations"] = critic_obs
        # self.storage.transition["base_lin_vel_gt"] = critic_obs[:, obs.shape[1] : obs.shape[1] + 3]
        self.storage.transition["cn_obs_history"] = cn_obs_hist
        self.storage.transition["gru_latent"] = self.gru.hidden_state.clone()
        cn_output = self.tf_encoder(cn_obs_hist)
        next_gru_latent = self.gru.gru_forward(cn_output, self.storage.transition["gru_latent"]).clone()
        ## sample the next_obs_latent
        mu = next_gru_latent[:, 3 : 3 + self.next_obs_latent_dim]
        logvar = next_gru_latent[:, 3 + self.next_obs_latent_dim :]
        distribution = torch.distributions.Normal(mu, (logvar.exp() + 1e-4).sqrt())
        next_obs_latent = distribution.sample()
        next_gru_latent = torch.cat(
            [
                next_gru_latent[:, :3],
                next_obs_latent,
                next_gru_latent[:, 3 + 2 * self.next_obs_latent_dim :],
            ],
            dim=-1,
        )
        actor_input = torch.cat([obs, next_gru_latent.detach()], dim=-1)
        self.storage.transition["actor_inputs"] = actor_input
        actions = self.actor_critic.act(actor_input)
        # actions = self.corrupt_action(actions, num=1024)
        self.storage.transition["actions"] = actions
        if self.flow_matching:
            zero_scalar = torch.zeros(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
            self.storage.transition["actions_log_prob"] = zero_scalar
            self.storage.transition["action_mean"] = self.actor_critic.action_mean.detach()
            self.storage.transition["action_sigma"] = self.actor_critic.action_std.detach()
            self.storage.transition["fm_old_logprob"] = zero_scalar
            self.storage.transition["fm_t"] = zero_scalar
            self.storage.transition["fm_noise"] = torch.zeros_like(actions)
            self.storage.transition["fm_condition_mask"] = torch.ones_like(actor_input)
        else:
            self.storage.transition["actions_log_prob"] = self.actor_critic.get_actions_log_prob(actions).detach().unsqueeze(-1)
            self.storage.transition["action_mean"] = self.actor_critic.action_mean.detach()
            self.storage.transition["action_sigma"] = self.actor_critic.action_std.detach()
            self.storage.transition["fm_old_logprob"] = torch.zeros(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
            self.storage.transition["fm_t"] = torch.zeros(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
            self.storage.transition["fm_noise"] = torch.zeros_like(actions)
            self.storage.transition["fm_condition_mask"] = torch.ones_like(actor_input)
        self.storage.transition["values"] = self.actor_critic.evaluate(critic_obs).detach()
        return actions

    def corrupt_action(self, actions, num=2048):
        actions[:num, :] = torch.distributions.Normal(0.0, 5.0).sample(actions[:num, :].shape).to(self.device)
        # actions[:] = 0
        return actions

    def process_env_step(self, rewards, dones, infos, next_obs):
        self.storage.transition["rewards"] = rewards.clone()
        self.storage.transition["dones"] = dones.clone()
        self.storage.transition["next_obs_gt"] = next_obs.clone()
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.storage.transition["rewards"] += self.gamma * torch.squeeze(
                self.storage.transition["values"] * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions()
        self.storage.reset_transition()
        self.gru.reset_hidden_states(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        if self.distributed:
            # RolloutBuffer normalizes locally. Recompute from the unmodified
            # returns/values and normalize over the full cross-rank batch.
            advantages = self.storage.buffers["returns"] - self.storage.buffers["values"]
            moments = torch.stack(
                (
                    advantages.sum(),
                    advantages.square().sum(),
                    torch.tensor(advantages.numel(), device=self.device, dtype=advantages.dtype),
                )
            )
            torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
            mean = moments[0] / moments[2]
            variance = moments[1] / moments[2] - mean.square()
            self.storage.buffers["advantages"].copy_(
                (advantages - mean) / (variance.clamp_min(0.0).sqrt() + 1.0e-8)
            )

    def _calc_kld_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds.mean()

    def _compute_losses(self, mini_batch):
        losses = {}
        # -- prepare quantities
        old_actions_log_prob_batch = mini_batch["actions_log_prob"]
        advantages_batch = mini_batch["advantages"]
        returns_batch = mini_batch["returns"]
        critic_obs = mini_batch["privileged_observations"]
        target_values_batch = mini_batch["values"]
        action_batch = mini_batch["actions"]
        old_sigma_batch = mini_batch["action_sigma"]
        old_mu_batch = mini_batch["action_mean"]
        obs = mini_batch["observations"]
        base_lin_vel_gt = critic_obs[:, obs.shape[1] : obs.shape[1] + 3]
        next_obs_gt = mini_batch["next_obs_gt"]
        old_gru_latent = mini_batch["gru_latent"]
        cn_obs_hist = mini_batch["cn_obs_history"]

        prepared_obs = {
            "observations": obs,
            "critic_obs": critic_obs,
            "cn_obs_hist": cn_obs_hist,
            "next_obs_gt": next_obs_gt,
        }
        _, aux_losses = self._prepare_actor_input(prepared_obs, old_gru_latent)
        losses.update(aux_losses)

        value_batch = self.actor_critic.evaluate(critic_obs)
        for param_group in self.tf_gru_optimizer.param_groups:
            param_group["lr"] = self.tf_gru_lr
        for param_group in self.ppo_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        if self.flow_matching:
            actor_input = mini_batch["actor_inputs"].clone().detach()
            actor_input.requires_grad_(True)
            fm_old_logprob = mini_batch["fm_old_logprob"].squeeze(-1)
            fm_t = mini_batch["fm_t"].squeeze(-1)
            fm_noise = mini_batch["fm_noise"]
            fm_condition_mask = mini_batch["fm_condition_mask"]
            new_logprob, _, = self.actor_critic.flow_matching_loss(
                action_batch,
                actor_input,
                noise=fm_noise,
                t=fm_t,
                condition_mask=fm_condition_mask,
                use_condition_dropout=False,
            )
            ratio = torch.exp(new_logprob - fm_old_logprob)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            losses["surrogate"] = torch.max(surrogate, surrogate_clipped).mean()
            losses["grad_penalty"] = self._calc_grad_penalty([actor_input], new_logprob)
            log_ratio = new_logprob - fm_old_logprob
            losses["kl"] = (((ratio - 1.0) - log_ratio)).mean()
            losses["entropy"] = torch.zeros((), device=self.device)
        else:
            actor_input = mini_batch["actor_inputs"].clone().detach()
            actor_input.requires_grad_(True)
            self.actor_critic.act(actor_input)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(action_batch)
            losses["grad_penalty"] = self._calc_grad_penalty([actor_input], actions_log_prob_batch)

            sigma_batch = self.actor_critic.action_std
            mu_batch = self.actor_critic.action_mean

            losses["kl"] = self._update_lr_kl(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            losses["surrogate"] = torch.max(surrogate, surrogate_clipped).mean()
            losses["entropy"] = self.actor_critic.entropy.mean()

        # -- Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            losses["value"] = torch.max(value_losses, value_losses_clipped).mean()
        else:
            losses["value"] = (returns_batch - value_batch).pow(2).mean()

        return losses

    def _update_lr_kl(self, sigma_batch, old_sigma_batch, mu_batch, old_mu_batch):
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                sigma_batch = sigma_batch.clamp_min(1.0e-6)
                old_sigma_batch = old_sigma_batch.clamp_min(1.0e-6)
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = self._distributed_mean(torch.mean(kl))

                if torch.isfinite(kl_mean):
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(
                            self.initial_learning_rate,
                            self.learning_rate * 1.5,
                        )
                    for param_group in self.ppo_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                return kl_mean

    def update_gru(self, rewards, next_critic_obs):
        raise NotImplementedError

    def update(self, it):
        accumulated_losses = {
            "value": 0.0,
            "surrogate": 0.0,
            "kl": 0.0,
            "entropy": 0.0,
            "grad_penalty": 0.0,
            "lin_vel": 0.0,
            "next_obs": 0.0,
            "beta_vae": 0.0,
        }
        if self.flow_matching:
            self._refresh_flow_rollout_stats()
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for mini_batch in generator:

            # -- compute loss
            loss_item = self._compute_losses(mini_batch)
            invalid_losses = [
                name for name, loss in loss_item.items() if not torch.isfinite(loss).all()
            ]
            if self.distributed:
                invalid_flag = torch.tensor(bool(invalid_losses), device=self.device, dtype=torch.int32)
                torch.distributed.all_reduce(invalid_flag, op=torch.distributed.ReduceOp.MAX)
                if invalid_flag.item() and not invalid_losses:
                    invalid_losses = ["non-finite loss on another rank"]
            if invalid_losses:
                self.tf_gru_optimizer.zero_grad()
                self.ppo_optimizer.zero_grad()
                raise FloatingPointError(
                    f"Non-finite PPO losses at iteration {it}: {invalid_losses}. "
                    "Optimizer step was skipped to preserve finite model weights."
                )

            mse_loss = loss_item["lin_vel"] + loss_item["next_obs"] + self.beta * loss_item["beta_vae"]
            ppo_loss = (
                loss_item["surrogate"]
                + self.value_loss_coef * loss_item["value"]
                - self.entropy_coef * loss_item["entropy"]
            )
            if self.grad_coef != 0.0:
                # Avoid 0 * inf producing NaN when gradient-penalty logging is
                # enabled but the penalty is not part of the optimization.
                ppo_loss = ppo_loss + self.grad_coef * loss_item["grad_penalty"]

            # Build and validate both gradient sets before either optimizer is
            # allowed to mutate model weights.
            self.tf_gru_optimizer.zero_grad()
            self.ppo_optimizer.zero_grad()
            mse_loss.backward()
            ppo_loss.backward()

            self._average_gradients(self.gru.parameters())
            self._average_gradients(self.tf_encoder.parameters())
            self._average_gradients(self.actor_critic.parameters())
            grad_norms = {
                "gru": nn.utils.clip_grad_norm_(self.gru.parameters(), self.max_grad_norm),
                "tf_encoder": nn.utils.clip_grad_norm_(self.tf_encoder.parameters(), self.max_grad_norm),
                "actor_critic": nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm),
            }
            invalid_gradients = [
                name for name, norm in grad_norms.items() if not torch.isfinite(norm)
            ]
            if self.distributed:
                invalid_flag = torch.tensor(bool(invalid_gradients), device=self.device, dtype=torch.int32)
                torch.distributed.all_reduce(invalid_flag, op=torch.distributed.ReduceOp.MAX)
                if invalid_flag.item() and not invalid_gradients:
                    invalid_gradients = ["non-finite gradient on another rank"]
            if invalid_gradients:
                self.tf_gru_optimizer.zero_grad()
                self.ppo_optimizer.zero_grad()
                raise FloatingPointError(
                    f"Non-finite gradient norms at iteration {it}: {invalid_gradients}. "
                    "Optimizer step was skipped to preserve finite model weights."
                )

            self.tf_gru_optimizer.step()
            self.ppo_optimizer.step()
            for key in loss_item:
                accumulated_losses[key] += self._distributed_mean(loss_item[key].detach()).item()
        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {name: loss / num_updates for name, loss in accumulated_losses.items()}
