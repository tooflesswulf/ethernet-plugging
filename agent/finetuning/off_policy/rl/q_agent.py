# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn
from einops import rearrange
from agent.finetuning.config.rlpd import QAgentConfig
from agent.finetuning.off_policy import common_utils
from agent.finetuning.off_policy.common_utils import utils
from agent.model.networks import VitEncoder
from agent.finetuning.off_policy.rl.actor import Actor
from agent.finetuning.off_policy.rl.critic import Critic

def to_torch(inp, device):
    for k, v in inp.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        inp[k] = v.float().to(device)
    return inp


class QAgent(nn.Module):
    def __init__(
        self,
        condition_length: int,
        prop_dim: int,
        action_dim: int,
        cfg: QAgentConfig,
        residual_actor: bool = False,
    ):
        """Initialize the Q-agent.

        Parameters
        ----------
        prop_dim : tuple[int]
            Number of state dimensions.
        action_dim : int
            Number of action dimensions.
        rl_cameras : list[str] | str
            Name(s) of the camera images to be used by the RL policy.
            These are keys into the env's observation. A single string
            is accepted for backwards-compatibility but the preferred interface
            is to pass a list of camera names.
        cfg : QAgentConfig
            Hyper-parameter configuration dataclass.
        """
        super().__init__()
        # Normalise *rl_cameras* to a list for unified processing
        
        self.cfg = cfg
        self.residual_actor = residual_actor
        self.condition_length = condition_length
        # create critics & actor
        self.critic = Critic( prop_dim=prop_dim, action_dim=action_dim, cfg=self.cfg.critic )
        self.actor = Actor(condition_length, prop_dim, action_dim, cfg.actor, residual_actor=residual_actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        # Dataset normalization stats as buffers: saved in state_dict, moved with .to(device).
        # Defaults are the identity transform (states untouched, actions already in [-1, 1]),
        # so an agent whose stats were never set behaves exactly as before.
        self.register_buffer("action_min", -torch.ones(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))
        self.register_buffer("state_mean", torch.zeros(prop_dim))
        self.register_buffer("state_std", torch.ones(prop_dim))
        self.register_buffer("_norm_stats_set", torch.tensor(False))

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        
        # Create optimizers (PyTorch will ignore frozen parameters)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr)

        # LR schedulers for warmup (if warmup is enabled)
        self.critic_scheduler = None
        self.actor_scheduler = None

        if self.cfg.lr_warmup_steps > 0:
            # LinearLR scheduler that linearly ramps from start_factor to 1.0 over total_iters steps
            # Note: start_factor must be > 0 for LinearLR scheduler
            warmup_start = self.cfg.lr_warmup_start

            # Calculate start factors for each optimizer
            critic_start_factor = warmup_start / self.cfg.critic_lr if self.cfg.critic_lr > 0 else 1e-8
            critic_start_factor = max(critic_start_factor, 1e-8)

            actor_start_factor = warmup_start / self.cfg.actor_lr if self.cfg.actor_lr > 0 else 1e-8
            actor_start_factor = max(actor_start_factor, 1e-8)

            # Create schedulers with appropriate start factor
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=self.cfg.lr_warmup_steps
            )

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    # ------------------------------------------------------------------
    # Observation / action normalization --------------------------------
    # ------------------------------------------------------------------
    def set_norm_stats(self, stats: dict, min_action_range: float = 1e-1, min_state_std: float = 1e-1):
        """Install the dataset normalization statistics.

        Args:
            stats: ``{'actions': {'min', 'max'}, 'states': {'mean', 'std'}}`` in raw
                (unnormalized) units -- see ``_compute_dataset_norm_stats`` in
                ``train_residual_rl.py``.
            min_action_range: per-dim floor on the action range, so a dimension that
                barely moves in the dataset does not blow up when scaled to [-1, 1].
            min_state_std: per-dim floor on the state std, same reasoning.

        The safeguards are applied here rather than at use time, so the buffers that
        land in a checkpoint are exactly the transform act()/update() apply.
        """
        action_min = torch.as_tensor(stats["actions"]["min"], dtype=torch.float32)
        action_max = torch.as_tensor(stats["actions"]["max"], dtype=torch.float32)
        state_mean = torch.as_tensor(stats["states"]["mean"], dtype=torch.float32)
        state_std = torch.as_tensor(stats["states"]["std"], dtype=torch.float32)

        # Widen degenerate action dims symmetrically around their midpoint
        action_mid = (action_min + action_max) / 2
        action_half_range = torch.clamp((action_max - action_min) / 2, min=min_action_range / 2)

        self.action_min.copy_(action_mid - action_half_range)
        self.action_max.copy_(action_mid + action_half_range)
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(torch.clamp(state_std, min=min_state_std))
        self._norm_stats_set.fill_(True)

        print("QAgent normalization stats set:")
        print(f"  action min: {self.action_min.tolist()}")
        print(f"  action max: {self.action_max.tolist()}")
        print(f"  state mean: {self.state_mean.tolist()}")
        print(f"  state std : {self.state_std.tolist()}")

    @property
    def norm_stats_set(self) -> bool:
        """False while the agent still holds the identity-transform defaults."""
        return bool(self._norm_stats_set)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Raw state -> zero-mean / unit-std."""
        return (state - self.state_mean) / self.state_std

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Raw action -> [-1, 1] (clamped, matching how the env executes actions)."""
        action = torch.clamp(action, self.action_min, self.action_max)
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """[-1, 1] action -> raw units."""
        action = torch.clamp(action, -1.0, 1.0)
        return self.action_min + (action + 1.0) * (self.action_max - self.action_min) / 2.0

    def unscale_action_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """[-1, 1]-space *offset* -> raw-space offset.

        A residual is a difference of two actions, so only the linear part of the
        scaling applies -- shifting by action_min would double-count the offset.
        """
        return delta * (self.action_max - self.action_min) / 2.0

    def _normalize_obs(self, obs):
        """Raw observation -> the normalized view the networks are trained on.

        Writes into a shallow copy, so the caller's dict (or the sampled batch) keeps
        its raw values. Images are left alone; ``_encode`` handles those.
        """
        obs = obs.copy()
        obs["observation.state"] = self.normalize_state(obs["observation.state"].float())
        if "observation.base_action" in obs.keys():
            obs["observation.base_action"] = self.scale_action(obs["observation.base_action"].float())
        return obs

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _min_over_random_two(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute min over a random subset of 2 heads from q_values [K, B, 1]."""
        assert q_values.dim() == 3 and q_values.size(-1) == 1
        num_heads = q_values.size(0)
        if num_heads <= 2:
            return torch.min(q_values[0], q_values[1])
        # Sample two unique head indices uniformly at random
        idx = torch.randperm(num_heads, device=q_values.device)[:2]
        subset = q_values.index_select(dim=0, index=idx)  # [2, B, 1]
        return subset.min(dim=0).values

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def maskout_action(self, action):
        weight = torch.zeros_like(action, device = action.device).float(); weight[:,2] = 1.0
        # only 2nd dimension valid
        action *= weight
        return action

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        """Take a *raw* (unnormalized) observation and return a *raw* action.

        Normalization happens inside: the state is standardized and the base action is
        scaled to [-1, 1] before the networks see them, and the action coming out of the
        actor is converted back to raw units -- as a residual offset when the actor is a
        residual actor, otherwise as a full action.
        """
        assert not self.training
        assert not self.actor.training
        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        obs = to_torch(obs, device=self.cfg.device)
        obs = self._normalize_obs(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        action = self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=None,
            use_target=False,
        )

        if unsqueezed:
            action = action.squeeze(0)

        # Networks work in [-1, 1]; the caller (env) works in raw units
        action = self.unscale_action_delta(action) if self.residual_actor else self.unscale_action(action)

        action = action.detach()
        if cpu:
            action = action.cpu()
        # only z-axis is effective as a residual action
        
        action = self.maskout_action(action)
        
        return action

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: float | None,
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
 
        dist = actor.forward(obs, stddev)

        # Only assert not training when this is called from the public act() method
        # (which is used for actual evaluation), not when called internally during training
        if eval_mode and not use_target:
            assert not self.training

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)
        # only z-axis is effective as a residual action
        action = self.maskout_action(action)
        return action

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        importance_weights: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            # Predict next residual action and form the combined next action
            # Use target_action_noise config to control whether to add noise to target actions
            next_residual_action = self._act_default(
                obs=next_obs,
                eval_mode=not self.cfg.target_action_noise,  # Disable noise if target_action_noise=False
                stddev=stddev,
                clip=self.cfg.stddev_clip,
                use_target=True,
            )
 
            if self.residual_actor:
                next_base_action = next_obs['observation.base_action'][:, -1]
                # Current step: 'action' from the replay buffer is the executed combined action
                # Next step: combine and clamp to match environment execution
                next_action = torch.clamp(next_base_action + next_residual_action, -1.0, 1.0)
            else:
                next_action = next_residual_action

            # Compute target Q using min over a random subset of 2 heads
            target_all = self.critic_target.q_value( next_obs["observation.state"][:, -1], next_action)
            target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)  # Sparse rewards are in {0, 1}

        td_errors = None
        q_all = self.critic(obs["observation.state"][:, -1], action).squeeze(-1)  # [K,B]
        # Compute TD errors for prioritized experience replay (before taking mean)
        td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

        # Apply importance sampling weights if provided (for prioritized experience replay)
        if importance_weights is not None:
            # Weight the squared TD errors by importance sampling weights
            weighted_td_errors = td_errors**2 * importance_weights
            critic_loss = weighted_td_errors.mean()
        else:
            # Mean squared error across heads and batch (uniform sampling)
            critic_loss = (td_errors**2).mean()

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()
        # Store target_q for potential logging (calculated only when needed)
        metrics["_target_q"] = target_q.detach().cpu()
        # Store TD errors for prioritized experience replay
        if td_errors is not None:
            metrics["_td_errors"] = td_errors.detach().cpu()
        # Log importance sampling weights for monitoring PER behavior
        if importance_weights is not None:
            metrics["train/importance_weights_mean"] = importance_weights.mean().item()
            metrics["train/importance_weights_std"] = importance_weights.std().item()
            metrics["train/importance_weights_min"] = importance_weights.min().item()
            metrics["train/importance_weights_max"] = importance_weights.max().item()

        # Zero gradients
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        # Gradient clipping
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)

        # Store gradient norms for logging
        metrics["train/critic_grad_norm"] = critic_grad_norm.item()
        self.critic_opt.step()

        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
    
        action_pred: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            # stddev=stddev,
            # NOTE: This fix has not been fully verified yet.
            stddev=0.0,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )

        # Add L2 regularization on action magnitude (before we add the residual action to the base)
        action_l2_penalty = self.cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))

        if self.residual_actor:
            # Create the full action by combining the base action and the residual action
            # and clamp to the valid range [-1, 1] to match environment execution
            combined_action = torch.clamp(obs["observation.base_action"][:, -1] + action_pred, -1.0, 1.0)
        else:
            combined_action = action_pred

        q = self.critic.q_value_for_policy( obs["observation.state"][:, -1], combined_action)
        actor_loss_base = -q.mean()

        actor_loss_total = actor_loss_base + action_l2_penalty

        return actor_loss_total, actor_loss_base, combined_action, action_pred, action_l2_penalty

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_total.backward()

        # Gradient clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)

        # Store gradient norm for logging
        metrics["train/actor_grad_norm"] = actor_grad_norm.item()

        self.actor_opt.step()

        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
    ):
        # The buffers hold raw measurements; normalize them here so every caller can
        # stay in raw units (see set_norm_stats / _normalize_obs).
        obs: dict[str, torch.Tensor] = self._normalize_obs(batch["obs"].float())
        action: torch.Tensor = self.scale_action(batch["action"].float())
        reward: torch.Tensor = batch[("next", "reward")].float()
        discount: torch.Tensor = batch["gamma"]
        next_nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: dict[str, torch.Tensor] = self._normalize_obs(batch[("next", "obs")].float())
    
        # To not bootstrap on terminal states we zero out the discount factor for terminal next states
        effective_discount = discount * next_nonterminal

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()

        # Extract importance sampling weights if available (for prioritized experience replay)
        importance_weights = batch.get("_weight", None)

        critic_metric = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            stddev=stddev,
            importance_weights=importance_weights,
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        actor_metric = self.update_actor(obs, stddev)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()