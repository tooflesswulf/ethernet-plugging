"""Real-robot test-time adaptation for the ethernet-plugging DiffusionPolicy.

Ports the contrastive TTA update used in this directory's sim/offline codebase
(`do_tta_update` -> `tta_update_dpo` -> `DiffusionUnetAgent.forward_dpo`, see
TTA_evaluator_real.py and TD_diffusion_policy.py / diffusion_unet.py) to the
real-robot `agent.model.policy.DiffusionPolicy` driven by test-far.py.

Key difference from the reference implementation: that version mines positive
action targets using a value net + preference net trained alongside the
policy (`construct_cl_dataset_with_critic`). `DiffusionPolicy` here has no
such critic, so positives are instead obtained by resampling the current
policy at the same observation and keeping whichever candidate is farthest
(L2, normalized action space) from the negative/failed chunk -- the same
distance-based tie-breaker `construct_cl_dataset_with_critic` itself falls
back on once it has narrowed down a candidate pool.
"""

import collections

import einops
import numpy as np
import torch
import torch.nn as nn

from agent.model.policy import DiffusionPolicy
from agent.utils.utils import resize_image
from agent.utils.robot_utils import build_states


def _build_conditions(policy: DiffusionPolicy, obs_deque, device):
    img_size = policy.img_size
    images = np.stack([resize_image(o['image'], (img_size, img_size), flip_channel=True) for o in obs_deque])
    states = build_states(obs_deque, policy.obs_fields)

    nimages = einops.rearrange(
        torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
    nstates = torch.from_numpy(states).to(device, dtype=torch.float32)
    return {
        'rgb': (nimages / 255.0).unsqueeze(0),   # (1, T, C, H, W)
        'state': nstates.unsqueeze(0),           # (1, T, state_dim)
    }


@torch.no_grad()
def predict_action_normalized(policy: DiffusionPolicy, conditions):
    """Like DiffusionPolicy.predict_action, but returns the network's raw
    normalized output (before unnormalize_actions/integrate_actions), i.e.
    the space the diffusion training loss operates in.
    """
    obs_cond = policy.encode_obs(conditions)
    device = obs_cond.device
    naction = torch.randn((1, policy.action_horizon, policy.action_dim), device=device)
    for k in policy.noise_scheduler.timesteps:
        noise_pred = policy.nets['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)
        naction = policy.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
    return naction


@torch.no_grad()
def get_actions_with_naction(policy: DiffusionPolicy, obs_deque, device='cuda'):
    """Same as agent.utils.robot_utils.get_actions, but also returns the
    conditioning + raw normalized action chunk so a TTATrajectoryRecorder can
    tie each executed chunk back to the exact network output that produced
    it (needed for the TTA loss below). Runs a single diffusion sampling
    pass, same cost as the original get_actions.
    """
    conditions = _build_conditions(policy, obs_deque, device)
    naction = predict_action_normalized(policy, conditions)

    last = obs_deque[-1]['state']
    curr_pose, curr_gripper_width = np.asarray(last['actual_pose']), last['gripper_width']
    unnorm_actions = policy.unnormalize_actions(naction).detach().cpu().numpy()[0]
    des_poses, des_grips, des_done = policy.integrate_actions(unnorm_actions, curr_pose, curr_gripper_width)
    return des_poses, des_grips, des_done, conditions, naction


class TTATrajectoryRecorder:
    """Rolling record of (conditions, normalized raw action chunk) pairs, one
    per prediction-loop tick. Each entry already carries the exact network
    output for its observation -- unlike the robot's executed motion, which
    is a recency-weighted blend of several overlapping async chunks and
    can't be tied back to a single (obs, action) pair.
    """

    def __init__(self, maxlen=200):
        self.entries = collections.deque(maxlen=maxlen)

    def record(self, conditions, naction_normalized):
        self.entries.append((
            {k: v.detach() for k, v in conditions.items()},
            naction_normalized.detach(),
        ))

    def recent(self, n):
        return list(self.entries)[-n:]

    def __len__(self):
        return len(self.entries)


def do_tta_update(
    policy: DiffusionPolicy,
    recorder: TTATrajectoryRecorder,
    device='cuda',
    num_negatives=4,
    num_candidates=4,
    train_steps=5,
    lr=1e-5,
    beta=0.1,
    bc_weight=1.0,
):
    """Real-robot port of TTA_evaluator_real.py:do_tta_update /
    TD_diffusion_policy.py:tta_update_dpo /
    diffusion_unet.py:DiffusionUnetAgent.forward_dpo.

    Uses the most recent `num_negatives` recorded (obs, action) entries --
    the network's own predictions that led into the just-detected failure --
    as the DPO "loser" examples. For each, resamples `num_candidates`
    alternative action chunks at the same observation and keeps the one
    farthest from the negative as the "winner" target. Then runs
    `train_steps` Adam updates of noise_pred_net on the Diffusion-DPO loss:
    push the winner's noise-prediction error down and the loser's up (same
    noise draw for both, per forward_dpo), plus a BC term on the winner.
    """
    entries = recorder.recent(num_negatives)
    if len(entries) == 0:
        print('TTA: no recorded trajectory, skipping update')
        return

    noise_net = policy.nets['noise_pred_net']
    noise_net.eval()

    neg_actions = torch.cat([ac for _, ac in entries], dim=0)  # (B, horizon, action_dim)
    pos_candidates = []
    for conditions, neg_ac in entries:
        best, best_dist = None, -1.0
        for _ in range(num_candidates):
            cand = predict_action_normalized(policy, conditions)
            dist = (cand - neg_ac).norm(p=2, dim=-1).mean().item()
            if best is None or dist > best_dist:
                best, best_dist = cand, dist
        pos_candidates.append(best)
    pos_actions = torch.cat(pos_candidates, dim=0)

    with torch.no_grad():
        obs_conds = torch.cat([policy.encode_obs(c) for c, _ in entries], dim=0)

    scheduler = policy.noise_scheduler
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    B = neg_actions.shape[0]

    noise_net.train()
    optimizer = torch.optim.Adam(noise_net.parameters(), lr=lr)
    for step in range(train_steps):
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()

        # Same noise draw for winner and loser, per forward_dpo -- controls variance.
        noise = torch.randn_like(pos_actions)
        noisy_pos = scheduler.add_noise(pos_actions, noise, timesteps)
        noisy_neg = scheduler.add_noise(neg_actions, noise, timesteps)

        noise_pred_pos = noise_net(noisy_pos, timesteps, global_cond=obs_conds)
        noise_pred_neg = noise_net(noisy_neg, timesteps, global_cond=obs_conds)

        error_pos = nn.functional.mse_loss(noise_pred_pos, noise, reduction='none').sum(dim=(1, 2))
        error_neg = nn.functional.mse_loss(noise_pred_neg, noise, reduction='none').sum(dim=(1, 2))

        loss_dpo = -nn.functional.logsigmoid(beta * (error_neg - error_pos)).mean()
        loss_bc = error_pos.mean()
        total_loss = loss_dpo + bc_weight * loss_bc

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(noise_net.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"TTA step {step}: dpo={loss_dpo.item():.4f} bc={loss_bc.item():.4f} total={total_loss.item():.4f}")

    noise_net.eval()
    print(f"TTA update complete ({len(entries)} negative example(s), {train_steps} steps)")
