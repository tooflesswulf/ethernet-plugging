
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from agent.model.networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn


class DiffusionPolicy(nn.Module):
    """
    Wraps the vision encoder + noise prediction network with action/observation
    normalization. Normalization stats are stored as buffers, so they are saved
    and restored through the regular state_dict / checkpoint machinery — no
    separate stats file needed at eval time.

    Training:   loss = policy.compute_loss(actions, conditions)
    Inference:  actions = policy.predict_action(conditions)  # unnormalized
    """

    def __init__(
        self,
        obs_horizon=1,
        action_horizon=16,
        vision_feature_dim=512,
        state_dim=7,
        action_dim=7,
        num_diffusion_iters=100,
        norm_stats: dict | None = None,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters

        # construct ResNet18 encoder; replace all BatchNorm with GroupNorm to
        # work with EMA — performance will tank if you forget to do this!
        vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=(vision_feature_dim + state_dim) * obs_horizon,
        )
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net,
        })

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',  # squared cosine works the best
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type='epsilon',  # network predicts noise
        )

        # Normalization stats as buffers: saved in state_dict, moved with .to(device).
        # Defaults are the identity transform ([-1, 1] range).
        self.register_buffer('action_min', -torch.ones(action_dim))
        self.register_buffer('action_max', torch.ones(action_dim))
        self.register_buffer('state_min', -torch.ones(state_dim))
        self.register_buffer('state_max', torch.ones(state_dim))
        if norm_stats is not None:
            self.set_norm_stats(norm_stats)

    def set_norm_stats(self, stats: dict):
        """stats: output of agent.utils.utils.compute_norm_stats"""
        self.action_min.copy_(torch.as_tensor(stats['actions']['min'], dtype=torch.float32))
        self.action_max.copy_(torch.as_tensor(stats['actions']['max'], dtype=torch.float32))
        self.state_min.copy_(torch.as_tensor(stats['states']['min'], dtype=torch.float32))
        self.state_max.copy_(torch.as_tensor(stats['states']['max'], dtype=torch.float32))

    @staticmethod
    def _normalize(x, min_val, max_val):
        range_val = max_val - min_val
        safe_range = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return torch.where(range_val == 0, x, 2 * (x - min_val) / safe_range - 1)

    @staticmethod
    def _unnormalize(x, min_val, max_val):
        range_val = max_val - min_val
        return torch.where(range_val == 0, x, (x + 1) / 2 * range_val + min_val)

    def normalize_actions(self, actions):
        return self._normalize(actions, self.action_min, self.action_max)

    def unnormalize_actions(self, actions):
        return self._unnormalize(actions, self.action_min, self.action_max)

    def normalize_states(self, states):
        return self._normalize(states, self.state_min, self.state_max)

    def encode_obs(self, conditions):
        """
        conditions: {'rgb': (B, T, C, H, W) in [0, 1], 'state': (B, T, state_dim) raw}
        Returns flattened observation conditioning (B, T * obs_dim).
        """
        images = conditions['rgb'].float()
        states = self.normalize_states(conditions['state'].float())
        # BxTxCxHxW -> (B T)xCxHxW -> (B T) x d -> BxTxd
        image_features = self.nets['vision_encoder'](images.flatten(end_dim=1)).reshape(*images.shape[:2], -1)
        obs_features = torch.cat([image_features, states], dim=-1)
        return obs_features.flatten(start_dim=1)

    def compute_loss(self, actions, conditions):
        """
        Diffusion (epsilon-prediction) training loss on normalized actions.

        actions:    (B, horizon, action_dim) raw/unnormalized
        conditions: {'rgb', 'state'} as in encode_obs
        """
        actions = self.normalize_actions(actions.float())
        obs_cond = self.encode_obs(conditions)
        B = actions.shape[0]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, conditions, num_inference_steps=None):
        """
        Run reverse diffusion and return UNNORMALIZED actions (B, action_horizon, action_dim).
        """
        obs_cond = self.encode_obs(conditions)
        B = obs_cond.shape[0]
        device = obs_cond.device

        naction = torch.randn((B, self.action_horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(num_inference_steps or self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.nets['noise_pred_net'](
                sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction).prev_sample

        return self.unnormalize_actions(naction)
