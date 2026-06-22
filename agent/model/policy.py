
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R, RigidTransform as Tf
from diffusers import DDIMScheduler
from agent.dataset.sequence import ActionMode
from agent.model.networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn

ACTION_MODES = ('absolute', 'local_delta', 'global_delta', 'umi')


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
        img_size=128,
        num_diffusion_iters=100,
        num_inference_steps=10,
        norm_stats: dict | None = None,
        action_mode: ActionMode = 'local_delta',
        encoder_type='resnet',
        augment=True,
    ):
        super().__init__()
        # Architecture/config args; saved alongside the weights by save_checkpoint so
        # from_checkpoint can rebuild the policy without the caller knowing the dims.
        self.config = dict(
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            vision_feature_dim=vision_feature_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            img_size=img_size,
            num_diffusion_iters=num_diffusion_iters,
            action_mode=action_mode,
            encoder_type=encoder_type,
            augment=augment
        )
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.img_size = img_size
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

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',  # squared cosine works the best
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type='epsilon',  # network predicts noise
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Normalization stats as buffers: saved in state_dict, moved with .to(device).
        # Defaults are the identity transform ([-1, 1] range).
        self.register_buffer('action_min', -torch.ones(action_dim))
        self.register_buffer('action_max', torch.ones(action_dim))
        self.register_buffer('state_min', -torch.ones(state_dim))
        self.register_buffer('state_max', torch.ones(state_dim))

        # action_mode as a buffer so checkpoints remember how their actions are encoded
        self.register_buffer('_action_mode_idx', torch.tensor(ACTION_MODES.index(action_mode)))
        if norm_stats is not None:
            self.set_norm_stats(norm_stats)

    @property
    def action_mode(self) -> str:
        return ACTION_MODES[int(self._action_mode_idx)]

    @classmethod
    def from_checkpoint(cls, ckpt_path, device='cpu'):
        """
        Rebuild a policy entirely from a checkpoint: architecture config, weights,
        and normalization stats all come from the file.
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        assert 'config' in checkpoint, (
            f"Checkpoint {ckpt_path} has no 'config' entry; it predates config-saving. "
            "Construct DiffusionPolicy with explicit dims and use load_checkpoint instead.")
        policy = cls(**checkpoint['config'])
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Checkpoint] Loaded policy from {ckpt_path} (config: {checkpoint['config']})")
   
        return policy.to(device)

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
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.nets['noise_pred_net'](
                sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction).prev_sample

        return self.unnormalize_actions(naction)

    def integrate_actions(self, actions, curr_pose, curr_gripper_width):
        """
        Convert a predicted action chunk into absolute desired poses + gripper widths,
        inverting the dataset's action encoding (see StitchedSequenceDataset.pose_action):

            absolute:     [tx, ty, tz, rx, ry, rz];           des_i = a_i
            local_delta:  exp coords [rot, trans] of t0⁻¹tᵢ;  des_i = t0 * exp(a_i)
            global_delta: exp coords [rot, trans] of tᵢt0⁻¹;  des_i = exp(a_i) * t0

        Args:
            actions:            (horizon, action_dim) unnormalized actions from
                                predict_action; last dim is the gripper (+1=open, -1=closed).
            curr_pose:          (6,) current pose [tx, ty, tz, rx, ry, rz] (rotvec).
            curr_gripper_width: current physical gripper width, used as fallback when
                                width stats are unavailable.

        Returns:
            des_poses:  (horizon, 6) absolute poses [tx, ty, tz, rx, ry, rz].
            des_widths: (horizon,) desired gripper widths.
        """
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        curr_pose = np.asarray(curr_pose, dtype=float)
        pose_actions, g_actions = actions[:, :6], actions[:, -1]

        if self.action_mode == 'absolute':
            des_poses = pose_actions.copy()
        elif self.action_mode == "umi":
            curr_pos = curr_pose[:3]
            curr_euler = (R.from_rotvec(curr_pose[3:]) .as_euler("xyz") )

            des_poses = []
            for a in pose_actions:
                delta_pos = a[:3]
                delta_euler = a[3:]

                pos = curr_pos + delta_pos
                euler = curr_euler + delta_euler
                rot =  R.from_euler("xyz", euler).as_rotvec()

                des_poses.append(
                    np.concatenate([pos, rot])
                )

            des_poses = np.asarray(des_poses)
        else:
            t0 = Tf.from_components(curr_pose[:3], R.from_rotvec(curr_pose[3:]))
            if self.action_mode == 'local_delta':
                des_tfs = [t0 * Tf.from_exp_coords(a) for a in pose_actions]
            else:  # global_delta
                des_tfs = [Tf.from_exp_coords(a) * t0 for a in pose_actions]
            des_poses = np.array([
                np.concatenate([t.translation, t.rotation.as_rotvec()]) for t in des_tfs])

        # Map gripper action (-1 -> 1, 1 -> 0)
        des_gripper = np.where(g_actions > 0, 0, 1)
        return des_poses, des_gripper
