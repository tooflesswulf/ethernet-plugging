from scipy.spatial.transform import Rotation as R, RigidTransform as Tf
import torchvision.transforms.functional as F
from collections import namedtuple
from typing import Literal
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
import torch
import pathlib
import h5py

IMAGE_SIZE = 128
DataBatch = namedtuple('DataBatch', ['actions', 'conditions'])
ActionMode = Literal['absolute', 'local_delta', 'global_delta', 'umi']


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    From: https://github.com/irom-princeton/dppo
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.
    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=16,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        obs_fields=['pose', 'gripper_width'],
        action_mode: ActionMode = 'local_delta',
        transform=None,
        device="cuda:0",
    ):
        assert img_cond_steps <= cond_steps, 'consider using more cond_steps than img_cond_steps'
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.action_mode = action_mode
        self.transform = transform

        self.max_n_episodes = max_n_episodes
        self.dataset_path = pathlib.Path(dataset_path)

        # Load dataset to device
        with h5py.File(self.dataset_path / 'dataset.h5', 'r') as f:
            traj_lengths = f['metadata/length'][:max_n_episodes]  # 1-D array
            total_num_steps = np.sum(traj_lengths)

            # Observations
            all_obs = []
            for key in obs_fields:
                if key.startswith('metadata/'):
                    # Metadata fields need to be expanded to per-timestep values for easier indexing later (e.g. rng)
                    meta_vals = np.array(f[key][:max_n_episodes])
                    vals_rep = [np.repeat(val[None], traj_len, axis=0)
                                for val, traj_len in zip(meta_vals, traj_lengths)]
                    all_obs.append(np.concatenate(vals_rep, axis=0))
                else:
                    all_obs.append(f[key][:total_num_steps])
            all_obs = np.c_[*all_obs]

            # Actions
            poses = np.array(f['pose'][:total_num_steps])  # (N, 6)
            g_widths = np.array(f['gripper_width'][:total_num_steps])  # (N,)

            if f['images'].attrs['stored_as'] == 'image':
                self.h5_image = True
            elif f['images'].attrs['stored_as'] == 'filepath':
                self.h5_image = False

        # Store dataset in memory for fast sampling during training
        self.indices = self.make_indices(traj_lengths, horizon_steps)
        self.obs = all_obs  # (N, obs_dim)
        self.h5 = None
        self._precompute_actions(poses, g_widths)

        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.actions.shape[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        if self.h5 is None:
            self.h5 = h5py.File(self.dataset_path / 'dataset.h5', 'r')

        start, ep_start, ep_end = self.indices[idx]
        end = min(start + self.horizon_steps, ep_end)
        if end > ep_end:
            # Shouldn't happen because make_indices should ensure we only sample valid start indices, but just in case
            raise RuntimeError(f"Error: end index {end} exceeds episode end {ep_end}.")

        # Conditioning observations: current and history states + images
        obs_np = np.array([self.obs[max(start - t, ep_start)]
                          for t in reversed(range(self.cond_steps))])  # more recent is at the end, # cond_steps x dim
        images = np.array([self.getimage(max(start - t, ep_start))
                           for t in reversed(range(self.img_cond_steps))])
        conditions = {'state': obs_np, 'rgb': images / 255.0}

        batch = DataBatch(self.actions[idx], conditions)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def getimage(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.dataset_path / 'dataset.h5', 'r')
        im = self.h5['images'][idx]
        if self.h5_image:
            img = Image.fromarray(im).resize((IMAGE_SIZE, IMAGE_SIZE))
        else:
            im_path = self.dataset_path / im.decode()
            img = Image.open(im_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        im_np = np.array(img).transpose(2, 0, 1)  # (C, H, W)
        return im_np

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint and its bounds within the same trajectory.
        Returns list[(start_index, traj_start_index, traj_end_index)]
        """
        indices = []
        traj_start = 0
        for traj_length in traj_lengths:
            max_start = traj_start + traj_length - horizon_steps
            traj_end = traj_start + traj_length
            indices += [(i, traj_start, traj_end) for i in range(traj_start, max_start + 1)]
            traj_start += traj_length
        return np.array(indices)

    def _precompute_actions(self, poses, g_widths):
        g_thr = (np.amax(g_widths) + np.amin(g_widths)) / 2  # threshold for binary gripper action

        actions = []
        for i in tqdm(range(len(self)), desc='precomputing actions'):
            start, ep_start, ep_end = self.indices[i]
            end = min(start + self.horizon_steps, ep_end)
            if end > ep_end:
                # TODO: replication pad if end out of ep_end
                raise RuntimeError(f"Error: end index {end} exceeds episode end {ep_end}.")

            g_width = g_widths[start:end]
            pose = poses[start:end]

            g_action = self.gripper_action(g_width, threshold=g_thr)
            pose_action = self.pose_action(pose)
            actions.append(np.c_[pose_action, g_action])
        self.actions = np.array(actions)
        return self.actions

    def gripper_action(self, g_widths, threshold=20):
        """
        Binary gripper predictions. 1=open, -1=closed
        """
        return 2 * (g_widths > threshold).astype(int).reshape(-1, 1) - 1

    def _pose_action_absolute(self, poses):
        # Returns (N, 6): [tx, ty, tz, rx, ry, rz]
        return poses

    def _pose_action_local_delta(self, poses):
        # Returns (N, 6): [rx, ry, rz, tx, ty, tz] (SE(3) exp coords, NOT the same ordering as absolute)
        transforms = [Tf.from_components(pos[:3], R.from_rotvec(pos[3:])) for pos in poses]
        t0 = transforms[0]
        deltas = [t0.inv() * t for t in transforms]
        return np.array([delta.as_exp_coords() for delta in deltas])

    def _pose_action_umi(self, poses):
        # Returns (N, 6): delta between META timestep and current timetstep given absolute xyz and Euler angle
        delta_xyz = poses[1:, :3] - poses[:1, :3]
        eulers = np.array([R.from_rotvec(rxyz).as_euler("xyz") for rxyz in poses[:, 3:]])
        # delta_rotations = np.array( [ (r2*rotations[0].inv()).as_rotvec() for r2 in rotations[1:] ] )
        delta_eulers = eulers[1:] - eulers[:1]
        # wrap to [-pi, pi]
        delta_euler = (delta_eulers + np.pi) % (2 * np.pi) - np.pi
        delta_umi = np.concatenate([delta_xyz, delta_euler], -1)
        return np.concatenate([delta_umi, delta_umi[-1:]])  # poor decision here, pad by 1 by repeating last one.

    def _pose_action_global_delta(self, poses):
        # Returns (N, 6): [rx, ry, rz, tx, ty, tz] (SE(3) exp coords, NOT the same ordering as absolute)
        transforms = [Tf.from_components(pos[:3], R.from_rotvec(pos[3:])) for pos in poses]
        t0 = transforms[0]
        deltas = [t * t0.inv() for t in transforms]
        return np.array([delta.as_exp_coords() for delta in deltas])

    def pose_action(self, poses):
        if self.action_mode == 'absolute':
            return self._pose_action_absolute(poses)
        elif self.action_mode == 'local_delta':
            return self._pose_action_local_delta(poses)
        elif self.action_mode == 'global_delta':
            return self._pose_action_global_delta(poses)
        elif self.action_mode == 'umi':
            return self._pose_action_umi(poses)
        else:
            raise ValueError(f"Invalid action_mode: {self.action_mode}")


if __name__ == '__main__':
    dataset_dir = '/home/albertxu/data/ethernet_plug_v3_dataset'
    dataset = StitchedSequenceDataset(dataset_dir, obs_fields=['pose', 'gripper_width', 'metadata/rng'])

    for _ in dataset:
        print(_.actions.shape, _.conditions['state'].shape, _.conditions['rgb'].shape)
        break
