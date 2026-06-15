
from collections import namedtuple
from typing import Literal
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, RigidTransform as Tf
from einops import rearrange, repeat
import numpy as np
import torch
import pathlib
import os
# from agent.utils.utils import get_chunk_actions

DataBatch = namedtuple('DataBatch', ['actions', 'conditions'])
ActionMode = Literal['absolute', 'local_delta', 'global_delta']


def get_images(dir_path, total_num_steps=None, img_size=128):
    N = len(os.listdir(dir_path)) if total_num_steps is None else min(len(os.listdir(dir_path)), total_num_steps)
    images = []
    for i in tqdm(range(N), desc="loading images to RAM"):
        img = Image.open(os.path.join(dir_path, f'{i}.png')).resize((img_size, img_size))
        images.append(np.array(img))
    return np.array(images)


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
        img_dir = self.dataset_path / 'images'
        state_path = self.dataset_path / 'states.npz'
        dataset = np.load(state_path, allow_pickle=False)  # only np arrays
        traj_lengths = dataset['traj_length'][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        obs = np.c_[*[dataset[key] for key in obs_fields]]  # Concat along columns, (total_num_steps, obs_dim)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.poses = dataset['pose'][:total_num_steps]  # (N, 6)
        self.g_widths = dataset['gripper_width'][:total_num_steps]  # (N,)
        self.obs = torch.from_numpy(obs[:total_num_steps]).float().to(device)  # (N, obs_dim)
        self.images = torch.from_numpy(get_images(img_dir, total_num_steps)).to(device)  # (N, H, W, C)
        self.obs_dim = self.obs.shape[1]

        self.g_thr = (np.amax(self.g_widths) + np.amin(self.g_widths)) / 2  # threshold for binary gripper action
        self._precompute_actions()  # precompute all actions for faster sampling during training
        self.act_dim = self.actions.shape[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, ep_start, ep_end = self.indices[idx]
        end = min(start + self.horizon_steps, ep_end)
        if end > ep_end:
            # Shouldn't happen because make_indices should ensure we only sample valid start indices, but just in case
            raise RuntimeError(f"Error: end index {end} exceeds episode end {ep_end}.")

        # Conditioning observations: current and history states + images
        obs = torch.stack([self.obs[max(start - t, ep_start)]
                          for t in reversed(range(self.cond_steps))])  # more recent is at the end, # cond_steps x dim
        images = torch.stack([self.images[max(start - t, ep_start)]
                             for t in reversed(range(self.img_cond_steps))])  # img_cond_steps x H x W x C
        conditions = {'state': obs, 'rgb': rearrange(images, ' T H W C -> T C H W') / 255.0}

        batch = DataBatch(self.actions[idx], conditions)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

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

    def _precompute_actions(self):
        actions = []
        for i in tqdm(range(len(self)), desc='precomputing actions'):
            start, ep_start, ep_end = self.indices[i]
            end = min(start + self.horizon_steps, ep_end)
            if end > ep_end:
                # TODO: replication pad if end out of ep_end
                raise RuntimeError(f"Error: end index {end} exceeds episode end {ep_end}.")

            g_width = self.g_widths[start:end]
            poses = self.poses[start:end]

            g_action = self.gripper_action(g_width, threshold=self.g_thr)
            pose_action = self.pose_action(poses)
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
        delta_xyz = poses[1:, :3][1:] - poses[:1, :3]; rotations = [ R.from_rotvec(rxyz) for rxyz in poses[:, 3:] ]
        delta_rotations = np.array( [ (r2*rotations[0].inv()).as_rotvec() for r2 in rotations[1:] ] )
        delta_umi = np.concatenate([delta_xyz, delta_rotations], -1)
        return np.concatenate( [delta_umi, delta_umi[-1:]] ) # poor decision here, pad by 1 by repeating last one.

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
    task = 'ethernet_unplug'
    dataset_dir = '/zfsauton/scratch/yiqiw2/100%/datasets'
    dataset_path = os.path.join(dataset_dir, task)
    dataset = StitchedSequenceDataset(dataset_path)

    for _ in dataset:
        break
