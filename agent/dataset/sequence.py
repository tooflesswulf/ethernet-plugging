
from collections import namedtuple
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
        transform=None,
        device="cuda:0",
    ):
        assert img_cond_steps <= cond_steps, 'consider using more cond_steps than img_cond_steps'
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.transform = transform

        self.max_n_episodes = max_n_episodes
        self.dataset_path = pathlib.Path(dataset_path)

        # Load dataset to device
        img_dir = self.dataset_path / 'images'
        state_path = self.dataset_path / 'states.npz'
        dataset = np.load(state_path, allow_pickle=False)  # only np arrays
        traj_lengths = dataset['traj_length'][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        states = np.c_[*[dataset[key] for key in obs_fields]]  # Concat along columns, (total_num_steps, obs_dim)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = torch.from_numpy(states[:total_num_steps]).float().to(device)  # (total_num_steps, obs_dim)
        self.images = torch.from_numpy(get_images(img_dir, total_num_steps)).to(device)  # (total_num_steps, H, W, C)

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start, traj_end = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        # actions = self.actions[start:end] # horizon x dim

        # Delta action:
        # s_t - s_0 (delta between current state and meta state), with absolute gripper
        _end = min(traj_end, end + 1)
        future_states = self.states[(start + 1): _end]
        actions = future_states - self.states[start]
        # fix last dimension with absolute gripper width
        gripper = future_states[:, -1]
        # if > 20, set 0, otherwise set 1
        # gripper = 1-(gripper > 20).float()
        actions[:, -1] = gripper
        if len(actions) < self.horizon_steps:
            padding = self.horizon_steps - len(actions)
            actions = torch.cat(
                [actions, actions[-1:].repeat(padding, 1)], dim=0
            )  # repeat last action if not enough future states

        actions = normalize(actions, min_val=self.action_min, max_val=self.action_max)
        # binary gripper
        m_close, m_open = actions[:, -1] <= 0, actions[:, -1] > 0
        actions[:, -1][m_close] = -1
        actions[:, -1][m_open] = 1
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end, # cond_steps x dim
        states = normalize(states, min_val=self.state_min, max_val=self.state_max)
        conditions = {"state": states}

        images = self.images[(start - num_before_start): end]
        images = torch.stack(
            [
                images[max(num_before_start - t, 0)]
                for t in reversed(range(self.img_cond_steps))
            ]
        )  # img_cond_steps x H x W x C

        conditions["rgb"] = rearrange(images, ' T H W C -> T C H W') / 255.0  # - 0.5
        batch = DataBatch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory

        Returns list[(start_index, num_before_start, traj_end_index)], where
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            traj_end = cur_traj_index + traj_length
            indices += [
                (i, i - cur_traj_index, traj_end) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    task = 'ethernet_unplug'
    dataset_dir = '/zfsauton/scratch/yiqiw2/100%/datasets'
    dataset_path = os.path.join(dataset_dir, task)
    dataset = StitchedSequenceDataset(dataset_path)

    for _ in dataset:
        break
