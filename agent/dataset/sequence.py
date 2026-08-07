
from collections import namedtuple
from dataclasses import dataclass
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
GripperStats = namedtuple('GripperStats', ['grip_width_mm', 'grip_force_n', 'grip_speed_mmps', 'grip_pullback_mm'])
ActionMode = Literal['absolute', 'local_delta', 'global_delta', 'umi']

# def get_images(dir_path, total_num_steps=None, img_size=128):
#     N = len(os.listdir(dir_path)) if total_num_steps is None else min(len(os.listdir(dir_path)), total_num_steps)
#     images = []
#     for i in tqdm(range(N), desc="loading images to RAM"):
#         img = Image.open(os.path.join(dir_path, f'{i}.png')).resize((img_size, img_size))
#         images.append(np.array(img))
#     return np.array(images)

def get_images(dir_path, total_num_steps=None, img_size=128, num_workers=8):
    N = len(os.listdir(dir_path))
    if total_num_steps is not None:
        N = min(N, total_num_steps)

    def load_one(i):
        path = os.path.join(dir_path, f"{i}.png")
        with Image.open(path) as img:
            img = img.resize((img_size, img_size))
            return np.asarray(img)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(
            tqdm(
                executor.map(load_one, range(N)),
                total=N,
                desc="loading images to RAM",
            )
        )

    return np.stack(images)

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
        dataset_paths,
        horizon_steps=16,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        obs_fields=['pose', 'gripper_width'],
        action_mode: ActionMode = 'local_delta',
        predict_done=True,
        end_signal_steps=None,
        transform=None,
        img_size=96,
        device="cuda:0",
    ):
        assert img_cond_steps <= cond_steps, 'consider using more cond_steps than img_cond_steps'
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.action_mode = action_mode
        self.transform = transform
        self.img_size = img_size
        self.max_n_episodes = max_n_episodes
        self.g_thr = 18 # (np.amax(self.g_widths) + np.amin(self.g_widths)) / 2  # threshold for binary gripper action
        self.indices = None; traj_start = 0
        self.poses, self.g_widths, self.obs, self.images = None, None, None, None
        for dataset_path in dataset_paths:
            dataset_path = pathlib.Path(dataset_path)

            # Load dataset to device
            img_dir = dataset_path / 'images' ; state_path = dataset_path / 'states.npz'
            dataset = np.load(state_path, allow_pickle=False)  # only np arrays
            traj_lengths = dataset['traj_length'][:max_n_episodes]  # 1-D array
            total_num_steps = np.sum(traj_lengths)
            obs = np.c_[*[dataset[key] for key in obs_fields]]  # Concat along columns, (total_num_steps, obs_dim)

            # Set up indices for sampling
            indices = self.make_indices(traj_lengths, horizon_steps, traj_start=traj_start)
            if self.indices is None:
                self.indices = indices 
            else:
                self.indices = np.concatenate([self.indices, indices])
            traj_start += total_num_steps

            # Extract states and actions up to max_n_episodes
            poses = dataset['pose'][:total_num_steps]  # (N, 6)
            g_widths = dataset['gripper_width'][:total_num_steps]  # (N,)
            obs = torch.from_numpy(obs[:total_num_steps]).float().to(device)  # (N, obs_dim)
            images = torch.from_numpy(get_images(img_dir, total_num_steps, img_size=img_size)).to(device)  # (N, H, W, C)
            if self.poses is None:
                self.poses, self.g_widths, self.obs, self.images = poses, g_widths, obs, images 
            else:
                self.poses = np.concatenate([self.poses, poses])
                self.g_widths = np.concatenate([self.g_widths, g_widths])
                self.obs = torch.cat([self.obs, obs])
                self.images = torch.cat([self.images, images]) 
               

        self.obs_dim = self.obs.shape[-1]
        self._precompute_actions()  # precompute all actions for faster sampling during training
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
        images = torch.stack([self.images[max(start - t, ep_start)]
                             for t in reversed(range(self.img_cond_steps))])  # img_cond_steps x H x W x C
        conditions = {'state': obs, 'rgb': rearrange(images, ' T H W C -> T C H W') / 255.0}

        batch = DataBatch(self.actions[idx], conditions)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def make_indices(self, traj_lengths, horizon_steps, traj_start = 0):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint and its bounds within the same trajectory.
        Returns list[(start_index, traj_start_index, traj_end_index)]
        """
        indices = []
        
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

    def done_action(self, start, end, ep_end):
        """
        Binary end-of-episode signal for the chunk covering absolute timesteps
        [start, end). 1=task complete (within the final `end_signal_steps` frames
        of the episode), -1=not done.
        """
        abs_t = np.arange(start, end)
        done = abs_t >= (ep_end - self.end_signal_steps)
        return 2 * done.astype(int).reshape(-1, 1) - 1

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
        eulers = np.array([ R.from_rotvec(rxyz).as_euler("xyz") for rxyz in poses[:, 3:] ])
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

class FailureDetectDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        label_path,
        horizon_steps=8,
        cond_steps=1,
        force_cond_steps=8,
        img_cond_steps=1,
        obs_fields=['pose', 'gripper_width', 'force'],       
        transform=None,
        device="cuda:0",
    ):
        """
        Return (obs=(images, ....), label=0/1) where 0 is success, 1 is failled
        """
        self.dataset_path = dataset_path
        df = pd.read_csv(label_path)
        episode_ids, fail_idx = df.episode.values, df.idx.values
        self.ep_states = [
            np.load(os.path.join(dataset_path, f"episode{int(episode_id):06d}", 'states.npz'))
            for episode_id in episode_ids
        ]
        self.ep_lengths = [ len(ep_states['pose']) for ep_states in self.ep_states]
        self.horizon_steps = horizon_steps
        self.img_cond_steps = img_cond_steps; self.cond_steps = cond_steps; self.force_cond_steps = force_cond_steps
        self.make_indices(episode_ids, fail_idx)
        self.obs_fields = obs_fields
        self.device = device

    def make_indices(self, episode_ids, fail_idx):
        self.indices = [] # list of item: [image_dir, state_idx, start, end, fail/success]
        for i, episode_id in enumerate(episode_ids):
            image_dir = os.path.join(self.dataset_path, f"episode{int(episode_id):06d}", "images")
            state_idx = i 
            ep_fail_idx = fail_idx[i]
            for j in range(self.ep_lengths[i]-self.horizon_steps):
                start = max(0, j-self.horizon_steps)
                label = 0 if ep_fail_idx == -1 else int( j > ep_fail_idx) # if j > ep_fail_idx, 1 else 0
                self.indices.append([image_dir, state_idx, start, j, label])
    def get_image(self, image_path):
        return np.array( Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE)) ) 
    
    def __getitem__(self, index):
        image_dir, state_idx, start, end, label = self.indices[index]
        image_indices = np.linspace(start, end, self.img_cond_steps, dtype=int) if self.img_cond_steps > 1 else [end]
        force_indices = np.linspace(start, end, self.force_cond_steps, dtype=int) if self.force_cond_steps > 1 else [end]
        cond_indices = np.linspace(start, end, self.cond_steps, dtype=int) if self.cond_steps > 1 else [end]
        np_images = np.array(
            [ self.get_image(os.path.join(image_dir, f'{i:06d}.png')) for i in image_indices] ) # TxHxWxC
        data_dict = {'image': torch.from_numpy(np_images).float().to(self.device)}
        states = self.ep_states[state_idx]
        for k in self.obs_fields:
            obs = states[k]; obs_index = cond_indices if 'force' not in k else force_indices
            obs = np.array([obs[i] for i in obs_index])
            data_dict[k] = torch.from_numpy(obs).float().to(self.device)
        return data_dict, label
    
    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    task = 'ethernet_unplug'
    dataset_dir = '/zfsauton/scratch/yiqiw2/100%/datasets'
    dataset_path = os.path.join(dataset_dir, task)
    dataset = StitchedSequenceDataset(dataset_path)

    for _ in dataset:
        break