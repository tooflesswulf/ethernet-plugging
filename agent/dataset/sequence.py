
from PIL import Image
from tqdm import tqdm
import os, numpy as np
from collections import namedtuple
from einops import rearrange, repeat
import torch


Batch = namedtuple("Batch", "actions conditions")

def pose2actions(poses, grippers, traj_lengths):
    poses, grippers = poses[:sum(traj_lengths)], grippers[:sum(traj_lengths)]
    # compute delta actions between poses, add absolute gripper length
    actions = []
    offset = 0
    for traj_length in traj_lengths:
        start, end = offset, offset + traj_length
        ep_poses = poses[start:end]
        ep_poses_shifted = np.concatenate( [poses[start+1:end],poses[end-1:end]]  )
        # delta action with absolute gripper width as action
        ep_actions = ep_poses_shifted - ep_poses
        shifted_grippers = np.concatenate( [grippers[start+1:end], grippers[end-1:end]]  )[:, None]
        ep_actions = np.concatenate( [ep_actions, shifted_grippers], axis = -1 )
        actions = ep_actions if len(actions) == 0 else np.concatenate( [actions, ep_actions] )
        # increase counter 
        offset += traj_length
    assert len(actions) == len(poses)
    return actions

def get_images(dir_path, total_num_steps=None, img_size = 128):
    N = len(os.listdir(dir_path)) if total_num_steps is None else min(len(os.listdir(dir_path)), total_num_steps)
    images = [ np.array(Image.open(os.path.join(dir_path, f'{i}.png')).resize((img_size, img_size))) for i in tqdm(range(N),  desc="loading images to RAM") ]
    return np.array(images)

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to the range [-1, 1] using min/max scaling.
    If min == max, the array is returned unchanged.
    """
    min_val = arr.min()
    max_val = arr.max()

    if min_val == max_val:
        return arr.copy()

    return 2 * (arr - min_val) / (max_val - min_val) - 1

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
        device="cuda:0",
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
 
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset to device specified
        img_dir = os.path.join(dataset_path, 'images')
        state_path = os.path.join(dataset_path, 'states.npz')
        dataset = np.load(state_path, allow_pickle=False)  # only np arrays
        traj_lengths = dataset["traj_length"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        actions = pose2actions(dataset['actual_pose'],  dataset['gripper_width'], traj_lengths)
        states = np.concatenate([dataset['actual_pose'], dataset['gripper_width'][:, None] ], axis = -1)
        
        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(normalize(states[:total_num_steps])).float().to(device)
        )  # (total_num_steps, obs_dim)
        self.actions = (
            torch.from_numpy(normalize(actions[:total_num_steps])).float().to(device)
        )  # (total_num_steps, action_dim)
        
        self.images = torch.from_numpy(get_images(img_dir, total_num_steps)).to(
            device
        )  # (total_num_steps, H, W, C)

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end] # horizon x dim
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end, # cond_steps x dim
       
        conditions = {"state": states}

        images = self.images[(start - num_before_start) : end]
        images = torch.stack(
            [
                images[max(num_before_start - t, 0)]
                for t in reversed(range(self.img_cond_steps))
            ]
        ) # img_cond_steps x H x W x C
        conditions["rgb"] = rearrange(images, ' T H W C -> T C H W') / 255.0 - 0.5
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
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
