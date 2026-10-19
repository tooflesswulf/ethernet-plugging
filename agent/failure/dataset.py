from PIL import Image
import numpy as np
import pandas as pd
import torch
import os

from agent.dataset.sequence import IMAGE_SIZE


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
    # dataset_dir = '/home/albertxu/data/ethernet_plug_v3_dataset'
    # dataset = StitchedSequenceDataset(dataset_dir, obs_fields=['pose', 'gripper_width', 'metadata/rng'])

    # for _ in dataset:
    #     print(_.actions.shape, _.conditions['state'].shape, _.conditions['rgb'].shape)
    #     break
    dataset_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc'
    label_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc/label.csv'
    dataset = FailureDetectDataset(dataset_path, label_path)

    from torch.utils.data import  DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
    )
    for data, label in dataloader:
        print(label.shape)
        for k,v in data.items():
            print(k, v.shape)

        break
