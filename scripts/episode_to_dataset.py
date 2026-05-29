import os
import numpy as np 
from PIL import Image

def get_episode(episode_dir):
    states_path = os.path.join(episode_dir, 'states.npz')
    states = np.load(states_path)
    poses, g_widths, g_forces = states['actual_pose'], states['gripper_width'], states['gripper_force']
    states_N = len(poses)
    image_dir = os.path.join(episode_dir, 'images')
    image_N = len(os.listdir(image_dir))
    if image_N != states_N:
        print(episode_dir)
    image_paths = [ os.path.join( image_dir, f"{i}.png") for i in range(image_N)]
    return image_paths, poses, g_widths, g_forces
    


dir = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset'
task = 'ethernet_unplug'
episodes = 32
for ep_id in range(episodes):
    episode_path = os.path.join( dir, task, str(ep_id))
    image_paths, poses, g_widths, g_forces = get_episode( episode_path)