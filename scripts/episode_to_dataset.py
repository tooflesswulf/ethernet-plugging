import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import shutil


def get_episode(episode_dir):
    states_path = os.path.join(episode_dir, 'states.npz')
    states = np.load(states_path)
    poses, g_widths, g_forces, forces = states['pose'], states['gripper_width'], states['gripper_force'], states['force']
    states_N = len(poses)
    image_dir = os.path.join(episode_dir, 'images')
    image_N = len(os.listdir(image_dir))
    if image_N != states_N:
        print(episode_dir)
    image_paths = [os.path.join(image_dir, f"{i:06d}.png") for i in range(image_N)]
    return image_paths, poses, g_widths, g_forces, forces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--path', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_unplug_red',
                        help='Base dataset directory')
    args = parser.parse_args()

    path = args.path
    save_dir = path.rstrip('/') + '_dataset' if path.endswith('/') else path + '_dataset'

    save_img_dir, save_state_path = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'states.npz')

    total_images, total_poses, total_widths, total_g_forces, total_forces, lens = [], [], [], [], [], []

    for ep_str in sorted(os.listdir(path)):
        if not ep_str.startswith('episode'):
            continue
        episode_path = os.path.join(path, ep_str)
        image_paths, poses, widths, g_forces, forces = get_episode(episode_path)
        total_images += image_paths
        total_poses = poses if len(total_poses) == 0 else np.concatenate([total_poses, poses], 0)
        total_widths = widths if len(total_widths) == 0 else np.concatenate([total_widths, widths], 0)
        total_g_forces = g_forces if len(total_g_forces) == 0 else np.concatenate([total_g_forces, g_forces])
        total_forces = forces if len(total_forces) == 0 else np.concatenate([total_forces, forces])
        lens.append(len(image_paths))

    print(len(total_images), total_poses.shape, total_widths.shape, total_g_forces.shape, total_forces.shape, sum(lens))
    assert len(total_images) == len(total_poses)

    # save states
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        save_state_path,
        pose=total_poses,
        force=total_forces,
        gripper_width=total_widths,
        gripper_force=total_g_forces,
        traj_length=np.array(lens)
    )
    # save images
    os.makedirs(save_img_dir, exist_ok=True)
    for i, img_path in enumerate(tqdm(total_images)):
        save_img_path = os.path.join(save_img_dir, f'{i}.png')
        shutil.copy(img_path, save_img_path)
