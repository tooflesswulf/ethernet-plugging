from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import pathlib
import h5py
import os

from util import dict2hdf5, hdf52dict


def ewma(x, alpha):
    """Exponentially weighted moving average."""
    ema = np.zeros_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema


def collate_metadata(meta_list):
    collated = {}
    for key in meta_list[0].keys():
        values = [meta[key] for meta in meta_list]
        if isinstance(values[0], (int, float, str)):
            collated[key] = values
        elif isinstance(values[0], dict):
            collated[key] = collate_metadata(values)
        else:
            collated[key] = np.stack(values)
    return collated


@dataclass
class EpisodeData:
    images: list
    poses: np.ndarray
    forces: np.ndarray
    gripper_widths: np.ndarray
    gripper_forces: np.ndarray
    meta: dict


def proc_h5(h5_path, framerate=10.0, alpha=0.03):
    with h5py.File(h5_path, 'r') as f:
        rt = np.array(f['robot_obs/time'])
        actual_pose = np.array(f['robot_obs/actual_pose'])
        actual_force = np.array(f['robot_obs/actual_force'])

        gt = np.array(f['gripper_obs/time'])
        gripper_width = np.array(f['gripper_obs/gripper_width'])
        gripper_force = np.array(f['gripper_obs/gripper_force'])

        it = np.array(f['camera_obs/time'])
        images = np.array(f['camera_obs/image_bgr'])

        meta = hdf52dict(f['metadata'])

    force_smoothed = ewma(actual_force, alpha=alpha)

    # sample at the specified framerate
    dt = 1.0 / framerate
    t0, tf = 0, max(rt[-1], gt[-1], it[-1])
    sample_times = np.arange(t0 + dt, tf, dt)

    imgs = []
    poses = []
    g_widths = []
    forces = []
    g_forces = []
    for t in sample_times:
        rt_idx = np.searchsorted(rt, t, side='right') - 1
        gt_idx = np.searchsorted(gt, t, side='right') - 1
        it_idx = np.searchsorted(it, t, side='right') - 1

        imgs.append(images[it_idx])
        poses.append(actual_pose[rt_idx])
        g_widths.append(gripper_width[gt_idx])
        forces.append(force_smoothed[rt_idx])
        g_forces.append(gripper_force[gt_idx])

    meta['length'] = len(imgs)
    return EpisodeData(
        images=imgs,
        poses=poses,
        forces=forces,
        gripper_widths=g_widths,
        gripper_forces=g_forces,
        meta=meta
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a set of `rawdata.h5` files to a dataset')
    parser.add_argument('--path', type=str,
                        default='/home/albertxu/data/ethernet_plug_v3',
                        help='Base dataset directory')
    parser.add_argument('--framerate', type=float, default=10.0, help='Framerate to sample the raw data at')
    parser.add_argument('--alpha', '-a', type=float, default=0.03, help='Smoothing factor for force EWMA')
    parser.add_argument('--h5_images', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to save images in the HDF5 file (can make it very large)')
    args = parser.parse_args()

    path = pathlib.Path(args.path)
    save_dir = path.parent / (path.stem + '_dataset')
    save_dir.mkdir(exist_ok=True)

    episodes: list[EpisodeData] = []
    for ep_str in sorted(os.listdir(path)):
        if not ep_str.startswith('episode'):
            continue
        episode_path = path / ep_str
        episode = proc_h5(path / ep_str / 'rawdata.h5', framerate=args.framerate)
        episodes.append(episode)

    if not args.h5_images:
        image_paths = []
        total = sum(ep.meta['length'] for ep in episodes)
        os.makedirs(save_dir / 'images', exist_ok=True)
        tq = tqdm(enumerate(episodes), total=total, desc='Saving images')
        for ep_idx, episode in tq:
            for img_idx, img in enumerate(episode.images):
                suffix = f'images/ep{ep_idx}_img{img_idx}.png'
                img_save_path = save_dir / suffix
                image_paths.append(suffix)
                Image.fromarray(img).save(save_dir / suffix)
                tq.update()

    meta = collate_metadata([ep.meta for ep in episodes])
    with h5py.File(save_dir / 'dataset.h5', 'w') as f:
        f.create_dataset('num_episodes', data=len(episodes))
        f.create_dataset('pose', data=np.concatenate([ep.poses for ep in episodes], axis=0))
        f.create_dataset('force', data=np.concatenate([ep.forces for ep in episodes], axis=0))
        f.create_dataset('gripper_width', data=np.concatenate([ep.gripper_widths for ep in episodes], axis=0))
        f.create_dataset('gripper_force', data=np.concatenate([ep.gripper_forces for ep in episodes], axis=0))
        dict2hdf5(f.create_group('metadata'), meta)
        f['metadata'].attrs['framerate'] = args.framerate

        if args.h5_images:
            img_chunk_shape = (1,) + episodes[0].images[0].shape
            ds = f.create_dataset('images', data=np.concatenate([ep.images for ep in episodes], axis=0),
                                  chunks=img_chunk_shape, compression='lzf')
            ds.attrs['stored_as'] = 'image'
        else:
            ds = f.create_dataset('images', data=image_paths, dtype=h5py.string_dtype())
            ds.attrs['stored_as'] = 'filepath'

    print(f'Saved dataset to {save_dir / "dataset.h5"}')
