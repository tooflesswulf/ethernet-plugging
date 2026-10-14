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
    # Impedance controller logs (FDCC, env.py ControlLog); None for stiff-controller data.
    # Gains are diagonals in base-frame axes (see base_frame_gains).
    target_poses: np.ndarray | None = None
    stiffness: np.ndarray | None = None
    damping: np.ndarray | None = None
    mass: np.ndarray | None = None


def base_frame_gains(gains, frame_base):
    """
    Logged gains are diagonal on the axes of fdcc.Impedance.frame. Gains on 'tool' axes are
    only allowed to differ from 'base' when they are not isotropic on each half, and
    set_gains() only permits a frame change through isotropic gains -- so check that every
    tool-frame row is isotropic, which makes the base-frame diagonal exact.
    """
    tool = ~frame_base
    aniso = (np.ptp(gains[:, :3], axis=1) > 0) | (np.ptp(gains[:, 3:], axis=1) > 0)
    assert not np.any(tool & aniso), 'anisotropic gains on tool axes have no base-frame diagonal'
    assert np.all(gains > 0), 'gains must be positive (they are learned in log space)'
    return gains


def proc_h5(h5_path, framerate=10.0, alpha=0.03):
    with h5py.File(h5_path, 'r') as f:
        rt = np.array(f['robot_obs/time'])
        actual_pose = np.array(f['robot_obs/actual_pose'])
        actual_force = np.array(f['robot_obs/actual_force'])
        # the env's own EWMA of actual_force: exactly the live 'filtered_force' obs at eval
        filtered_force = np.array(f['robot_obs/filtered_force']) if 'robot_obs/filtered_force' in f else None

        control = None
        if 'control' in f:
            frame_base = np.array(f['control/frame_base'])
            control = {
                'time': np.array(f['control/time']),
                'target_poses': np.array(f['control/target']),
                'stiffness': base_frame_gains(np.array(f['control/K']), frame_base),
                'damping': base_frame_gains(np.array(f['control/D']), frame_base),
                'mass': base_frame_gains(np.array(f['control/M']), frame_base),
            }

        gt = np.array(f['gripper_obs/time'])
        gripper_width = np.array(f['gripper_obs/gripper_width'])
        gripper_force = np.array(f['gripper_obs/gripper_force'])

        it = np.array(f['camera_obs/time'])
        images = np.array(f['camera_obs/image_bgr'])[..., ::-1]  # convert BGR to RGB

        if 'metadata' in f:
            meta = hdf52dict(f['metadata'])
        else:
            meta = {}

    force_smoothed = filtered_force if filtered_force is not None else ewma(actual_force, alpha=alpha)

    # sample at the specified framerate
    dt = 1.0 / framerate
    t0, tf = 0, max(rt[-1], gt[-1], it[-1])
    sample_times = np.arange(t0 + dt, tf, dt)

    imgs = []
    poses = []
    g_widths = []
    forces = []
    g_forces = []
    ctrl = {k: [] for k in ('target_poses', 'stiffness', 'damping', 'mass')} if control else None
    for t in sample_times:
        rt_idx = np.searchsorted(rt, t, side='right') - 1
        gt_idx = np.searchsorted(gt, t, side='right') - 1
        it_idx = np.searchsorted(it, t, side='right') - 1

        imgs.append(images[it_idx])
        poses.append(actual_pose[rt_idx])
        g_widths.append(gripper_width[gt_idx])
        forces.append(force_smoothed[rt_idx])
        g_forces.append(gripper_force[gt_idx])
        if control is not None:
            ct_idx = np.searchsorted(control['time'], t, side='right') - 1
            for k in ctrl:
                ctrl[k].append(control[k][ct_idx])

    meta['length'] = len(imgs)
    return EpisodeData(
        images=imgs,
        poses=poses,
        forces=forces,
        gripper_widths=g_widths,
        gripper_forces=g_forces,
        meta=meta,
        **({k: np.array(v) for k, v in ctrl.items()} if ctrl else {}),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a set of `rawdata.h5` files to a dataset')
    parser.add_argument('--path', type=str,
                        default='/home/albertxu/data/ethernet_plug_v3',
                        help='Base dataset directory')
    parser.add_argument('--framerate', type=float, default=20.0, help='Framerate to sample the raw data at')
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
        # Impedance fields (see EpisodeData), only when every episode has them
        for field, key in (('target_poses', 'target_pose'), ('stiffness', 'stiffness'),
                           ('damping', 'damping'), ('mass', 'mass')):
            if all(getattr(ep, field) is not None for ep in episodes):
                f.create_dataset(key, data=np.concatenate([getattr(ep, field) for ep in episodes], axis=0))
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
