"""
Aggregate images from several raw datasets (directories of `episodeXXXXXX/rawdata.h5`
files, as written by `env.py`) into a single image-classification dataset labeled with
whether the cable is held between the gripper fingers.

Two kinds of source datasets are expected:
  * successful demonstrations (`--success`): the cable is assumed to be in the
    fingers exactly when the gripper is closed (`gripper_width < threshold`);
  * empty demonstrations (`--empty`): the gripper never holds the cable, so
    every frame is labeled 0 regardless of the gripper width.

Raw data is NOT time-aligned: camera, gripper and robot observations are logged by
separate threads at their own rates. The camera stream defines the timeline here (one
sample per image, no resampling to a fixed framerate), and every other signal is
zero-order held onto the camera timestamps -- for each image at time t, the most recent
gripper/robot sample at or before t is used, which is the same rule
`rawdata_to_dataset.py` uses. `gripper_dt` stores how stale that gripper sample was, so
frames logged across a gap can be dropped downstream.

As in `concat_datasets.py`, images are linked into the output with HDF5 virtual datasets
(one mapping per episode), so no image data is copied to disk -- only labels, the small
resampled per-frame signals, and metadata are written. The output is therefore NOT
standalone: the source episode directories must stay where they are for it to stay
readable.

Note that raw images are stored BGR (`camera_obs/image_bgr`) and a virtual dataset cannot
reorder channels, so the output images stay BGR; this is recorded in
`images.attrs['color_order']`.

Example:
    python scripts/build_grasp_dataset.py \\
        --success /data/ethernet_plug_v3 \\
        --success /data/ethernet_plug_v4 0:50 \\
        --empty /data/ethernet_plug_empty \\
        --output /data/grasp_cls_dataset
"""
from dataclasses import dataclass, field
import argparse
import pathlib
import h5py
import numpy as np

try:
    from concat_datasets import parse_episode_slice, resolve_episode_range
except ImportError:  # when run as `python -m scripts.build_grasp_dataset` from the repo root
    from scripts.concat_datasets import parse_episode_slice, resolve_episode_range

# Per-frame signals resampled onto the camera timestamps and copied into the output.
# These are tiny next to the images, so they are stored for real rather than linked.
# name in output -> (group, dataset) in rawdata.h5
RESAMPLED_FIELDS = {
    'pose': ('robot_obs', 'actual_pose'),
    'force': ('robot_obs', 'actual_force'),
    'gripper_width': ('gripper_obs', 'gripper_width'),
    'gripper_force': ('gripper_obs', 'gripper_force'),
}
IMAGE_FIELD = 'camera_obs/image_bgr'


def ewma(x, alpha):
    """Exponentially weighted moving average (same smoothing as rawdata_to_dataset.py)."""
    ema = np.zeros_like(x, dtype=np.float64)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema


def zero_order_hold(src_times, src_values, query_times):
    """Sample `src_values` at `query_times`, holding the most recent value at or before
    each query time. Returns (values, dt), where dt is how old the held sample was.

    Queries before the first source sample hold that first sample instead (dt < 0).
    """
    idx = np.searchsorted(src_times, query_times, side='right') - 1
    idx = np.clip(idx, 0, len(src_times) - 1)
    return src_values[idx], query_times - src_times[idx]


@dataclass
class EpisodeInfo:
    h5_path: pathlib.Path
    n_frames: int
    img_shape: tuple          # full shape of camera_obs/image_bgr, including the time axis
    img_dtype: np.dtype
    times: np.ndarray         # camera timestamps, (n_frames,)
    fields: dict              # resampled RESAMPLED_FIELDS, each (n_frames, ...)
    gripper_dt: np.ndarray    # staleness of the held gripper sample, (n_frames,)
    meta: dict


@dataclass
class SourceInfo:
    dataset_dir: pathlib.Path
    is_empty: bool
    start_ep: int
    stop_ep: int
    episodes: list = field(default_factory=list)

    @property
    def n_episodes(self):
        return len(self.episodes)

    @property
    def n_steps(self):
        return sum(ep.n_frames for ep in self.episodes)

    @property
    def gripper_width(self):
        if not self.episodes:
            return np.zeros(0)
        return np.concatenate([ep.fields['gripper_width'] for ep in self.episodes])


def inspect_episode(h5_path: pathlib.Path, force_alpha: float) -> EpisodeInfo:
    with h5py.File(h5_path, 'r') as f:
        img_shape = f[IMAGE_FIELD].shape
        img_dtype = f[IMAGE_FIELD].dtype
        cam_times = np.asarray(f['camera_obs/time']).reshape(-1)

        fields, gripper_dt = {}, None
        for name, (group, dset) in RESAMPLED_FIELDS.items():
            src_times = np.asarray(f[f'{group}/time']).reshape(-1)
            src_values = np.asarray(f[f'{group}/{dset}'])
            if name == 'force' and force_alpha is not None and len(src_values) > 0:
                src_values = ewma(src_values, alpha=force_alpha)
            values, dt = zero_order_hold(src_times, src_values, cam_times)
            fields[name] = values
            if group == 'gripper_obs':
                gripper_dt = dt

        meta = read_group(f['metadata']) if 'metadata' in f else {}

    if img_shape[0] != len(cam_times):
        raise ValueError(f'{h5_path}: {img_shape[0]} images but {len(cam_times)} camera timestamps')
    return EpisodeInfo(h5_path, img_shape[0], img_shape, img_dtype, cam_times, fields, gripper_dt, meta)


def read_group(h5group: h5py.Group) -> dict:
    data = {}
    for key, item in h5group.items():
        data[key] = read_group(item) if isinstance(item, h5py.Group) else item[()]
    return data


def find_episodes(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    """Episode directories, in the order `rawdata_to_dataset.py` would process them."""
    return [p / 'rawdata.h5' for p in sorted(dataset_dir.iterdir())
            if p.is_dir() and p.name.startswith('episode')]


def inspect_source(dataset_dir: str, ep_slice: slice, is_empty: bool, force_alpha: float) -> SourceInfo:
    dataset_dir = pathlib.Path(dataset_dir).resolve()
    ep_paths = find_episodes(dataset_dir)
    if not ep_paths:
        raise ValueError(f'{dataset_dir}: no `episode*/` directories found')

    start_ep, stop_ep = resolve_episode_range(ep_slice, len(ep_paths))
    episodes = []
    for h5_path in ep_paths[start_ep:stop_ep]:
        if not h5_path.exists():
            print(f'Warning: skipping {h5_path.parent.name} -- no rawdata.h5 (episode never finished saving?)')
            continue
        episode = inspect_episode(h5_path, force_alpha)
        if episode.n_frames == 0:
            print(f'Warning: skipping {h5_path.parent.name} -- no camera frames')
            continue
        episodes.append(episode)

    return SourceInfo(dataset_dir, is_empty, start_ep, stop_ep, episodes)


def check_consistent(sources: list[SourceInfo]):
    episodes = [ep for s in sources for ep in s.episodes]
    if not episodes:
        raise ValueError('No usable episodes found in any source')
    first = episodes[0]
    for ep in episodes[1:]:
        if ep.img_shape[1:] != first.img_shape[1:]:
            raise ValueError(f'image shape mismatch: {ep.h5_path} has {ep.img_shape[1:]}, '
                             f'expected {first.img_shape[1:]} (from {first.h5_path})')
        if ep.img_dtype != first.img_dtype:
            raise ValueError(f'image dtype mismatch: {ep.h5_path} has {ep.img_dtype}, '
                             f'expected {first.img_dtype} (from {first.h5_path})')


def auto_threshold(sources: list[SourceInfo]) -> float:
    """Midpoint between the widest and narrowest gripper width seen in the successful
    demos, i.e. halfway between "fully open" and "clamped on cable".

    Same rule as `StitchedSequenceDataset._precompute_actions`.
    """
    widths = [s.gripper_width for s in sources if not s.is_empty]
    widths = np.concatenate(widths) if widths else np.zeros(0)
    if widths.size == 0:
        raise ValueError('Cannot infer a gripper threshold without any --success sources; pass --gripper-thr')
    return float((widths.max() + widths.min()) / 2)


def build_labels(sources: list[SourceInfo], threshold: float) -> np.ndarray:
    """1 = cable between the fingers, 0 = not."""
    labels = []
    for s in sources:
        if s.is_empty:
            labels.append(np.zeros(s.n_steps, dtype=np.uint8))
        else:
            labels.append((s.gripper_width < threshold).astype(np.uint8))
    return np.concatenate(labels) if labels else np.zeros(0, dtype=np.uint8)


def build_images_vds(out_f: h5py.File, episodes: list[EpisodeInfo]):
    dtype = episodes[0].img_dtype
    trailing_shape = episodes[0].img_shape[1:]
    total = sum(ep.n_frames for ep in episodes)

    layout = h5py.VirtualLayout(shape=(total,) + trailing_shape, dtype=dtype)
    offset = 0
    for ep in episodes:
        vsource = h5py.VirtualSource(str(ep.h5_path), IMAGE_FIELD, shape=ep.img_shape, dtype=dtype)
        layout[offset:offset + ep.n_frames] = vsource
        offset += ep.n_frames
    ds = out_f.create_virtual_dataset('images', layout)
    ds.attrs['stored_as'] = 'image'
    ds.attrs['color_order'] = 'bgr'  # raw camera_obs/image_bgr, un-flipped by the virtual mapping


def collate_metadata(episodes: list[EpisodeInfo]) -> dict:
    """Stack the per-episode raw metadata, keeping only the keys shared by every
    episode whose values actually stack (schemas drift between collection sessions)."""
    metas = [ep.meta for ep in episodes]
    common = set.intersection(*[set(m) for m in metas]) if metas else set()
    dropped = set.union(*[set(m) for m in metas]) - common if metas else set()

    collated = {}
    for key in sorted(common):
        values = [m[key] for m in metas]
        if all(isinstance(v, dict) for v in values):
            dropped.add(key)  # nested metadata groups are rare; not worth recursing for
            continue
        try:
            collated[key] = np.stack(values)
        except ValueError:
            dropped.add(key)
    if dropped:
        print(f'Warning: metadata fields dropped (missing from some episodes, or not stackable): '
              f'{", ".join(sorted(dropped))}')
    return collated


def build_grasp_dataset(sources: list[SourceInfo], out_dir: pathlib.Path, threshold: float | None) -> np.ndarray:
    check_consistent(sources)
    if threshold is None:
        threshold = auto_threshold(sources)
        print(f'Inferred gripper threshold: {threshold:.3f}')
    labels = build_labels(sources, threshold)
    episodes = [ep for s in sources for ep in s.episodes]

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / 'dataset.h5', 'w') as f:
        f.create_dataset('num_episodes', data=len(episodes))
        build_images_vds(f, episodes)

        for name in RESAMPLED_FIELDS:
            f.create_dataset(name, data=np.concatenate([ep.fields[name] for ep in episodes]))
        f.create_dataset('time', data=np.concatenate([ep.times for ep in episodes]))
        ds = f.create_dataset('gripper_dt', data=np.concatenate([ep.gripper_dt for ep in episodes]))
        ds.attrs['meaning'] = ('seconds between an image and the gripper sample held onto it; '
                               'large values mean the label is extrapolated across a logging gap')

        ds = f.create_dataset('label', data=labels)
        ds.attrs['meaning'] = '1 = cable held between the gripper fingers, 0 = not'
        ds.attrs['gripper_thr'] = threshold

        meta_group = f.create_group('metadata')
        for key, value in collate_metadata(episodes).items():
            meta_group.create_dataset(key, data=value)
        meta_group.create_dataset('length', data=np.array([ep.n_frames for ep in episodes]))
        # Per-episode provenance: where the episode came from, and whether it was an
        # empty demo (label 0 everywhere) or a successful one.
        meta_group.create_dataset('episode_path', data=[str(ep.h5_path) for ep in episodes],
                                  dtype=h5py.string_dtype())
        meta_group.create_dataset('source_index', data=np.concatenate(
            [np.full(s.n_episodes, i, dtype=np.int32) for i, s in enumerate(sources)]))
        meta_group.create_dataset('is_empty', data=np.concatenate(
            [np.full(s.n_episodes, s.is_empty, dtype=bool) for s in sources]))
        meta_group.create_dataset('source_path', data=[str(s.dataset_dir) for s in sources],
                                  dtype=h5py.string_dtype())

    return labels


def parse_source_arg(parser: argparse.ArgumentParser, values: list[str]) -> tuple[str, str]:
    if len(values) == 1:
        return values[0], ':'
    if len(values) == 2:
        # A leading space is how a range starting with '-' gets past argparse's option
        # detection (e.g. --success DIR " -25:"); int() ignores it.
        return values[0], values[1].strip()
    parser.error(f'expected DATASET_DIR [EPISODES], got {len(values)} values: {values}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aggregate images from raw successful and empty demonstrations into a single '
                    'dataset labeled with whether the cable is in the gripper fingers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example:
  python scripts/build_grasp_dataset.py \\
      --success /data/ethernet_plug_v3 \\
      --success /data/ethernet_plug_v4 0:50 \\
      --empty /data/ethernet_plug_empty \\
      --output /data/grasp_cls_dataset''')
    parser.add_argument('--success', nargs='+', action='append', default=[],
                        metavar='DATASET_DIR [EPISODES]',
                        help='Raw dataset directory (containing episode*/rawdata.h5) of successful demos, '
                             'optionally followed by an episode range in Python slice syntax (e.g. "0:50"; '
                             'default all episodes -- a range starting with "-" needs a leading space, as in '
                             '" -25:", so argparse does not read it as a flag). Frames are labeled 1 wherever '
                             'gripper_width < threshold. Repeatable.')
    parser.add_argument('--empty', nargs='+', action='append', default=[],
                        metavar='DATASET_DIR [EPISODES]',
                        help='Raw dataset directory of empty demos (cable never grasped); every frame is '
                             'labeled 0. Same episode-range syntax as --success. Repeatable.')
    parser.add_argument('--gripper-thr', type=float, default=None,
                        help='gripper_width below which the cable counts as grasped. Default: midpoint '
                             'between the min and max gripper width over the successful demos.')
    parser.add_argument('--force-alpha', type=float, default=0.03,
                        help='EWMA smoothing factor applied to the force before resampling, matching '
                             'rawdata_to_dataset.py. Pass 0 to keep the raw force.')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory to create')
    parser.add_argument('--force', action='store_true', help='Overwrite the output dataset.h5 if it already exists')
    args = parser.parse_args()

    specs = ([(parse_source_arg(parser, v), False) for v in args.success] +
             [(parse_source_arg(parser, v), True) for v in args.empty])
    if not specs:
        parser.error('at least one --success or --empty source is required')

    out_dir = pathlib.Path(args.output)
    out_h5 = out_dir / 'dataset.h5'
    if out_h5.exists() and not args.force:
        parser.error(f'{out_h5} already exists; pass --force to overwrite')

    force_alpha = args.force_alpha if args.force_alpha > 0 else None
    sources = [inspect_source(path, parse_episode_slice(spec), is_empty, force_alpha)
               for (path, spec), is_empty in specs]
    labels = build_grasp_dataset(sources, out_dir, args.gripper_thr)

    n_pos = int(labels.sum())
    print(f'Wrote {out_h5}')
    print(f'{sum(s.n_episodes for s in sources)} episodes, {len(labels)} images '
          f'({n_pos} in-gripper, {len(labels) - n_pos} not):')
    offset = 0
    for ((path, spec), _), s in zip(specs, sources):
        kind = 'empty  ' if s.is_empty else 'success'
        pos = int(labels[offset:offset + s.n_steps].sum())
        offset += s.n_steps
        dts = np.concatenate([ep.gripper_dt for ep in s.episodes]) if s.episodes else np.zeros(0)
        stale = f'gripper lag med {np.median(dts) * 1e3:.0f} ms, max {dts.max() * 1e3:.0f} ms' if dts.size else 'no frames'
        print(f'  [{kind}] {path} [{spec}] -> episodes {s.start_ep}:{s.stop_ep} '
              f'({s.n_episodes} kept, {s.n_steps} images, {pos} in-gripper; {stale})')
