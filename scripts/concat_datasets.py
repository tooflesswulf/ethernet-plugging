"""
Concatenate `dataset.h5` files produced by `rawdata_to_dataset.py`, optionally
taking only a sub-range of episodes from each source.

The pose/force/gripper/image arrays are linked into the output with HDF5
virtual datasets, so none of that data is copied to disk -- the output file
only stores small metadata (episode lengths, etc.) plus a mapping back to the
source files. This means the output is NOT standalone: the source dataset
directories must stay where they are for the output to remain readable.

Example -- first 50 episodes of A followed by the last 25 episodes of B:
    python scripts/concat_datasets.py \\
        --source /data/ethernet_plug_v3_dataset 0:50 \\
        --source /data/ethernet_plug_v4_dataset -25: \\
        --output /data/combined_dataset
"""
from dataclasses import dataclass
import argparse
import pathlib
import h5py
import numpy as np

STITCHED_FIELDS = ['pose', 'force', 'gripper_width', 'gripper_force']


def parse_episode_slice(spec: str) -> slice:
    """Parse a Python-slice string ('0:50', '-25:', ':') into a `slice`."""
    parts = spec.split(':')
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f'Invalid episode range {spec!r}: expected Python slice syntax, '
            f'e.g. "0:50", "-25:", or ":" for all episodes')
    nums = [int(p) if p else None for p in parts]
    return slice(*nums)


def resolve_episode_range(ep_slice: slice, num_episodes: int) -> tuple[int, int]:
    start, stop, step = ep_slice.indices(num_episodes)
    if step != 1:
        raise ValueError(f'Episode range step must be 1 (a contiguous range), got step={step}')
    return start, stop


def read_metadata(h5group: h5py.Group, ep_range: slice) -> dict:
    data = {}
    for key, item in h5group.items():
        if isinstance(item, h5py.Group):
            data[key] = read_metadata(item, ep_range)
        else:
            data[key] = item[ep_range]
    return data


def merge_metadata(dicts: list[dict]) -> dict:
    merged = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            merged[key] = merge_metadata(values)
        else:
            merged[key] = np.concatenate(values, axis=0)
    return merged


def write_metadata(h5group: h5py.Group, data: dict):
    for key, value in data.items():
        if isinstance(value, dict):
            write_metadata(h5group.create_group(key), value)
        else:
            h5group.create_dataset(key, data=value)


def flatten_keys(d: dict, prefix: str = '') -> list[str]:
    keys = []
    for key, value in d.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, prefix=path + '/'))
        else:
            keys.append(path)
    return keys


@dataclass
class SourceInfo:
    dataset_dir: pathlib.Path
    h5_path: pathlib.Path
    start_ep: int
    stop_ep: int
    time_start: int
    time_stop: int
    field_shapes: dict
    field_dtypes: dict
    images_stored_as: str
    metadata: dict
    framerate: float | None

    @property
    def n_episodes(self):
        return self.stop_ep - self.start_ep

    @property
    def n_steps(self):
        return self.time_stop - self.time_start


def inspect_source(dataset_dir: str, ep_slice: slice) -> SourceInfo:
    dataset_dir = pathlib.Path(dataset_dir).resolve()
    h5_path = dataset_dir / 'dataset.h5'
    with h5py.File(h5_path, 'r') as f:
        lengths = f['metadata/length'][:]
        start_ep, stop_ep = resolve_episode_range(ep_slice, len(lengths))
        time_start = int(lengths[:start_ep].sum())
        time_stop = int(lengths[:stop_ep].sum())

        field_shapes = {name: f[name].shape for name in STITCHED_FIELDS}
        field_dtypes = {name: f[name].dtype for name in STITCHED_FIELDS}
        field_shapes['images'] = f['images'].shape
        field_dtypes['images'] = f['images'].dtype
        images_stored_as = f['images'].attrs['stored_as']

        metadata = read_metadata(f['metadata'], slice(start_ep, stop_ep))
        framerate = f['metadata'].attrs.get('framerate')

    return SourceInfo(dataset_dir, h5_path, start_ep, stop_ep, time_start, time_stop,
                       field_shapes, field_dtypes, images_stored_as, metadata, framerate)


def check_consistent(sources: list[SourceInfo]):
    first = sources[0]
    first_meta_keys = set(flatten_keys(first.metadata))
    for s in sources[1:]:
        for name in STITCHED_FIELDS:
            if s.field_shapes[name][1:] != first.field_shapes[name][1:]:
                raise ValueError(
                    f"{name!r} trailing shape mismatch: {s.dataset_dir} has "
                    f"{s.field_shapes[name][1:]}, expected {first.field_shapes[name][1:]}")
            if s.field_dtypes[name] != first.field_dtypes[name]:
                raise ValueError(
                    f"{name!r} dtype mismatch: {s.dataset_dir} has {s.field_dtypes[name]}, "
                    f"expected {first.field_dtypes[name]}")

        if s.images_stored_as != first.images_stored_as:
            raise ValueError(
                f"images storage mode mismatch: {s.dataset_dir} stores images as "
                f"{s.images_stored_as!r}, expected {first.images_stored_as!r} "
                f"(regenerate one of the datasets so both use the same --h5_images setting)")
        if s.images_stored_as == 'image' and s.field_shapes['images'][1:] != first.field_shapes['images'][1:]:
            raise ValueError(
                f"image shape mismatch: {s.dataset_dir} has {s.field_shapes['images'][1:]}, "
                f"expected {first.field_shapes['images'][1:]}")

        if first.framerate is not None and s.framerate != first.framerate:
            raise ValueError(f"framerate mismatch: {s.dataset_dir} has {s.framerate}, expected {first.framerate}")

        if set(flatten_keys(s.metadata)) != first_meta_keys:
            raise ValueError(f"metadata fields differ between {first.dataset_dir} and {s.dataset_dir}")


def build_virtual_dataset(out_f: h5py.File, name: str, sources: list[SourceInfo]):
    dtype = sources[0].field_dtypes[name]
    trailing_shape = sources[0].field_shapes[name][1:]
    total = sum(s.n_steps for s in sources)

    layout = h5py.VirtualLayout(shape=(total,) + trailing_shape, dtype=dtype)
    offset = 0
    for s in sources:
        vsource = h5py.VirtualSource(str(s.h5_path), name, shape=s.field_shapes[name], dtype=dtype)
        n = s.n_steps
        layout[offset:offset + n] = vsource[s.time_start:s.time_stop]
        offset += n
    out_f.create_virtual_dataset(name, layout)


def build_filepath_images(out_f: h5py.File, sources: list[SourceInfo]):
    all_paths = []
    for s in sources:
        with h5py.File(s.h5_path, 'r') as f:
            raw_paths = f['images'][s.time_start:s.time_stop]
        for p in raw_paths:
            p = p.decode() if isinstance(p, bytes) else p
            resolved = pathlib.Path(p)
            if not resolved.is_absolute():
                resolved = s.dataset_dir / resolved
            all_paths.append(str(resolved))
    out_f.create_dataset('images', data=all_paths, dtype=h5py.string_dtype())
    out_f['images'].attrs['stored_as'] = 'filepath'


def concat_datasets(sources: list[SourceInfo], out_dir: pathlib.Path):
    check_consistent(sources)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / 'dataset.h5', 'w') as f:
        f.create_dataset('num_episodes', data=sum(s.n_episodes for s in sources))

        for name in STITCHED_FIELDS:
            build_virtual_dataset(f, name, sources)

        if sources[0].images_stored_as == 'image':
            build_virtual_dataset(f, 'images', sources)
            f['images'].attrs['stored_as'] = 'image'
        else:
            build_filepath_images(f, sources)

        meta_group = f.create_group('metadata')
        write_metadata(meta_group, merge_metadata([s.metadata for s in sources]))
        if sources[0].framerate is not None:
            meta_group.attrs['framerate'] = sources[0].framerate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate dataset.h5 files (from rawdata_to_dataset.py) using virtual datasets, '
                     'optionally taking only a sub-range of episodes from each one.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example -- first 50 episodes of A followed by the last 25 episodes of B:
  python scripts/concat_datasets.py \\
      --source /data/ethernet_plug_v3_dataset 0:50 \\
      --source /data/ethernet_plug_v4_dataset -25: \\
      --output /data/combined_dataset''')
    parser.add_argument('--source', nargs=2, action='append', required=True,
                        metavar=('DATASET_DIR', 'EPISODES'),
                        help='A dataset directory (containing dataset.h5) and the episode range to take '
                             'from it, in Python slice syntax (e.g. "0:50", "-25:", ":" for all episodes). '
                             'Repeat --source for each input, in concatenation order.')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory to create')
    parser.add_argument('--force', action='store_true', help='Overwrite the output dataset.h5 if it already exists')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output)
    out_h5 = out_dir / 'dataset.h5'
    if out_h5.exists() and not args.force:
        parser.error(f'{out_h5} already exists; pass --force to overwrite')

    sources = [inspect_source(path, parse_episode_slice(spec)) for path, spec in args.source]
    concat_datasets(sources, out_dir)

    print(f'Wrote {out_h5}')
    print(f'{sum(s.n_episodes for s in sources)} episodes, {sum(s.n_steps for s in sources)} steps:')
    for (path, spec), s in zip(args.source, sources):
        print(f'  {path} [{spec}] -> episodes {s.start_ep}:{s.stop_ep} ({s.n_steps} steps)')
