"""
Aggregate images from several `dataset.h5` files (produced by
`rawdata_to_dataset.py`) into a single image-classification dataset labeled with
whether the cable is held between the gripper fingers.

Two kinds of source datasets are expected:
  * successful demonstrations (`--success`): the cable is assumed to be in the
    fingers exactly when the gripper is closed (`gripper_width < threshold`);
  * empty demonstrations (`--empty`): the gripper never holds the cable, so
    every frame is labeled 0 regardless of the gripper width.

As in `concat_datasets.py`, the per-timestep arrays are linked into the output
with HDF5 virtual datasets, so no image data is copied to disk -- only the small
label array and metadata are written. The output is therefore NOT standalone:
the source dataset directories must stay where they are for it to stay readable.

Example:
    python scripts/build_grasp_dataset.py \\
        --success /data/ethernet_plug_v3_dataset \\
        --success /data/ethernet_plug_v4_dataset 0:50 \\
        --empty /data/ethernet_plug_empty_dataset \\
        --output /data/grasp_cls_dataset
"""
from dataclasses import dataclass
import argparse
import pathlib
import h5py
import numpy as np

try:
    from concat_datasets import (parse_episode_slice, resolve_episode_range, read_metadata,
                                 merge_metadata, write_metadata, intersect_metadata)
except ImportError:  # when run as `python -m scripts.build_grasp_dataset` from the repo root
    from scripts.concat_datasets import (parse_episode_slice, resolve_episode_range, read_metadata,
                                         merge_metadata, write_metadata, intersect_metadata)

# Per-timestep fields linked into the output alongside the images. `gripper_width` is
# what the labels are derived from; the rest are kept so the labels can be inspected
# (or re-derived with a different threshold) without reopening the sources.
STITCHED_FIELDS = ['pose', 'force', 'gripper_width', 'gripper_force']


@dataclass
class SourceInfo:
    dataset_dir: pathlib.Path
    h5_path: pathlib.Path
    is_empty: bool
    start_ep: int
    stop_ep: int
    time_start: int
    time_stop: int
    field_shapes: dict
    field_dtypes: dict
    images_stored_as: str
    gripper_width: np.ndarray  # (n_steps,), for the selected episode range only
    metadata: dict
    framerate: float | None

    @property
    def n_episodes(self):
        return self.stop_ep - self.start_ep

    @property
    def n_steps(self):
        return self.time_stop - self.time_start


def inspect_source(dataset_dir: str, ep_slice: slice, is_empty: bool) -> SourceInfo:
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

        gripper_width = np.asarray(f['gripper_width'][time_start:time_stop]).reshape(-1)
        metadata = read_metadata(f['metadata'], slice(start_ep, stop_ep))
        framerate = f['metadata'].attrs.get('framerate')

    return SourceInfo(dataset_dir, h5_path, is_empty, start_ep, stop_ep, time_start, time_stop,
                      field_shapes, field_dtypes, images_stored_as, gripper_width, metadata, framerate)


def check_consistent(sources: list[SourceInfo]):
    first = sources[0]
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


def auto_threshold(sources: list[SourceInfo]) -> float:
    """Midpoint between the widest and narrowest gripper width seen in the
    successful demos, i.e. halfway between "fully open" and "clamped on cable".

    Same rule as `StitchedSequenceDataset._precompute_actions`.
    """
    widths = [s.gripper_width for s in sources if not s.is_empty]
    if not widths or all(w.size == 0 for w in widths):
        raise ValueError('Cannot infer a gripper threshold without any --success sources; pass --gripper-thr')
    widths = np.concatenate(widths)
    return float((widths.max() + widths.min()) / 2)


def build_labels(sources: list[SourceInfo], threshold: float) -> np.ndarray:
    """1 = cable between the fingers, 0 = not."""
    labels = []
    for s in sources:
        if s.is_empty:
            labels.append(np.zeros(s.n_steps, dtype=np.uint8))
        else:
            labels.append((s.gripper_width < threshold).astype(np.uint8))
    return np.concatenate(labels)


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
    """Image paths are stored relative to their own dataset dir, so make them
    absolute -- the output lives somewhere else."""
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


def build_grasp_dataset(sources: list[SourceInfo], out_dir: pathlib.Path, threshold: float | None) -> np.ndarray:
    check_consistent(sources)
    if threshold is None:
        threshold = auto_threshold(sources)
        print(f'Inferred gripper threshold: {threshold:.3f}')
    labels = build_labels(sources, threshold)

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

        ds = f.create_dataset('label', data=labels)
        ds.attrs['meaning'] = '1 = cable held between the gripper fingers, 0 = not'
        ds.attrs['gripper_thr'] = threshold

        meta_group = f.create_group('metadata')
        write_metadata(meta_group, merge_metadata(intersect_metadata(sources)))
        # Per-episode provenance: which source file the episode came from, and whether
        # it was an empty demo (label 0 everywhere) or a successful one.
        meta_group.create_dataset('source_index', data=np.concatenate(
            [np.full(s.n_episodes, i, dtype=np.int32) for i, s in enumerate(sources)]))
        meta_group.create_dataset('is_empty', data=np.concatenate(
            [np.full(s.n_episodes, s.is_empty, dtype=bool) for s in sources]))
        meta_group.create_dataset('source_path', data=[str(s.h5_path) for s in sources],
                                  dtype=h5py.string_dtype())
        if sources[0].framerate is not None:
            meta_group.attrs['framerate'] = sources[0].framerate

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
        description='Aggregate images from successful and empty demonstrations into a single dataset '
                    'labeled with whether the cable is in the gripper fingers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example:
  python scripts/build_grasp_dataset.py \\
      --success /data/ethernet_plug_v3_dataset \\
      --success /data/ethernet_plug_v4_dataset 0:50 \\
      --empty /data/ethernet_plug_empty_dataset \\
      --output /data/grasp_cls_dataset''')
    parser.add_argument('--success', nargs='+', action='append', default=[],
                        metavar='DATASET_DIR [EPISODES]',
                        help='Dataset directory of successful demos, optionally followed by an episode '
                             'range in Python slice syntax (e.g. "0:50"; default all episodes -- a range '
                             'starting with "-" needs a leading space, as in " -25:", so argparse does not '
                             'read it as a flag). Frames are labeled 1 wherever gripper_width < threshold. '
                             'Repeatable.')
    parser.add_argument('--empty', nargs='+', action='append', default=[],
                        metavar='DATASET_DIR [EPISODES]',
                        help='Dataset directory of empty demos (cable never grasped); every frame is '
                             'labeled 0. Same episode-range syntax as --success. Repeatable.')
    parser.add_argument('--gripper-thr', type=float, default=None,
                        help='gripper_width below which the cable counts as grasped. Default: midpoint '
                             'between the min and max gripper width over the successful demos.')
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

    sources = [inspect_source(path, parse_episode_slice(spec), is_empty)
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
        print(f'  [{kind}] {path} [{spec}] -> episodes {s.start_ep}:{s.stop_ep} '
              f'({s.n_steps} images, {pos} in-gripper)')
