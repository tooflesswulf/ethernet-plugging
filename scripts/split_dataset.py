"""
Split teleoperation episodes into a PICKUP and an INSERT dataset, dropping the scripted
motions in between (teleoperation.py), for agent/eval/eval_split.py:

    pickup     teleop: start of episode -> DpadDown (operator triggers move_to_port)
    transport  scripted: move_relative up 2 cm, move_to(port_pose)         (dropped)
    insert     teleop: transport done -> network up (teleop autodetects the plug)
    unplug     scripted: unplug_and_release                                (dropped)

Segment boundaries come from each episode's rawdata.h5: the first DpadDown press, the
end of transport (the scripted stiffness, fdcc.toml [scripted], is restored when the
last MotionStep converges), and the first network_status = True. Frames come from the
already-processed `<raw>_dataset` (scripts/rawdata_to_dataset.py), whose frame k was
sampled at t = (k + 1) / framerate on the rawdata.h5 clock -- nothing is resampled and
no images are rewritten; image paths are stored absolute, as concat_datasets.py does,
so the source dataset directories must stay where they are.

Episodes without the scripted markers (e.g. DAgger rollouts from teleoperation-rl2.py)
are skipped with a warning.

Example:
    python scripts/split_dataset.py \\
        --source ../data/ethernet-impedance-v2 --source ../data/ethernet-impedance-v3 \\
        --output ../data/ethernet-impedance-v2v3
    # -> ../data/ethernet-impedance-v2v3-pickup_dataset, ../data/ethernet-impedance-v2v3-insert_dataset
"""
from dataclasses import dataclass
import argparse
import pathlib
import h5py
import numpy as np

STITCHED_FIELDS = ['pose', 'force', 'gripper_width', 'gripper_force',
                   'target_pose', 'stiffness', 'damping', 'mass']
SCRIPTED_K = 2999.     # fdcc.toml [scripted] stiffness is 3000 N/m; teleop runs at 300
PHASES = ('pickup', 'insert')


@dataclass
class Segments:
    t_down: float           # DpadDown: pickup ends, transport starts
    t_transport_end: float  # scripted gains restored at the port pose
    t_net: float            # network up: insert ends, unplug starts
    t_final: float          # last sample of any stream (rawdata_to_dataset's tf)


def rising_edges(t, x):
    x = np.asarray(x, bool)
    return t[1:][x[1:] & ~x[:-1]]


def falling_edges(t, x):
    x = np.asarray(x, bool)
    return t[1:][~x[1:] & x[:-1]]


def find_segments(raw_h5: pathlib.Path) -> Segments:
    """Phase boundaries of one episode, on the rawdata.h5 clock. Raises ValueError if missing."""
    with h5py.File(raw_h5, 'r') as f:
        down = rising_edges(f['dualsense/time'][:], f['dualsense/DpadDown'][:])
        net_t, net = f['network/time'][:], f['network/status'][:]
        kt, kx = f['control/time'][:], f['control/K'][:, 0]
        t_final = max(f['robot_obs/time'][-1], f['gripper_obs/time'][-1], f['camera_obs/time'][-1])
    if len(down) == 0:
        raise ValueError('no DpadDown press (no scripted transport)')
    if not net.any():
        raise ValueError('network never came up (no insertion)')
    t_down, t_net = float(down[0]), float(net_t[np.argmax(net)])
    if t_net < t_down:
        raise ValueError(f'network up at {t_net:.1f} s, before DpadDown at {t_down:.1f} s')
    # Transport is two MotionSteps, each stiffening then restoring; the last restore
    # before the plug goes in ends it. (Cross's stiff z only touches base z, not K[0].)
    restores = falling_edges(kt, kx >= SCRIPTED_K)
    restores = restores[(restores > t_down) & (restores < t_net)]
    if len(restores) == 0:
        raise ValueError('no scripted-gain restore between DpadDown and network up')
    return Segments(t_down, float(restores[-1]), t_net, float(t_final))


def phase_windows(seg: Segments, margin: float) -> dict:
    """Time window [t0, t1) of each phase. `margin` skips the gain ramp after transport."""
    return {'pickup': (0., seg.t_down),
            'insert': (seg.t_transport_end + margin, seg.t_net)}


def split(sources, out_prefix: pathlib.Path, margin: float, min_transport_s: float,
          max_transport_s: float, grip_open_mm: float, force: bool):
    outs = {p: out_prefix.parent / f'{out_prefix.name}-{p}_dataset' for p in PHASES}
    for d in outs.values():
        if (d / 'dataset.h5').exists() and not force:
            raise SystemExit(f'{d / "dataset.h5"} already exists; pass --force to overwrite')

    collected = {p: {k: [] for k in STITCHED_FIELDS + ['images']} for p in PHASES}
    meta = {p: {k: [] for k in ('length', 'id', 'grip_width_mm', 'grip_force_n',
                                'grip_speed_mmps', 'grip_pullback_mm', 't_start', 't_end')}
            for p in PHASES}
    source_names = {p: [] for p in PHASES}
    framerate = None

    for raw_dir in sources:
        raw_dir = pathlib.Path(raw_dir).resolve()
        ds_dir = raw_dir.parent / (raw_dir.name + '_dataset')
        episodes = sorted(p for p in raw_dir.iterdir() if p.name.startswith('episode'))
        with h5py.File(ds_dir / 'dataset.h5', 'r') as f:
            fr = float(f['metadata'].attrs['framerate'])
            assert framerate in (None, fr), f'framerate mismatch: {ds_dir} is {fr}, expected {framerate}'
            framerate = fr
            assert f['images'].attrs['stored_as'] == 'filepath', f'{ds_dir}: h5 images not supported'
            lengths = f['metadata/length'][:]
            assert len(lengths) == len(episodes), \
                f'{ds_dir} has {len(lengths)} episodes, {raw_dir} has {len(episodes)}'
            offsets = np.r_[0, np.cumsum(lengths)]
            data = {k: f[k][:] for k in STITCHED_FIELDS}
            images = f['images'][:]
            ds_meta = {k: f[f'metadata/{k}'][:] for k in meta['pickup'] if f'metadata/{k}' in f}

        n_kept = {p: 0 for p in PHASES}
        for i, ep in enumerate(episodes):
            try:
                seg = find_segments(ep / 'rawdata.h5')
            except ValueError as e:
                print(f'  skip {raw_dir.name}/{ep.name}: {e}')
                continue
            # dataset episode i is the i-th sorted episode dir (metadata/id can be stale when
            # an episode was re-recorded with --id); check it by its frame count
            n_frames = len(np.arange(1 / framerate, seg.t_final, 1 / framerate))
            if n_frames != lengths[i]:
                raise ValueError(f'{ds_dir} episode {i} has {lengths[i]} frames, {ep} gives {n_frames}')
            transport = seg.t_transport_end - seg.t_down
            if not min_transport_s <= transport <= max_transport_s:
                print(f'  skip {raw_dir.name}/{ep.name}: transport took {transport:.1f} s')
                continue
            frame_t = (np.arange(lengths[i]) + 1) / framerate      # rawdata_to_dataset sample times
            for phase, (t0, t1) in phase_windows(seg, margin).items():
                idx = offsets[i] + np.where((frame_t >= t0) & (frame_t < t1))[0]
                # The cable must be held where a phase hands over to a scripted motion: a
                # DpadDown before the fingers closed (v2 ep 47, 48) labels 'done' while open;
                # an operator release before the network came up (v2 ep 5) teaches opening.
                held = data['gripper_width'][idx] < grip_open_mm
                if not (held[-1] if phase == 'pickup' else held.all()):
                    print(f'  skip {raw_dir.name}/{ep.name} {phase}: gripper open '
                          f'{"at the end" if phase == "pickup" else "during insertion"}')
                    continue
                for k in STITCHED_FIELDS:
                    collected[phase][k].append(data[k][idx])
                collected[phase]['images'].append(
                    [str(ds_dir / p.decode()) for p in images[idx]])
                meta[phase]['length'].append(len(idx))
                meta[phase]['t_start'].append(t0)
                meta[phase]['t_end'].append(t1)
                meta[phase]['id'].append(int(ep.name.removeprefix('episode')))
                for k in ('grip_width_mm', 'grip_force_n', 'grip_speed_mmps', 'grip_pullback_mm'):
                    meta[phase][k].append(ds_meta[k][i])
                source_names[phase].append(raw_dir.name)
                n_kept[phase] += 1
        print(f'{raw_dir}: kept ' + ', '.join(f'{n}/{len(episodes)} {p}' for p, n in n_kept.items()))

    for phase, out_dir in outs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        c, m = collected[phase], meta[phase]
        with h5py.File(out_dir / 'dataset.h5', 'w') as f:
            f.create_dataset('num_episodes', data=len(m['length']))
            for k in STITCHED_FIELDS:
                f.create_dataset(k, data=np.concatenate(c[k], axis=0))
            f.create_dataset('images', data=sum(c['images'], []), dtype=h5py.string_dtype())
            f['images'].attrs['stored_as'] = 'filepath'
            g = f.create_group('metadata')
            for k, v in m.items():
                g.create_dataset(k, data=np.asarray(v))
            g.create_dataset('source', data=source_names[phase], dtype=h5py.string_dtype())
            g.attrs['framerate'] = framerate
            g.attrs['phase'] = phase
        L = np.asarray(m['length'])
        print(f'Wrote {out_dir / "dataset.h5"}: {len(L)} episodes, {L.sum()} steps '
              f'(length {L.min()}-{L.max()}, median {np.median(L):.0f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split teleop episodes into pickup / insert datasets around the scripted motions.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--source', action='append', required=True,
                        help='raw episode directory (with episode*/rawdata.h5); its processed dataset '
                             'must be next to it as <source>_dataset. Repeat for several.')
    parser.add_argument('--output', type=str, required=True,
                        help='output prefix: writes <output>-pickup_dataset and <output>-insert_dataset')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='seconds dropped at the start of insert, while the gains ramp back from '
                             'the scripted stiffness (fdcc.toml gain_ramp_s = 0.1)')
    parser.add_argument('--min_transport_s', type=float, default=2.0)
    parser.add_argument('--max_transport_s', type=float, default=10.0)
    parser.add_argument('--grip_open_mm', type=float, default=20.0,
                        help='gripper width above which the cable counts as not held')
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')
    args = parser.parse_args()
    split(args.source, pathlib.Path(args.output), args.margin,
          args.min_transport_s, args.max_transport_s, args.grip_open_mm, args.force)
