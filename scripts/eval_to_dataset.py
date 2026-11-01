"""
Turn successful evaluation episodes (agent/eval/eval_splits.py, eval_residual.py and the
experiments built on them) into a PICKUP and an INSERT dataset in split_dataset.py's
format, for fine-tuning / retraining the split policies on their own successes:

    pickup     pickup.t_start -> the end of the grasp settle at the start of transport
    insert     insert.t_start + margin -> network up (or just after a release), only when
               the plug seated, with a long contact search trimmed off (below)

Phase boundaries come from each episode's phases.json (on the rawdata.h5 clock), not
from DpadDown presses as in split_dataset.py: an autonomous pickup ends on its done
score. Frames are sampled by rawdata_to_dataset.proc_h5, so they match the teleop
datasets frame for frame (frame k at t = (k + 1) / framerate).

Pickup runs past its phase: eval only closes the gripper at the start of transport
(seq.gripper, then a hold, then the scripted lift), so a pickup cut at the phase end
never closes and would label every step 'open' -- a policy trained on it never commands
the close the done trigger waits for. The scripted hold until the target first moves
is what a teleop operator did before DpadDown (close, wait ~0.7 s), and the last
end_signal_steps frames label done there, as in the teleop data. Pickups the operator
ended with DpadDown (the done score never fired: the policy hovered) are skipped unless
--dpad_pickups.

The network can come up before the plug is fully in. A seated insert ends on the stiff
final push the teleop demos all end on (Cross: K_z 3000, every v2v3 insert demo); an
unseated one sits at K_z 300 pressing a few N, until the link happens to come up. An
insert is kept only if, at its end, K_z >= --stiff_k and fz reached --push_n during the
trailing stiff stretch. Its end is network up, or the policy opening the gripper if
that came first: a release is fine once the plug is seated (the force drops as the
fingers slide down the latched plug), so a kept release stays in, --post_release_s of it
(after that the arm only sinks along the plug).

A long contact search is trimmed off the start, never spliced out of the middle: a
chunk across a splice would teach a jump of the arm (the search drifts mm in xy). With
z_f the height at the insert's end: contact (fz > --contact_n) -> entry (the last frame
above z_f + --entry_mm) longer than --max_search_s starts the episode --search_keep_s
before entry, dropping the approach and the early search (the fast episodes cover both).

Example:
    python scripts/eval_to_dataset.py \\
        --source ../robot-logs/logs-video-75 --output ../data/zforce-v75
    # -> ../data/zforce-v75-pickup_dataset, ../data/zforce-v75-insert_dataset
    # then e.g. concat_datasets.py with ../data/ethernet-impedance-v2v3-<phase>_dataset
"""
from dataclasses import dataclass
from PIL import Image
import numpy as np
import argparse
import pathlib
import json
import h5py

from scripts.rawdata_to_dataset import proc_h5

STITCHED_FIELDS = ['pose', 'force', 'gripper_width', 'gripper_force',
                   'target_pose', 'stiffness', 'damping', 'mass']
EPISODE_FIELDS = {'pose': 'poses', 'force': 'forces', 'gripper_width': 'gripper_widths',
                  'gripper_force': 'gripper_forces', 'target_pose': 'target_poses',
                  'stiffness': 'stiffness', 'damping': 'damping', 'mass': 'mass'}
PHASES = ('pickup', 'insert')
GRIP_RANGE_ATTR = 'grip_width_range'   # agent/dataset/sequence.py
MIN_FRAMES = 16                        # one action chunk (train.py --action_horizon)


@dataclass
class Insert:
    """How one insert ended, and what the search trimming did to it."""
    kz_end: float               # K_z at the end (network up, or the release)
    push_n: float               # max fz over the trailing K_z >= stiff_k stretch (0 if none)
    released: bool
    search_s: float = 0.
    cut_start_s: float = 0.     # dropped from the start (approach + early search)


def phases(ep_dir: pathlib.Path) -> tuple[dict, dict]:
    j = json.loads((ep_dir / 'phases.json').read_text())
    return j, {p['phase']: p for p in j['phases']}


def grasp_settle_end(f: h5py.File, t_transport: float, move_mm: float) -> float:
    """When the scripted lift starts: the first time the held target moves after transport."""
    kt, target = f['control/time'][:], f['control/target'][:, :3]
    i = np.searchsorted(kt, t_transport)
    moved = np.linalg.norm(target[i:] - target[i], axis=1) > move_mm / 1000
    if not moved.any():
        raise ValueError('the target never moves after transport starts')
    return float(kt[i + np.argmax(moved)])


def seating(kz, fz, released, args) -> Insert:
    """How an insert ended, from its K_z and fz up to the end (network up or release)."""
    stiff = kz >= args.stiff_k
    soft = np.where(~stiff)[0]
    tail = slice(soft[-1] + 1 if len(soft) else 0, None)
    return Insert(kz_end=float(kz[-1]), push_n=float(fz[tail].max()) if stiff[-1] else 0.,
                  released=released)


def search_trim(z, fz, ins: Insert, framerate, args) -> int:
    """First frame of an insert (pose z, force z, up to its end) to keep: trims a long search."""
    above = np.where(z >= z[-1] + args.entry_mm / 1000)[0]
    entry = int(above[-1]) if len(above) else 0
    contact = int(np.argmax(fz > args.contact_n)) if (fz > args.contact_n).any() else entry
    ins.search_s = (entry - min(contact, entry)) / framerate
    start = 0
    if ins.search_s > args.max_search_s:
        start = max(0, entry - round(args.search_keep_s * framerate))
    ins.cut_start_s = start / framerate
    return start


def convert(sources, out_prefix: pathlib.Path, args):
    outs = {p: out_prefix.parent / f'{out_prefix.name}-{p}_dataset' for p in PHASES}
    for d in outs.values():
        if (d / 'dataset.h5').exists() and not args.force:
            raise SystemExit(f'{d / "dataset.h5"} already exists; pass --force to overwrite')
        (d / 'images').mkdir(parents=True, exist_ok=True)

    collected = {p: {k: [] for k in STITCHED_FIELDS + ['images']} for p in PHASES}
    meta = {p: {k: [] for k in ('length', 'id', 't_start', 't_end', 'trim_start_s')} for p in PHASES}
    meta['insert'].update(kz_end=[], push_n=[], released=[])
    source_names = {p: [] for p in PHASES}
    # open-to-closed range of the full episodes (rawdata_to_dataset's), for the gripper label
    grip_range = np.array([np.inf, -np.inf])
    fr = args.framerate

    print(f'{"episode":>24}  {"pickup":>8}  {"insert":>10}  K_z end  push N  search s  trimmed s  released')
    for raw_dir in sources:
        raw_dir = pathlib.Path(raw_dir).resolve()
        for ep_dir in sorted(p for p in raw_dir.iterdir() if p.name.startswith('episode')):
            name = f'{raw_dir.name}/{ep_dir.name}'
            try:
                j, ph = phases(ep_dir)
            except FileNotFoundError:
                print(f'{name:>24}  skip: no phases.json')
                continue
            if not j.get('success') or ph.get('insert', {}).get('end') != 'network':
                continue

            windows, notes = {}, {}
            with h5py.File(ep_dir / 'rawdata.h5', 'r') as f:
                if ph['pickup']['end'].startswith('done') or args.dpad_pickups:
                    t_hold = grasp_settle_end(f, ph['transport']['t_start'], args.hold_move_mm)
                    windows['pickup'] = (ph['pickup']['t_start'], t_hold)
                else:
                    notes['pickup'] = ph['pickup']['end']
            windows['insert'] = (ph['insert']['t_start'] + args.margin, ph['insert']['t_end'])

            ep = proc_h5(ep_dir / 'rawdata.h5', framerate=fr)
            data = {k: np.asarray(getattr(ep, v)) for k, v in EPISODE_FIELDS.items()}
            frame_t = (np.arange(ep.meta['length']) + 1) / fr
            w = data['gripper_width']
            grip_range = np.array([min(grip_range[0], w.min()), max(grip_range[1], w.max())])
            ep_id = int(ep_dir.name.removeprefix('episode'))

            ins = None
            for phase, (t0, t1) in windows.items():
                idx = np.where((frame_t >= t0) & (frame_t < t1))[0]
                held = w[idx] < args.grip_open_mm
                if phase == 'pickup' and not held[-1]:
                    notes[phase] = 'gripper open'
                    continue
                if phase == 'insert':
                    # judged where the plug was let go of: network up, or the policy's release
                    n_held = len(idx) if held.all() else int(np.argmin(held))
                    hold = idx[:n_held]
                    ins = seating(data['stiffness'][hold, 2], data['force'][hold, 2], n_held < len(idx), args)
                    if ins.kz_end < args.stiff_k:
                        notes[phase] = 'not stiff'
                        continue
                    if ins.push_n < args.push_n:
                        notes[phase] = 'no push'
                        continue
                    start = search_trim(data['pose'][hold, 2], data['force'][hold, 2], ins, fr, args)
                    stop = n_held + round(args.post_release_s * fr) if ins.released else len(idx)
                    idx = idx[start:stop]
                if len(idx) < MIN_FRAMES:
                    notes[phase] = f'{len(idx)} frames'
                    continue
                for k in STITCHED_FIELDS:
                    collected[phase][k].append(data[k][idx])
                paths = []
                for i in idx:
                    rel = f'images/{raw_dir.name}_ep{ep_id}_img{i}.png'
                    Image.fromarray(ep.images[i]).save(outs[phase] / rel)
                    paths.append(rel)
                collected[phase]['images'].append(paths)
                m = meta[phase]
                m['length'].append(len(idx))
                m['id'].append(ep_id)
                m['t_start'].append(frame_t[idx[0]])
                m['t_end'].append(frame_t[idx[-1]] + 1 / fr)
                m['trim_start_s'].append(ins.cut_start_s if phase == 'insert' else 0.)
                if phase == 'insert':
                    m['kz_end'].append(ins.kz_end)
                    m['push_n'].append(ins.push_n)
                    m['released'].append(ins.released)
                source_names[phase].append(raw_dir.name)
                notes.setdefault(phase, f'{len(idx)}')

            print(f'{name:>24}  {notes.get("pickup", "-"):>8}  {notes.get("insert", "-"):>10}  '
                  f'{ins.kz_end:7.0f}  {ins.push_n:6.1f}  {ins.search_s:8.1f}  {ins.cut_start_s:9.1f}  '
                  f'{"yes" if ins.released else ""}')

    for phase, out_dir in outs.items():
        c, m = collected[phase], meta[phase]
        if not m['length']:
            print(f'No {phase} episodes kept; {out_dir} not written')
            continue
        n = len(m['length'])
        with h5py.File(out_dir / 'dataset.h5', 'w') as f:
            f.create_dataset('num_episodes', data=n)
            for k in STITCHED_FIELDS:
                f.create_dataset(k, data=np.concatenate(c[k], axis=0))
            f.create_dataset('images', data=sum(c['images'], []), dtype=h5py.string_dtype())
            f['images'].attrs['stored_as'] = 'filepath'
            g = f.create_group('metadata')
            for k, v in m.items():
                g.create_dataset(k, data=np.asarray(v))
            # eval logs no gripper settings (their metadata is empty): the policies' grip_stats
            for k, v in zip(('grip_width_mm', 'grip_force_n', 'grip_speed_mmps', 'grip_pullback_mm'),
                            args.grip_stats):
                g.create_dataset(k, data=np.full(n, v))
            g.create_dataset('source', data=source_names[phase], dtype=h5py.string_dtype())
            g.attrs['framerate'] = fr
            g.attrs['phase'] = phase
            g.attrs[GRIP_RANGE_ATTR] = grip_range
        L = np.asarray(m['length'])
        print(f'Wrote {out_dir / "dataset.h5"}: {n} episodes, {L.sum()} steps '
              f'(length {L.min()}-{L.max()}, median {np.median(L):.0f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Successful eval episodes -> pickup / insert datasets, with insert stalls trimmed.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--source', action='append', required=True,
                        help='eval log directory (episode*/rawdata.h5 + phases.json). Repeat for several.')
    parser.add_argument('--output', type=str, required=True,
                        help='output prefix: writes <output>-pickup_dataset and <output>-insert_dataset')
    parser.add_argument('--framerate', type=float, default=20.0,
                        help='sampling rate; must match the datasets these are mixed with')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='seconds dropped at the start of insert, while the gains ramp back from '
                             'the scripted stiffness (as split_dataset.py)')
    parser.add_argument('--hold_move_mm', type=float, default=0.5,
                        help='target motion that ends the grasp settle at the start of transport')
    parser.add_argument('--dpad_pickups', action='store_true',
                        help='also keep pickups the operator ended with DpadDown')
    parser.add_argument('--contact_n', type=float, default=2.5, help='z-force that marks first contact (N)')
    parser.add_argument('--entry_mm', type=float, default=3.0,
                        help='height above the end pose where the last descent starts (mm)')
    parser.add_argument('--max_search_s', type=float, default=7.0,
                        help='contact searches longer than this are a stall (successes: median ~5 s)')
    parser.add_argument('--search_keep_s', type=float, default=3.0, help='search kept before entry when trimming')
    parser.add_argument('--stiff_k', type=float, default=1000.,
                        help='K_z (N/m) at the end that counts as the stiff final push (teleop: 3000, '
                             'unseated: 300; this is about their log-midpoint)')
    parser.add_argument('--push_n', type=float, default=10.,
                        help='fz (N) the trailing stiff stretch must reach for the plug to count as seated')
    parser.add_argument('--post_release_s', type=float, default=0.5,
                        help='seconds kept after a release (before the network came up)')
    parser.add_argument('--grip_open_mm', type=float, default=20.0,
                        help='gripper width above which the cable counts as not held')
    parser.add_argument('--grip_stats', type=float, nargs=4, default=[10., 40., 50., 10.],
                        metavar=('WIDTH_MM', 'FORCE_N', 'SPEED_MMPS', 'PULLBACK_MM'),
                        help="gripper settings the eval ran with (the policies' grip_stats)")
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')
    args = parser.parse_args()
    convert(args.source, pathlib.Path(args.output), args)
