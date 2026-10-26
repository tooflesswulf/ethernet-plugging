"""
Hand-written residual for the insert policy: keep the z-force low while it searches for
the port, without stopping the insertion push.

The insert policy's failure mode (logs-split-policy, 2026-10): while it wiggles the plug
over the port it keeps descending (commanded z 1-8 mm/s, K_z 300 N/m), and the z-force
builds up slowly until the plug is pinned (~25 N for 20 s, the operator aborts). The
pushes that seat the plug are fast and stiff: 21-40 mm/s with K_z ~3000 N/m, 25-46 N.

ZForceRegulator is a function of the current observation only. Above f_floor it asks for
an upward z-velocity linear in the excess force, capped:

    v = min(gain (F_z - f_floor), max_speed),    F_z = filtered_force z (+ = pushing down)

and sends it as the offset dz = v * horizon. The insert policy re-anchors each chunk on the
corrected target, so a held dz is taken into its command and added again: the target
drifts up at about dz / (the time the policy takes to absorb it), and horizon is that
time (calibrate it: Episode ticks give the target's drift). Against a search descending
at v_s the force settles where v = v_s, f_floor + v_s / gain (10 N at 5 mm/s with the
defaults); a push faster than max_speed outraces the correction and builds force.

    python -m experiments.zforce_regulator --insert_ckpt <ckpt> --log_dir <dir> --episodes 5
"""
from agent.eval.eval_residual import ResidualEnv, ResidualActor, Correction
import numpy as np
import argparse
import json
import os


class ZForceRegulator(ResidualActor):
    def __init__(self, f_floor=5.0, gain=1e-3, max_speed=0.01, horizon=0.4):
        self.f_floor = f_floor      # [N] no correction below this
        self.gain = gain            # [m/s per N] upward velocity per N over f_floor
        self.max_speed = max_speed  # [m/s] fastest upward velocity: slower than an insertion push
        self.horizon = horizon      # [s] the time over which the policy absorbs an offset

    def act(self, step):
        fz = float(step.state['filtered_force'][2])
        v = min(self.gain * max(0., fz - self.f_floor), self.max_speed)
        return Correction(dpose=np.r_[0., 0., v * self.horizon, 0., 0., 0.], info={'fz': fz, 'v': v})


def summarize(ep, f_floor):
    """Per-episode numbers: outcome, and how the z-force, the offset and the target behaved."""
    s = {'episode': ep.index, 'success': ep.success, 'end': ep.end, 'aborted': ep.aborted,
         'duration': round(ep.duration, 2), 'path': ep.path, 'timing': ep.timing}
    if ep.ticks:
        t, fz, dz = ep.ticks['t'], ep.ticks['filtered_force'][:, 2], ep.ticks['dpose'][:, 2]
        dt = np.diff(t, append=t[-1])
        over = fz > f_floor
        s.update(fz_max=round(float(fz.max()), 1), fz_p95=round(float(np.percentile(fz, 95)), 1),
                 s_over_floor=round(float(dt[over].sum()), 2),
                 fz_mean_over_floor=round(float(fz[over].mean()), 1) if over.any() else None,
                 dz_max_mm=round(1000 * float(dz.max()), 1))
    return s


def main():
    parser = argparse.ArgumentParser(description='Insert with a z-force-regulating residual, episode after episode.')
    ResidualEnv.add_args(parser)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--f_floor', type=float, default=5.0, help='z-force above which the correction acts (N)')
    parser.add_argument('--gain', type=float, default=1e-3, help='upward velocity per N over f_floor (m/s/N)')
    parser.add_argument('--max_speed', type=float, default=0.01,
                        help='cap on the upward velocity (m/s); pushes faster than this outrace it')
    parser.add_argument('--horizon', type=float, default=0.4,
                        help='offset = velocity * horizon: the time the policy takes to absorb an offset (s)')
    parser.add_argument('--period', type=float, default=0.02, help='decision period (s)')
    args = parser.parse_args()

    results = []
    with ResidualEnv.from_args(args) as env:
        for _ in range(args.episodes):
            env.register(ZForceRegulator(args.f_floor, args.gain, args.max_speed, args.horizon), period=args.period)
            ep = env.run()
            s = summarize(ep, args.f_floor)
            results.append(s)
            print(f'\nepisode {s["episode"]}: {"plugged in" if ep.success else "failed"} ({ep.end})  '
                  + '  '.join(f'{k} {s[k]}' for k in ('fz_max', 'fz_mean_over_floor', 's_over_floor', 'dz_max_mm')
                              if k in s))
            if args.log_dir is not None:
                with open(os.path.join(args.log_dir, 'zforce_regulator.json'), 'w') as f:
                    json.dump({'args': vars(args), 'episodes': results}, f, indent=2, default=float)

    n = len(results)
    print(f'\n{sum(r["success"] for r in results)}/{n} plugged in '
          f'({sum(r["aborted"] for r in results)} aborted)')


if __name__ == '__main__':
    main()
