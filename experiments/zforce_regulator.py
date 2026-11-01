"""
Hand-written residual for the insert policy: keep the z-force low while it searches for
the port, without stopping the insertion push.

    python experiments/zforce_regulator.py --insert_ckpt <ckpt> --log_dir <dir>
"""
from agent.eval.eval_residual import ResidualEnv, ResidualActor, Correction
import numpy as np
import argparse
import json
import os


class ZForceRegulator(ResidualActor):
    def __init__(self, f_floor=5.0, gain=3e-3, max_speed=0.015, horizon=0.4):
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
    parser.add_argument('--gain', type=float, default=3e-3, help='upward velocity per N over f_floor (m/s/N)')
    parser.add_argument('--max_speed', type=float, default=0.015,
                        help='cap on the upward velocity (m/s); pushes faster than this outrace it')
    parser.add_argument('--horizon', type=float, default=0.4,
                        help='offset = velocity * horizon: the time the policy takes to absorb an offset (s)')
    parser.add_argument('--period', type=float, default=0.02, help='decision period (s)')
    args = parser.parse_args()

    last_successful = False
    results = []
    with ResidualEnv.from_args(args) as env:
        for _ in range(args.episodes):
            env.register(ZForceRegulator(args.f_floor, args.gain, args.max_speed, args.horizon), period=args.period)
            env.wait_circle = not last_successful
            ep = env.run()
            last_successful = ep.success
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
