from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer, CommandLowpass
from agent.eval.policy_worker import PolicyWorker
from agent.utils.robot_utils import build_states, wait_for_circle, apply_gains
from agent.dataset.sequence import GripperStats
from agent.model.policy import DiffusionPolicy
import robot_execution
import collections
import numpy as np
import threading
import argparse
import time
import os


class EvalRealtimeChunking(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', log_dir=None, control_freq=100.0, weight_decay=0.5, taper=0.15,
                 lpf_hz=3.0, done_threshold=0.5):
        # Inference runs in its own process (see policy_worker); it loads the policy while
        # the robot homes. This CPU copy only answers config questions (fields, horizons).
        self.worker = PolicyWorker(ckpt, device)
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, 'cpu')
        self.policy.eval()
        self.device = device
        # End the episode once the policy's predicted completion score crosses this.
        self.done_threshold = done_threshold
        grip = GripperStats(*self.policy.grip_stats)

        # Commands go out at control_freq, independent of the policy's framerate: the buffer
        # spaces each chunk's actions at 1 / framerate and interpolates them in time. At
        # 20 Hz the env's linear blend made the target velocity step every 50 ms, and fdcc
        # fed each step forward (teleop commands at 100 Hz).
        if control_freq is None:
            control_freq = self.policy.framerate

        # super().__init__() resets & starts the robot.
        super().__init__(
            path=log_dir,
            control_freq=control_freq,
            gwidth=grip.grip_width_mm,
            gforce=grip.grip_force_n,
            gspeed=grip.grip_speed_mmps,
            gpullback=grip.grip_pullback_mm,
        )

        self.buffer = RealtimeActionChunkingBuffer(action_dt=1.0 / self.policy.framerate,
                                                   weight_decay=weight_decay, taper=taper)
        # Low-pass on the commanded pose (None = off); the buffer is read its delay ahead
        self.lowpass = CommandLowpass(lpf_hz) if lpf_hz else None
        self.prediction_thread = threading.Thread(target=self.prediction_loop)

    def pre_run(self):
        wait_for_circle(self.env, self.iface, close_gripper=False)
        self.worker.wait_ready()
        print("Starting real-time chunked evaluation loop...")
        self._t_pred_start = time.time()

        self.prediction_thread.start()

    def runtime_info(self):
        zf = self.last_obs['state']['filtered_force']
        kz = f'  Kz: {self.env.imp.K[2]:6.0f}' if self.policy.impedance_fields else ''
        rate = self.buffer._chunk_count / max(time.time() - self._t_pred_start, 1e-3)    # chunks/s
        print(f'{rate:5.1f} chunks/s  zforce: {zf[2]:.05f}{kz}', end='\r')

    def get_action(self):
        if self.buffer.is_empty():
            return None
        # Env.step blends from the previous command to this one over control_dt, so this
        # command is only reached a period from now: ask for the action due then.
        now = time.time()
        act = self.buffer.get_action(now + self.control_dt, t_now=now,
                                     pose_lead=self.lowpass.lead if self.lowpass else 0.0)
        if act is None:
            return None
        des_pose, des_width, done, log_gains = act
        if self.lowpass:
            des_pose = self.lowpass(des_pose, now)
        # Predicted impedance, split back into fields and blended over one command period
        # (no-op without an impedance head)
        apply_gains(self.env, {f: log_gains[6 * i:6 * (i + 1)]
                               for i, f in enumerate(self.policy.impedance_fields)},
                    ramp_s=self.control_dt)
        # End-of-episode signal: stop once the executed action's done score crosses the threshold.
        if self.policy.predict_done and done > self.done_threshold:
            print(f"Policy thinks the task is complete (done={done:.3f} > threshold={self.done_threshold:.3f}).")
            self.stop()
        return des_pose, des_width

    def prediction_loop(self):
        action_horizon = self.policy.action_horizon
        obs_horizon = self.policy.obs_horizon

        obs_deque = collections.deque(maxlen=obs_horizon)
        while not self.stop_event.is_set():
            t_obs = time.time()  # observation time the chunk is anchored to
            obs_deque.append(self.env.get_obs())
            if len(obs_deque) < obs_horizon:
                continue

            # get_actions (in the worker process) builds images + the obs_fields state vector
            # from the deque; waiting on it releases the GIL for the control loop.
            des_poses, des_grips, des_done, des_gains = self.worker.get_actions(obs_deque)
            # (H, 6 * n_fields) log10 gains, in policy.impedance_fields order
            des_gains = np.concatenate([des_gains[f] for f in self.policy.impedance_fields], axis=1) \
                if des_gains else None

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs.
            # The done score rides through the buffer so it's ensembled and thresholded at
            # execution time in get_action (not averaged over the chunk here).
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = self.buffer.add_chunk(
                t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end],
                None if des_gains is None else des_gains[start:end])
            obs_state = build_states(obs_deque, self.policy.obs_fields)  # for offline logging
            self.buffer.dolog(chnk, obs_state, time.time())

    def close(self):
        if self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=2.0)
        self.worker.close()
        super().close()     # Env.close saves rawdata.h5 into env.epi_path
        self.save_chunks()

    def save_chunks(self):
        """Every predicted chunk, for offline debugging: chunks.npz next to rawdata.h5."""
        logs = self.buffer._logs
        if self.env.dataset_path is None or not logs:
            return
        c = [l['chunk'] for l in logs]
        np.savez_compressed(
            self.env.epi_path / 'chunks.npz',
            t_obs=np.array([x.t_obs for x in c]) - self.env.t0,   # same clock as rawdata.h5 times
            t_add=np.array([l['t'] for l in logs]) - self.env.t0,
            poses=np.stack([x.poses for x in c]), widths=np.stack([x.widths for x in c]),
            dones=np.stack([x.dones for x in c]), log_gains=np.stack([x.gains for x in c]),
            obs=np.stack([l['obs'] for l in logs]), obs_fields=np.array(self.policy.obs_fields))


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=100.0,
                        help='command frequency (Hz) of the real-time loop; chunks are interpolated in '
                             "time, so it need not match the policy's framerate")
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('--taper', type=float, default=0.15,
                        help='seconds over which a chunk\'s weight ramps in after it arrives and out '
                             'before its horizon ends (0 = off)')
    parser.add_argument('--lpf', type=float, default=3.0,
                        help='cutoff (Hz) of the low-pass on the commanded pose, delay-compensated (0 = off)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    evaluation = EvalRealtimeChunking(
        ckpt=args.ckpt,
        log_dir=args.log_dir,
        control_freq=args.control_freq,
        weight_decay=args.weight_decay,
        taper=args.taper,
        lpf_hz=args.lpf,
        device=args.device,
    )
    evaluation.run()
