"""
Split evaluation: a pickup and an insert policy, joined by the scripted motions of
teleoperation.py, in the phases scripts/split_dataset.py cut the demonstrations into:

    pickup     pickup policy, until its done score crosses --done_threshold while it
               commands the gripper closed (or DpadDown, as in teleop)
    transport  scripted: let the grasp settle, up 2 cm, to Teleoperation.port_pose
    insert     insert policy, until the network comes up (or DpadUp, as in teleop; with
               --insert_done, also when its done score crosses the threshold)
    unplug     scripted: as Teleoperation.unplug_and_release
    done       stop

Both policies run in their own worker processes on the real-time chunking pipeline of
eval_realtime.py, each with its own chunk buffer, so a chunk predicted across a phase
change is never executed. Square aborts, as always. With --log_dir, phases.json (phase
start/end times on the rawdata.h5 clock, what ended each, success) and
chunks_<phase>.npz are saved next to rawdata.h5.
"""
from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.eval.policy_worker import PolicyWorker
from agent.utils.robot_utils import interrupt, build_states
from agent.model.policy import DiffusionPolicy
from teleoperation import Teleoperation
from robot.env import URPose, GRIP_OPEN, GRIP_CLOSED
from robot import robot_execution
import numpy as np
import argparse
import json
import time
import os

POLICY_PHASES = ('pickup', 'insert')


class EvalSplits(EvalRealtimeChunking):
    def __init__(self, pickup_ckpt, insert_ckpt, device='cuda', insert_done=False, **kwargs):
        # The insert policy loads in its worker while the base class loads the pickup one
        # and homes the robot.
        insert_worker = PolicyWorker(insert_ckpt, device)
        insert_policy = DiffusionPolicy.from_checkpoint(insert_ckpt, 'cpu')
        insert_policy.eval()
        # The base class runs the pickup policy; its gripper settings configure the env.
        super().__init__(pickup_ckpt, device=device, **kwargs)
        if insert_policy.grip_stats != self.policy.grip_stats:
            print(f'Warning: gripper settings differ (pickup {self.policy.grip_stats}, '
                  f'insert {insert_policy.grip_stats}); using the pickup policy\'s.')

        self.ckpts = {'pickup': str(pickup_ckpt), 'insert': str(insert_ckpt)}
        self.policies = {'pickup': (self.worker, self.policy), 'insert': (insert_worker, insert_policy)}
        self.buffers = {
            'pickup': self.buffer,
            'insert': RealtimeActionChunkingBuffer(action_dt=1.0 / insert_policy.framerate,
                                                   weight_decay=self.buffer.weight_decay,
                                                   taper=self.buffer.taper),
        }
        self.insert_done = insert_done
        self.phase = 'pickup'
        self.phase_t_start = time.time()
        self.phase_log = []

    # ============================================================
    # Phases
    # ============================================================
    def set_phase(self, phase, reason=None):
        """Enter `phase`; `reason` is what ended the current one."""
        now = time.time() - self.env.t0
        if self.phase_log:
            self.phase_log[-1].update(t_end=now, end=reason)
        self.phase_log.append({'phase': phase, 't_start': now})
        self.phase = phase
        self.phase_t_start = time.time()
        if phase in POLICY_PHASES:
            # self.worker / policy / buffer are what the base class's get_action helpers,
            # runtime_info and save_chunks read
            self.worker, self.policy = self.policies[phase]
            self.buffer = self.buffers[phase]
            if self.lowpass:
                self.lowpass.reset()
        print(f'\n[{now:6.1f} s] {phase}' + (f'  ({reason})' if reason else ''))

    def start_transport(self, reason):
        """As Teleoperation.move_to_port, after the grasp settles (teleop operators waited
        ~0.7 s after closing before DpadDown). Returns the sequence's first action."""
        self.set_phase('transport', reason)
        seq = interrupt(self)
        seq.gripper(GRIP_CLOSED)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=.05)
        seq.move_to(Teleoperation.port_pose) \
            .then(lambda _: self.set_phase('insert', 'at port'))
        return self.get_action()

    def start_unplug(self, reason):
        """As Teleoperation.unplug_and_release, then stop. Returns the sequence's first action."""
        self.set_phase('unplug', reason)
        seq = interrupt(self)
        seq.release()               # let go of the push before the fingers let go of the plug
        seq.gripper(GRIP_OPEN)
        seq.move_to(Teleoperation.unplug_pose)
        seq.gripper(GRIP_CLOSED)
        seq.move_relative([0, 0, .03, 0, 0, 0], speed=.05)

        r1, r2 = Teleoperation.release_pose1, Teleoperation.release_pose2
        rel = np.array(r1)
        # rel[:2] = np.random.uniform([r1.x, r1.y], [r2.x, r2.y])
        rel[:2] = [(r1.x + r2.x) / 2, (r1.y + r2.y) / 2]
        seq.move_to(URPose(*rel))
        seq.gripper(GRIP_OPEN)
        seq.wait(1) \
            .then(lambda _: self.set_phase('done', 'released')) \
            .then(lambda _: self.stop())
        return self.get_action()

    # ============================================================
    # Control (only reached in policy phases: an InterruptSequence shadows get_action
    # during the scripted ones)
    # ============================================================
    def get_action(self):
        buttons = self.iface.dualsense.state
        if self.phase == 'pickup' and buttons.DpadDown:
            return self.start_transport('DpadDown')
        if self.phase == 'insert':
            if self.env.network_status:
                return self.start_unplug('network')
            if buttons.DpadUp:
                return self.start_unplug('DpadUp')

        act = self.policy_action()
        if act is None:
            return None
        des_pose, des_width, done, log_gains = act
        if self.policy.predict_done and done > self.done_threshold:
            if self.phase == 'pickup' and des_width > 0.5:      # commanding GRIP_CLOSED
                return self.start_transport(f'done={done:.2f}')
            if self.phase == 'insert' and self.insert_done:
                return self.start_unplug(f'done={done:.2f}')
        return self.command(des_pose, des_width, done, log_gains)

    def command(self, des_pose, des_width, done, log_gains):
        """A policy phase's command from the policy's action: apply its gains, return (pose,
        width). A hook (eval_residual.py corrects the insert policy's here)."""
        self.apply_gains(log_gains)
        return des_pose, des_width

    def policy_action(self):
        """EvalRealtimeChunking.get_action without its stop on done and without applying the
        gains: (pose, width, done, {field: (6,) log10 gains}) or None."""
        if self.buffer.is_empty():
            return None
        now = time.time()
        act = self.buffer.get_action(now + self.control_dt, t_now=now,
                                     pose_lead=self.lowpass.lead if self.lowpass else 0.0)
        if act is None:
            return None
        des_pose, des_width, done, log_gains = act
        if self.lowpass:
            des_pose = self.lowpass(des_pose, now)
        return des_pose, des_width, done, {f: log_gains[6 * i:6 * (i + 1)]
                                           for i, f in enumerate(self.policy.impedance_fields)}

    def prediction_loop(self):
        """EvalRealtimeChunking.prediction_loop for the current phase's policy; idle in scripted phases."""
        while not self.stop_event.is_set():
            phase = self.phase
            if phase not in POLICY_PHASES:
                time.sleep(0.01)
                continue
            worker, policy = self.policies[phase]
            buffer = self.buffers[phase]
            t_obs = time.time()
            # history from this phase only: training episodes are split by phase, and pad
            # the history at their start
            obs_deque = self.obs_history.frames(self.env.get_obs(), t_obs, policy.obs_horizon,
                                                1.0 / policy.framerate, t_min=self.phase_t_start)
            des_poses, des_grips, des_done, des_gains = worker.get_actions(obs_deque)
            if self.phase != phase:
                continue        # predicted across a phase change: never executed
            des_gains = np.concatenate([des_gains[f] for f in policy.impedance_fields], axis=1) \
                if des_gains else None
            start = 0   # action 0 is t_obs's step (see EvalRealtimeChunking.prediction_loop)
            end = start + policy.action_horizon
            chnk = buffer.add_chunk(
                t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end],
                None if des_gains is None else des_gains[start:end])
            buffer.dolog(chnk, build_states(obs_deque, policy.obs_fields), time.time())  # for offline logging

    # ============================================================
    # Run loop hooks
    # ============================================================
    def pre_run(self):
        super().pre_run()           # waits for Circle and the pickup worker, starts predicting
        self.policies['insert'][0].wait_ready()
        self.phase_log.append({'phase': 'pickup', 't_start': time.time() - self.env.t0})

    def runtime_info(self):
        print(f'{self.phase:9s} ', end='')
        super().runtime_info()

    def close(self):
        self.end_episode()
        for worker, _ in self.policies.values():
            worker.close()

    def end_episode(self):
        """Stop the robot and save the episode: rawdata.h5, phases.json, chunks_<phase>.npz.
        Returns whether it plugged in."""
        if self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=2.0)
        robot_execution.RobotExecution.close(self)     # Env.close saves rawdata.h5 into env.epi_path

        if self.phase_log and 't_end' not in self.phase_log[-1] and self.phase != 'done':
            self.phase_log[-1].update(t_end=time.time() - self.env.t0, end='aborted')
        success = any(p['phase'] == 'insert' and p.get('end') == 'network' for p in self.phase_log)
        print(f'\nPhases: ' + ' -> '.join(f"{p['phase']} ({p.get('end')})" for p in self.phase_log))
        print(f'Plugged in: {"yes" if success else "no"}')
        if self.env.dataset_path is None:
            return success
        with open(self.env.epi_path / 'phases.json', 'w') as f:
            json.dump({'ckpts': self.ckpts, 'success': success, 'phases': self.phase_log}, f, indent=2)
        # EvalRealtimeChunking.save_chunks writes self.buffer's chunks as chunks.npz
        for phase in POLICY_PHASES:
            self.buffer, (self.worker, self.policy) = self.buffers[phase], self.policies[phase]
            self.save_chunks()
            if (self.env.epi_path / 'chunks.npz').exists():
                os.replace(self.env.epi_path / 'chunks.npz', self.env.epi_path / f'chunks_{phase}.npz')
        return success


def parse_args():
    parser = argparse.ArgumentParser(description='Split (pickup -> transport -> insert -> unplug) evaluation.')
    parser.add_argument('--pickup_ckpt', type=str, help='pickup policy checkpoint',
                        default='../ckpts/ethernet-impedance-v2v3-pickup-tgt-noforce/ckpt_final.pth')
    parser.add_argument('--insert_ckpt', type=str, required=True, help='insert policy checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=100.0,
                        help='command frequency (Hz) of the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('--taper', type=float, default=0.15,
                        help='seconds over which a chunk\'s weight ramps in and out (0 = off)')
    parser.add_argument('--lpf', type=float, default=3.0,
                        help='cutoff (Hz) of the low-pass on the commanded pose (0 = off)')
    parser.add_argument('--done_threshold', type=float, default=0.5,
                        help='done score that ends pickup (and insert, with --insert_done)')
    parser.add_argument('--insert_done', action='store_true', default=False,
                        help='also end insert on the insert policy\'s done score, not only on the network')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    evaluation = EvalSplits(
        pickup_ckpt=args.pickup_ckpt,
        insert_ckpt=args.insert_ckpt,
        insert_done=args.insert_done,
        log_dir=args.log_dir,
        control_freq=args.control_freq,
        weight_decay=args.weight_decay,
        taper=args.taper,
        lpf_hz=args.lpf,
        done_threshold=args.done_threshold,
        device=args.device,
    )
    evaluation.run()
