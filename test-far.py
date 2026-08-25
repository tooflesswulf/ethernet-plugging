from agent.evaluate.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import interrupt, build_states
from collections import deque
import numpy as np
import argparse
import threading
import time
import os

from util import URPose
from env import GRIP_OPEN, GRIP_CLOSED
from FAR import tta_online


class StreamingForceEdge:
    """Online detector for a sharp rise in a noisy force signal, one sample per tick.

    Keeps a long trailing baseline window and a short reaction window; flags a rise
    when the reaction mean exceeds the baseline mean by `k` times the baseline's own
    std, so the threshold auto-scales to the current noise level. Refractory gating
    prevents repeated triggers on a single edge.
    """

    def __init__(self, hz=20, baseline_s=1.0, react_s=0.15, k=6.0, refractory_s=0.5):
        self.b = max(int(baseline_s * hz), 1)
        self.r = max(int(react_s * hz), 1)
        self.k = k
        self.refractory = int(refractory_s * hz)
        self.buf = deque(maxlen=self.b + self.r)
        self.cooldown = 0

    def update(self, value):
        """Feed one sample; return True on the tick a rising edge is detected."""
        self.buf.append(float(value))
        if self.cooldown > 0:
            self.cooldown -= 1
        if len(self.buf) < self.b + self.r:
            return False
        window = np.asarray(self.buf)
        base, cur = window[:self.b], window[self.b:]
        sigma = base.std() + 1e-6
        if self.cooldown == 0 and (cur.mean() - base.mean()) / sigma >= self.k:
            self.cooldown = self.refractory
            return True
        return False


class TeleoperationReset(EvalRealtimeChunking):
    # cable_drop_pos = URPose(-.0562, .6679, .0456, 2.508, 2.524, .936)
    cable_drop_pos = URPose(-0.03938359, 0.64969687, 0.07542422, -1.77502314, -1.78634705, -0.66244883)

    # Failed plugins wrench the cable out of the grippers: the plug catches on the
    # socket rim (first contact at z=68.6-71.1mm vs 55.5-64.0mm when it enters the
    # socket cleanly) and force z then ramps past ~42N before the cable slips, while
    # successful plugins stay under ~33N until release. The 38N trigger is therefore
    # gated on a rim-height first contact, so a clean insertion can push harder than
    # 38N without triggering; a 55N backstop covers wrench-out from a low contact.
    # Only applies while gripping the cable (~9mm); the unplug phase grips the plug
    # head (~16mm) and legitimately reaches 55-75N, and an empty gripper closes to ~5mm.
    FZ_THRESH_N = 38.0            # force z above episode-start baseline, rim contact
    FZ_BACKSTOP_N = 55.0          # trigger regardless of contact height
    FZ_TICKS = 3                  # consecutive control ticks above threshold
    CONTACT_RISE_N = 20.0         # force z rise marking first contact (2 ticks)
    CONTACT_Z_M = 0.0665          # first contact above this = rim hit, below = in socket
    CABLE_WIDTH_MM = (6.5, 12.0)  # gripper width range when holding the cable

    RETREAT_LOOKBACK_S = 2.0      # how far back (elapsed time) to retreat on failure
    RETREAT_NUM_WAYPOINTS = 10     # waypoints used to retrace that window in reverse
    RETREAT_SPEED = 0.03          # m/s; slow, since this retraces near the socket
    STEP_BACK_INCREMENT_S = 5.0   # trajectory retraced per manual Dpad-Up step

    _fz_baseline = None
    _contact_z = None
    _fz_count = 0
    _armed = True
    _force_edge = None
    _fz_cursor = 0
    _contact_flag = False
    _contact_t = -1

    def __init__(self, enable_tta=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._force_edge = StreamingForceEdge(hz=self.env.servo_frequency)
        # Test-time adaptation: rolling record of the policy's own recent
        # (obs, action) predictions, and the async machinery to fine-tune on
        # them without blocking the control loop or racing the prediction
        # thread. See FAR/tta_online.py.
        self._tta_recorder = tta_online.TTATrajectoryRecorder(maxlen=200)
        self._tta_thread = None
        self._tta_pause = threading.Event()  # set while a TTA update is training
        self.enable_tta = enable_tta

    def prediction_loop(self):
        """Overrides EvalRealtimeChunking.prediction_loop to additionally
        record each tick's (obs conditioning, raw normalized action chunk)
        into self._tta_recorder for later TTA updates, and to pause while a
        TTA update is training (it mutates noise_pred_net's weights, which
        this loop reads concurrently).
        """
        action_horizon = self.policy.action_horizon
        obs_horizon = self.policy.obs_horizon

        obs_deque = deque(maxlen=obs_horizon)
        while not self.stop_event.is_set():
            if self._tta_pause.is_set():
                time.sleep(0.05)
                continue

            t_obs = time.time()
            obs_deque.append(self.env.get_obs())
            if len(obs_deque) < obs_horizon:
                continue

            des_poses, des_grips, des_done, conditions, naction = tta_online.get_actions_with_naction(
                self.policy, obs_deque, self.device)
            self._tta_recorder.record(conditions, naction)

            start = obs_horizon - 1
            end = start + action_horizon
            chnk = self.buffer.add_chunk(
                t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end])
            obs_state = build_states(obs_deque, self.policy.obs_fields)
            self.buffer.dolog(chnk, obs_state, time.time())

    def trigger_tta_update(self):
        """Fire a background TTA update on the trajectory recorded since the
        last update, using the current failure as the negative signal. Runs
        in its own thread so reset_cable()'s interrupt sequence stays
        non-blocking; pauses prediction_loop for the duration so training
        doesn't race live inference on the same weights.
        """
        if self._tta_thread is not None and self._tta_thread.is_alive():
            print('TTA update already running, skipping.')
            return
        recorder, self._tta_recorder = self._tta_recorder, tta_online.TTATrajectoryRecorder(maxlen=200)
        if len(recorder) == 0:
            return

        def _run():
            self._tta_pause.set()
            try:
                tta_online.do_tta_update(self.policy, recorder, device=self.device, num_negatives=10, num_candidates=8, train_steps=5, lr=1e-5, beta=0.1, bc_weight=1.0)
            finally:
                self._tta_pause.clear()

        self._tta_thread = threading.Thread(target=_run, daemon=True)
        self._tta_thread.start()

    def detect_force_edge(self):
        # robot_obs is appended by the receive thread at servo_frequency (~500Hz),
        # much faster than this control loop, so drain every sample since last tick.
        robot_obs = self.env.robot_obs
        n = len(robot_obs)  # snapshot; the receive thread may append concurrently
        flag = False
        for obs in robot_obs[self._fz_cursor:n]:
            if self._force_edge.update(obs.actual_force[2]):
                flag = True
        self._fz_cursor = n
        return flag

    def _retreat_waypoints(self, lookback_s, num_waypoints):
        """Most-recent-first list of URPoses sampled from the last `lookback_s`
        seconds of self.env.robot_obs, for retracing the approach in reverse
        instead of jumping straight back to a fixed pose (safer near the
        socket, since it follows the path the arm actually took). Empty if
        there isn't `lookback_s` seconds of history yet.
        """
        robot_obs = self.env.robot_obs
        if not robot_obs:
            return []
        t_start = robot_obs[-1].time - lookback_s
        window = []
        for obs in reversed(robot_obs):
            if obs.time < t_start:
                break
            window.append(obs)
        if len(window) <= 1:
            return []
        # window[0] is ~now; skip it and space the rest evenly back to t_start.
        # (np.linspace(..., num=1) returns just the range's start, not its end,
        # so num_waypoints=1 is special-cased to still land on the oldest pose.)
        last_idx = len(window) - 1
        if num_waypoints <= 1:
            idxs = [last_idx]
        else:
            idxs = np.linspace(1, last_idx, num=min(num_waypoints, last_idx), dtype=int)
        return [window[i].actual_pose for i in idxs]

    def get_action(self):
        if self.detect_force_edge():
            last_pose, last_grip, _, _ = self.last_action
            if last_grip == GRIP_CLOSED:
                print('Detected rising force edge while gripper closed. Setting 2s timeout')
                self._contact_flag = True
                self._contact_t = time.time()

        if self.iface.dualsense.state.DpadLeft:
            last_pose, last_grip, _, _ = self.last_action
            if last_grip == GRIP_CLOSED:
                return self.reset_cable()
            print('Dpad-Left pressed, but gripper is open. Ignoring.')
        if self.iface.dualsense.state.DpadUp:
            return self.step_back()
        if self.iface.dualsense.state.DpadDown:
            return self.go_home()

        act = super().get_action()
        if act is not None and act[1] == GRIP_OPEN:
            self._contact_flag = False
            self._contact_t = -1
        if self._contact_flag:
            if time.time() - self._contact_t > 2:
                print('Contact timeout: shouldve been done plugging by now. Resetting.')
                self._contact_flag = False
                self._contact_t = -1
                return self.reset_cable()
        return act

    def runtime_info(self):
        obs = self.last_obs
        print(f"force_z: {obs['state']['filtered_force'][2]:7.2f}", end='\r')

    def reset_cable(self):
        # Start interrupt sequence. The seq methods queue up instructions behind the scenes,
        #   so custom logic needs to be 1. run through promise.then() and 2. be non-blocking.
        print('Starting cable reset sequence.')
        if self.enable_tta:
            self.trigger_tta_update()
        # Clear force-edge/contact-timeout state: normally this happens when the
        # reset opens the gripper, but a retreat keeps holding the cable, so it
        # needs to be done explicitly here.
        self._contact_flag = False
        self._contact_t = -1
        self._force_edge = StreamingForceEdge(hz=self.env.servo_frequency)

        waypoints = self._retreat_waypoints(self.RETREAT_LOOKBACK_S, self.RETREAT_NUM_WAYPOINTS)
        seq = interrupt(self)
        if not waypoints:
            print('No recent trajectory to retreat through; falling back to a full reset.')
            seq.move_relative([0, 0, .02, 0, 0, 0], speed=0.05)
            seq.move_to(self.cable_drop_pos)
            seq.gripper(GRIP_OPEN, settle_time=1.0)
            last = seq.move_to(self.home_pose)
        else:
            for pose in waypoints:
                last = seq.move_to(pose, speed=self.RETREAT_SPEED)
        last.then(lambda _: self.buffer.clear()) \
            .then(lambda _: self.env.request_zero_ft())
        return self.get_action()

    def step_back(self):
        """Manual retreat (Dpad-Up): steps back through recent history one
        small increment (STEP_BACK_INCREMENT_S) at a time, rather than the
        fixed multi-second retreat reset_cable() does on a detected failure.
        Number of steps back is entirely up to the operator: one tap moves
        one increment; holding the button re-triggers this every tick the
        prior increment's motion finishes on, so it keeps stepping back for
        as long as it's held. No TTA update -- no failure signal to learn
        from here, just manual repositioning.
        """
        waypoints = self._retreat_waypoints(self.STEP_BACK_INCREMENT_S, num_waypoints=1)
        if not waypoints:
            print('No recent trajectory to step back through.')
            return None  # repeat last action; nothing queued, so get_action isn't shadowed
        print('Manual step-back.')
        seq = interrupt(self)
        seq.move_to(waypoints[0], speed=self.RETREAT_SPEED)
        return self.get_action()

    def go_home(self):
        seq = interrupt(self)
        seq.move_to(self.home_pose) \
            .then(lambda _: self.buffer.clear()) \
            .then(lambda _: self.env.request_zero_ft())
        return self.get_action()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('--log', type=str, default=None, help='log directory')
    parser.add_argument('--tta', action='store_true', help='enable test-time adaptation (TTA) on failure')
    args = parser.parse_args()

    if args.log is not None:
        os.makedirs(args.log, exist_ok=True)
    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=args.log,
        control_freq=args.control_freq, weight_decay=args.weight_decay, enable_tta=args.tta
    )
    teleop.run()
