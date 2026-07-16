from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import interrupt
from collections import deque
import numpy as np
import argparse
import time
import os

from util import URPose
from env import GRIP_OPEN, GRIP_CLOSED


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

    _fz_baseline = None
    _contact_z = None
    _fz_count = 0
    _armed = True
    _force_edge = None
    _fz_cursor = 0
    _contact_flag = False
    _contact_t = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._force_edge = StreamingForceEdge(hz=self.env.servo_frequency)

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
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=0.05)
        seq.move_to(self.cable_drop_pos)
        seq.gripper(GRIP_OPEN, settle_time=1.0)
        seq.move_to(self.home_pose) \
            .then(lambda _: self.buffer.clear()) \
            .then(lambda _: self.env.ctrl.zeroFtSensor())
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
    args = parser.parse_args()

    if args.log is not None:
        os.makedirs(args.log, exist_ok=True)
    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=args.log,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
