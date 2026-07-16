from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import interrupt
from collections import deque
import numpy as np
import argparse
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
    cable_drop_pos = URPose(-0.04938359, 0.64969687, 0.07542422, -1.77502314, -1.78634705, -0.66244883)

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
    _contact_count = 0
    _fz_count = 0
    _armed = True
    _force_edge = None
    _fz_cursor = 0

    def detect_force_edge(self):
        # robot_obs is appended by the receive thread at servo_frequency (~500Hz),
        # much faster than this control loop, so drain every sample since last tick.
        robot_obs = self.env.robot_obs
        if self._force_edge is None:
            self._force_edge = StreamingForceEdge(hz=self.env.servo_frequency)
        n = len(robot_obs)  # snapshot; the receive thread may append concurrently
        for obs in robot_obs[self._fz_cursor:n]:
            if self._force_edge.update(obs.actual_force[2]):
                print('Detected rising force edge')
        self._fz_cursor = n

    def get_action(self):
        self.detect_force_edge()
        if self.iface.dualsense.state.DpadLeft:
            last_pose, last_grip, _, _ = self.last_action
            if last_grip == GRIP_CLOSED:
                return self.reset_cable()

            print('Dpad-Left pressed, but gripper is open. Ignoring.')
        if self.detect_wrench_risk():
            print('Force z spike while gripping cable: failed plugin, resetting before wrench-out.')
            return self.reset_cable()
        return super().get_action()

    def detect_wrench_risk(self):
        obs = getattr(self, 'last_obs', None)
        if obs is None:
            return False
        fz = obs['state']['actual_force'][2]
        width = obs['state']['gripper_width']
        z = obs['state']['actual_pose'][2]
        if self._fz_baseline is None:
            self._fz_baseline = fz
            return False
        fz_rel = fz - self._fz_baseline

        _, last_grip, _, _ = self.last_action
        holding_cable = (last_grip == GRIP_CLOSED
                         and self.CABLE_WIDTH_MM[0] < width < self.CABLE_WIDTH_MM[1])
        if not holding_cable:
            # Cable left the gripper (release, wrench-out, or reset): re-arm for the
            # next grasp and forget this attempt's contact point.
            self._armed = True
            self._contact_z = None
            self._contact_count = 0
            self._fz_count = 0
            return False
        if not self._armed:
            return False

        # Latch the height of first contact for this attempt.
        if self._contact_z is None:
            self._contact_count = self._contact_count + 1 if fz_rel > self.CONTACT_RISE_N else 0
            if self._contact_count >= 2:
                self._contact_z = z

        rim_hit = self._contact_z is not None and self._contact_z > self.CONTACT_Z_M
        thresh = self.FZ_THRESH_N if rim_hit else self.FZ_BACKSTOP_N
        self._fz_count = self._fz_count + 1 if fz_rel > thresh else 0

        if self._fz_count >= self.FZ_TICKS:
            self._armed = False
            self._fz_count = 0
            return True
        return False

    def reset_cable(self):
        # Start interrupt sequence. The seq methods queue up instructions behind the scenes,
        #   so custom logic needs to be 1. run through promise.then() and 2. be non-blocking.
        print('Starting cable reset sequence.')
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=0.05)
        seq.move_to(self.cable_drop_pos)
        seq.gripper(GRIP_OPEN, settle_time=1.0)
        seq.move_to(self.home_pose) \
           .then(lambda _: self.buffer.clear())
        return self.get_action()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    args = parser.parse_args()

    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=None,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
