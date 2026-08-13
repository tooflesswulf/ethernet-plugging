"""Teleoperation with a live state estimator overlay.

Same driving controls as teleoperation-qwen.py, but instead of the grasp cues it
draws the current task-state (from the attached state machine) as text on the
display image while you teleoperate.

State machine (green = normal, red = error, yellow = decision):

    Start -> Idle
    Idle        --Grasp--   Cable? -- yes -> Approaching
                                    -- no  -> Pickup error
    Pickup error --Release-> Idle
    Approaching --Release-- Network? -- yes -> Inserted
                                     -- no  -> Plugin Error
    Inserted    --Grasp--   Cable? -- yes -> Returning
                                    -- no  -> Inserted   (always retry)
    Returning   --Release-> Idle

Decision sources:
    Cable?   -> a grasp detector over the wrist-cam image. Default is the hand-tuned
                GraspDetector (agent/model/grasp.py); pass --grasp_ckpt to use the
                trained classifier instead (agent/model/grasp_cls.py, trained by
                agent/pretrain/train_grasp.py). Both expose the same debounced
                `update(rgb) -> bool`, so the state machine is unchanged.
    Network? -> self.last_obs['network_status'].

Ignored for now (per spec): the Approaching -> Plugin Error edge
(cable lost / force threshold).
"""
from enum import Enum
import argparse
import os
import time

import cv2
import numpy as np

from agent.utils.robot_utils import interrupt
from agent.model.grasp import GraspDetector
from agent.model.grasp_cls import NeuralGraspDetector
import robot_execution
from env import URPose, GRIP_OPEN, GRIP_CLOSED

GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10


class State(Enum):
    """Nodes of the teleop state machine. `.value` is the on-screen label."""
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    INSERTED = "INSERTED"
    RETURNING = "RETURNING"
    PICKUP_ERROR = "PICKUP ERROR"
    PLUGIN_ERROR = "PLUGIN ERROR"


ERROR_STATES = {State.PICKUP_ERROR, State.PLUGIN_ERROR}


class StateEstimator:
    """Tracks the task state from gripper Grasp/Release events + two sensors.

    The diagram's `Grasp` / `Release` edges are the commanded gripper transitions
    (open<->closed). At each such event the relevant yellow decision node is
    evaluated: `Cable?` from the grasp detector, `Network?` from network_status.

    A grasp/release physically takes a moment to seat the plug (and the detector
    is debounced), so a decision that could latch an error is given a short settle
    window: we take the success branch as soon as the sensor confirms, and only
    fall to the failure branch if it hasn't confirmed by SETTLE_S. An opposite
    gripper event during the window aborts the pending decision.
    """

    # Grace after the triggering gripper event before trusting a decision sensor.
    SETTLE_S = 2.0

    def __init__(self):
        self.state = State.IDLE
        self._prev_grip = GRIP_OPEN
        self._pending = None  # (kind, start_time) | None

    def update(self, gripper, held, network):
        """Advance the FSM one tick and return the current state.

        gripper: commanded gripper state this tick (GRIP_OPEN / GRIP_CLOSED).
        held:    GraspDetector result -- answers `Cable?`.
        network: self.last_obs['network_status'] -- answers `Network?`.
        """
        now = time.time()
        grasp = gripper == GRIP_CLOSED and self._prev_grip != GRIP_CLOSED
        release = gripper == GRIP_OPEN and self._prev_grip != GRIP_OPEN
        self._prev_grip = gripper

        s = self.state

        # --- arm / drive decisions on the triggering gripper event ---
        if s == State.IDLE and grasp:
            self._pending = ("pickup", now)          # Cable? -> Approaching / Pickup error
        elif s == State.INSERTED and grasp:
            self._pending = ("unplug", now)         # Cable? -> Returning / (stay Inserted)
        elif s == State.APPROACHING and release:
            self._pending = ("network", now)         # Network? -> Inserted / Plugin Error
        elif s == State.RETURNING and release:
            self.state = State.IDLE
        elif s == State.PICKUP_ERROR and release:
            self.state = State.IDLE
        # PLUGIN_ERROR is terminal (no outgoing edge in the diagram).

        # An opposite gripper event cancels a still-pending decision.
        if self._pending is not None:
            kind = self._pending[0]
            if kind in ("pickup", "unplug") and release:
                self._pending = None
            elif kind == "network" and grasp:
                self._pending = None

        # --- resolve the pending decision (as soon as it confirms, else at SETTLE_S) ---
        if self._pending is not None:
            kind, t0 = self._pending
            elapsed = now - t0
            if kind == "pickup":
                if held:
                    self.state = State.APPROACHING
                    self._pending = None
                elif elapsed >= self.SETTLE_S:
                    self.state = State.PICKUP_ERROR
                    self._pending = None
            elif kind == "unplug":
                if held:
                    self.state = State.RETURNING
                    self._pending = None
                elif elapsed >= self.SETTLE_S:
                    self._pending = None             # Cable? no -> stay Inserted, retry
            elif kind == "network":
                if network:
                    self.state = State.INSERTED
                    self._pending = None
                elif elapsed >= self.SETTLE_S:
                    self.state = State.PLUGIN_ERROR
                    self._pending = None

        return self.state


class Teleoperation(robot_execution.RobotExecution):
    port_pose = URPose(x=-0.1225, y=0.4358, z=0.0489, rx=1.7429, ry=1.5768, rz=-0.7897)
    unplug_pose = URPose(x=-0.1223, y=0.4333, z=0.0055, rx=1.6390, ry=1.6874, rz=-0.7822)
    release_pose1 = URPose(x=0.0120, y=0.7152, z=0.0271, rx=-1.8993, ry=-1.7929, rz=-0.4988)
    release_pose2 = URPose(x=-0.2713, y=0.5808, z=0.0271, rx=-1.8993, ry=-1.7928, rz=-0.4988)

    # How often to run the (cheap) grasp detector, s. ~10 Hz so the settle window
    # sees several samples without adding overhead at the 100 Hz control rate.
    GRASP_INTERVAL = 0.1

    @staticmethod
    def add_args(parser):
        parser.add_argument('--grasp_ckpt', type=str, default=None,
                            help='Checkpoint for the trained grasp classifier (e.g. '
                                 'logs/grasp-cls/ckpt_best.pth). Default: the hand-tuned '
                                 'GraspDetector in agent/model/grasp.py.')
        parser.add_argument('--grasp_device', type=str, default='cuda:0',
                            help='Device for --grasp_ckpt inference (falls back to CPU)')
        parser.add_argument('--grasp_thr', type=float, default=0.5,
                            help='P(held) above which --grasp_ckpt calls the cable grasped')

    def args2metadata(self, args):
        meta = {}
        meta['id'] = args.id
        meta['grip_width_mm'] = GRIP_WIDTH_MM
        meta['grip_force_n'] = GRIP_FORCE_N
        meta['grip_speed_mmps'] = GRIP_SPEED_MMPS
        meta['grip_pullback_mm'] = GRIP_PULLBACK_MM
        # Which detector answered `Cable?` for this episode.
        meta['grasp_detector'] = args.grasp_ckpt if args.grasp_ckpt else 'heuristic'
        return meta

    def pre_reset(self):
        pass

    def get_action(self):
        if self.iface.dualsense.state.DpadDown:
            return self.move_to_port()
        if self.iface.dualsense.state.DpadUp:
            return self.unplug_and_release()
        des_pose = URPose(*self.iface.target_pose)
        des_gripper = self.iface.gripper_state
        des_zforce = self.iface.target_zforce
        adaptive_mode = self.iface.adaptive_mode
        return des_pose, des_gripper, adaptive_mode, des_zforce

    def move_to_port(self):
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=.05)
        seq.move_to(self.port_pose)
        return self.get_action()

    def unplug_and_release(self):
        seq = interrupt(self)
        seq.move_to(self.unplug_pose)
        seq.gripper(GRIP_CLOSED)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=.05)

        r1, r2 = self.release_pose1, self.release_pose2
        rel = np.array(self.release_pose1)
        rel[:2] = np.random.uniform([r1.x, r1.y], [r2.x, r2.y])
        seq.move_to(URPose(*rel))
        seq.gripper(GRIP_OPEN)
        seq.wait(1)
        return self.get_action()

    def _draw_overlay(self, image):
        """Draw the current estimated state (+ raw decision signals) onto a copy."""
        img = image.copy()
        st = self._est.state

        # Big state banner: red for error states, green otherwise.
        color = (0, 0, 220) if st in ERROR_STATES else (0, 200, 0)  # BGR
        cv2.putText(img, st.value, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, st.value, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        # Raw signals feeding the decisions -- handy while teleoperating.
        cable = '?' if self._held is None else ('yes' if self._held else 'no ')
        # The trained detector also reports its confidence; the heuristic one has none.
        proba = getattr(self.grasp, 'last_proba', None)
        if proba is not None:
            cable += f'({proba:.2f})'
        grip = 'CLOSED' if self._grip == GRIP_CLOSED else 'OPEN'
        pend = self._est._pending[0] if self._est._pending else '-'
        cv2.putText(img,
                    f"cable:{cable}  net:{'yes' if self._network else 'no '}  "
                    f"grip:{grip}  pending:{pend}",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return img

    def post_step(self, obs, action):
        img = obs['image']
        self._grip = action[1]
        self._network = bool(obs.get('network_status'))

        now = time.time()
        if img is not None and now - self._last_grasp_t >= self.GRASP_INTERVAL:
            # obs image is BGR (OpenCV); GraspDetector expects RGB.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._held = self.grasp.update(rgb)  # debounced Cable? answer
            self._last_grasp_t = now

        self._est.update(self._grip, bool(self._held), self._network)
        self.display_image = self._draw_overlay(img) if img is not None else img

    def runtime_info(self):
        obs = self.last_obs
        p = obs['state']['actual_pose']
        print(f'[{self._est.state.value:<12}] '
              f'URPose(x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}, '
              f'rx={p.rx:.4f}, ry={p.ry:.4f}, rz={p.rz:.4f})', end='\r')

    def __init__(self, args):
        control_freq = 100
        home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)

        # Both detectors expose the same debounced update(rgb) -> bool.
        if args.grasp_ckpt:
            self.grasp = NeuralGraspDetector(args.grasp_ckpt, device=args.grasp_device,
                                             threshold=args.grasp_thr)
        else:
            self.grasp = GraspDetector()
        self._est = StateEstimator()
        self._last_grasp_t = 0.0
        self._held = None
        self._grip = GRIP_OPEN
        self._network = False

        data_path = None if args.debug else args.path
        metadata = self.args2metadata(args)
        super().__init__(home_pose=home_pose, control_freq=control_freq,
                         gforce=GRIP_FORCE_N, gwidth=GRIP_WIDTH_MM,
                         gspeed=GRIP_SPEED_MMPS, gpullback=GRIP_PULLBACK_MM,
                         env_metadata=metadata, show_image=True,
                         path=data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation with state-estimator overlay')
    parser.add_argument('--path', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_plugin_unplug',
                        help='Base dataset directory')
    parser.add_argument('--id', type=int, default=None,
                        help='Episode ID (default: next available)')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False)
    Teleoperation.add_args(parser)

    args = parser.parse_args()
    if args.id is None and not args.debug:
        indices = [
            int(d.removeprefix('episode'))
            for d in os.listdir(args.path)
            if d.startswith('episode') and d.removeprefix('episode').isdigit()
        ] if os.path.exists(args.path) else []
        args.id = max(indices, default=0) + 1
        print(f'Auto-selected episode ID: {args.id}')

    if not args.debug:
        print(f"Saving data to: {args.path}, Episode {args.id}")
    os.makedirs(args.path, exist_ok=True)
    teleop = Teleoperation(args)
    teleop.run()
