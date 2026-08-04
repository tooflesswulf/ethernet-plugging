from agent.utils.robot_utils import interrupt
from agent.model.qwen import QwenClient, ETHERNET_STATES
import robot_execution
from env import URPose, GRIP_OPEN, GRIP_CLOSED
import argparse
import collections
import threading
import time
import cv2
import numpy as np
import os
from PIL import Image

GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10


class QwenWorker(threading.Thread):
    """Runs the QwenClient off the control loop.

    The main (20+ Hz) loop feeds recent frames via `submit_frame` and reads the
    latest state probabilities via `get_probs`. The model runs slower than the
    control loop, so it lives here on its own thread and simply publishes its most
    recent result; the control loop never blocks on it.
    """

    def __init__(self, states=ETHERNET_STATES, window=8, **client_kwargs):
        super().__init__(daemon=True)
        self.states = list(states)
        self.client_kwargs = client_kwargs

        # Rolling buffer of recent frames (oldest -> newest). maxlen == window so
        # score_state can pick the sharpest of the recent tail.
        self._frames = collections.deque(maxlen=window)
        self._frame_lock = threading.Lock()

        self._result_lock = threading.Lock()
        self._probs = {s: None for s in self.states}
        self._infer_hz = 0.0
        self._ready = False

        self._stop = threading.Event()

    def submit_frame(self, image_bgr):
        """Push a control-loop frame (BGR uint8) into the rolling buffer."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._frame_lock:
            self._frames.append(Image.fromarray(rgb))

    def get_probs(self):
        """Return (probs, infer_hz, ready) as a snapshot for overlaying."""
        with self._result_lock:
            return dict(self._probs), self._infer_hz, self._ready

    def run(self):
        # Heavy: loads the VLM weights. Done on this thread so __init__ of the
        # teleop / robot setup isn't blocked by it.
        client = QwenClient(**self.client_kwargs)
        with self._result_lock:
            self._ready = True

        while not self._stop.is_set():
            with self._frame_lock:
                frames = list(self._frames)
            if not frames:
                time.sleep(0.02)
                continue

            t0 = time.time()
            probs = client.score_state(frames)
            dt = time.time() - t0
            with self._result_lock:
                self._probs = probs
                self._infer_hz = (1.0 / dt) if dt > 0 else 0.0

    def stop(self):
        self._stop.set()


class Teleoperation(robot_execution.RobotExecution):
    # port_pose = URPose(x=-0.1269, y=0.5979, z=0.0730, rx=1.7296, ry=1.7668, rz=-0.760357)
    # release_pose = URPose(x=-0.0297, y=0.7031, z=0.0813, rx=-2.1029, ry=-2.0230, rz=-0.4256)

    port_pose = URPose(x=-0.1221, y=0.4944, z=0.0583, rx=1.8300, ry=1.6718, rz=-0.6700)
    unplug_pose = URPose(x=-0.1217, y=0.4901, z=0.0236, rx=1.7954, ry=1.8063, rz=-0.6346)
    release_pose1 = URPose(x=-0.0168, y=0.7714, z=0.0450, rx=-1.8849, ry=-1.8845, rz=-0.4705)
    release_pose2 = URPose(x=-0.2665, y=0.6624, z=0.0450, rx=-1.8849, ry=-1.8845, rz=-0.4705)

    # How often to feed a frame to the Qwen worker (s). ~7 Hz spreads the 8-frame
    # window over ~1s of motion without flooding the buffer at control rate.
    QWEN_SUBMIT_INTERVAL = 0.15

    @staticmethod
    def add_args(parser):
        pass

    def args2metadata(self, args):
        meta = {}
        meta['id'] = args.id
        meta['grip_width_mm'] = GRIP_WIDTH_MM
        meta['grip_force_n'] = GRIP_FORCE_N
        meta['grip_speed_mmps'] = GRIP_SPEED_MMPS
        meta['grip_pullback_mm'] = GRIP_PULLBACK_MM

        # RNG for target port selection
        return meta

    def pre_reset(self):
        """Print info to operator at start of teleop session."""
        # print(f'Target port = {self.data["target_port"]}')
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
        seq.wait(1) \
            .then(lambda _: print('Released! Safe to exit.')) \
            .then(lambda _: self.stop())
        return self.get_action()

    def _draw_overlay(self, image):
        """Draw the latest Qwen state probabilities onto a copy of `image`."""
        img = image.copy()
        probs, infer_hz, ready = self.qwen.get_probs()

        x, y = 10, 24
        if not ready:
            cv2.putText(img, 'Qwen: loading model...', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            return img

        bar_x = x + 60
        bar_w = 140
        for state in self.qwen.states:
            p = probs.get(state)
            label = f'{p:.2f}' if p is not None else ' -- '
            # red (low) -> green (high) in BGR
            color = (0, 0, 200) if p is None else (0, int(255 * p), int(255 * (1 - p)))

            cv2.putText(img, label, (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)
            # probability bar
            cv2.rectangle(img, (bar_x, y - 8), (bar_x + bar_w, y + 4), (60, 60, 60), 1)
            if p is not None:
                cv2.rectangle(img, (bar_x, y - 8),
                              (bar_x + int(bar_w * p), y + 4), color, -1)
            # state text (truncated to fit)
            cv2.putText(img, state[:42], (bar_x + bar_w + 8, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
            y += 26

        cv2.putText(img, f'{infer_hz:.1f} Hz', (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def post_step(self, obs, action):
        img = obs['image']

        now = time.time()
        if now - self._last_qwen_submit >= self.QWEN_SUBMIT_INTERVAL:
            self.qwen.submit_frame(img)
            self._last_qwen_submit = now

        self.display_image = self._draw_overlay(img)

    def runtime_info(self):
        obs = self.last_obs
        st = obs['state']
        p = st['actual_pose']
        force = obs['state']['filtered_force']
        # print(f"Pose: {st['actual_pose']}", end='\r')
        print(f'URPose(x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}, rx={p.rx:.4f}, ry={p.ry:.4f}, rz={p.rz:.4f})', end='\r')

    def close(self):
        self.qwen.stop()
        super().close()

    def __init__(self, args):
        control_freq = 100
        home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)

        # Start the Qwen worker before super().__init__ so the (slow) model load
        # overlaps with robot/camera bring-up.
        self.qwen = QwenWorker(states=ETHERNET_STATES)
        self._last_qwen_submit = 0.0
        self.qwen.start()
        while not self.qwen._ready:
            time.sleep(1)

        data_path = None if args.debug else args.path
        metadata = self.args2metadata(args)
        super().__init__(home_pose=home_pose, control_freq=control_freq,
                         gforce=GRIP_FORCE_N, gwidth=GRIP_WIDTH_MM,
                         gspeed=GRIP_SPEED_MMPS, gpullback=GRIP_PULLBACK_MM,
                         env_metadata=metadata, show_image=True,
                         path=data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
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
