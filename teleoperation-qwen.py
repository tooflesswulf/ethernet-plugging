from agent.utils.robot_utils import interrupt
from agent.model.grasp import GraspDetector, cable_region
import robot_execution
from env import URPose, GRIP_OPEN, GRIP_CLOSED
import argparse
import time
import cv2
import numpy as np
import os

GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10


class Teleoperation(robot_execution.RobotExecution):
    # port_pose = URPose(x=-0.1269, y=0.5979, z=0.0730, rx=1.7296, ry=1.7668, rz=-0.760357)
    # release_pose = URPose(x=-0.0297, y=0.7031, z=0.0813, rx=-2.1029, ry=-2.0230, rz=-0.4256)

    port_pose = URPose(x=-0.1221, y=0.4944, z=0.0583, rx=1.8300, ry=1.6718, rz=-0.6700)
    unplug_pose = URPose(x=-0.1217, y=0.4901, z=0.0236, rx=1.7954, ry=1.8063, rz=-0.6346)
    release_pose1 = URPose(x=-0.0168, y=0.7714, z=0.0450, rx=-1.8849, ry=-1.8845, rz=-0.4705)
    release_pose2 = URPose(x=-0.2665, y=0.6624, z=0.0450, rx=-1.8849, ry=-1.8845, rz=-0.4705)

    # How often to run the (cheap) grasp detector, s. ~10 Hz so the debounce
    # window spans ~0.5 s without adding overhead at the 100 Hz control rate.
    GRASP_INTERVAL = 0.1

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
        """Draw the latest grasp state + cable region onto a copy of `image`."""
        img = image.copy()
        g = self._grasp
        x, y = 10, 24
        if g is None:
            return img

        # Draw the fixed finger-gap ROI the detector reads from.
        x0, y0, x1, y1 = self.grasp.roi
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)

        held = g['held']
        color = (0, 200, 0) if held else (0, 0, 200)  # green / red (BGR)
        cv2.putText(img, f"HELD: {'YES' if held else 'no ':>3}   region: {g['region']}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"warm {g['warm']:.2f}  spec {g['spec']:.2f}   {g['hz']:.0f} Hz",
                    (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return img

    def post_step(self, obs, action):
        img = obs['image']

        now = time.time()
        if img is not None and now - self._last_grasp_t >= self.GRASP_INTERVAL:
            # obs image is BGR (OpenCV); GraspDetector expects RGB.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            held = self.grasp.update(rgb)                       # debounced bool
            warm, spec = self.grasp.cues(rgb)                   # for the overlay
            pose = obs['state']['actual_pose']
            region = cable_region(pose, held, self.region_boxes)
            dt = now - self._last_grasp_t
            self._grasp = dict(held=held, region=region, warm=warm, spec=spec,
                               hz=(1.0 / dt) if dt > 0 else 0.0)
            self._last_grasp_t = now

        self.display_image = self._draw_overlay(img)

    def runtime_info(self):
        obs = self.last_obs
        st = obs['state']
        p = st['actual_pose']
        force = obs['state']['filtered_force']
        # print(f"Pose: {st['actual_pose']}", end='\r')
        print(f'URPose(x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}, rx={p.rx:.4f}, ry={p.ry:.4f}, rz={p.rz:.4f})', end='\r')

    def close(self):
        super().close()

    def _make_region_boxes(self):
        """Axis-aligned xyz boxes for cable_region, derived from the task poses.

        When the cable is grasped, its location is the EE region: near the port,
        near the staging/release area, or in transit between them.
        """
        p = self.port_pose
        r1, r2 = self.release_pose1, self.release_pose2
        return {
            "at_port": ((p.x - 0.03, p.y - 0.03, -0.01),
                        (p.x + 0.03, p.y + 0.03, 0.12)),
            "at_staging": ((min(r1.x, r2.x) - 0.04, min(r1.y, r2.y) - 0.04, -0.01),
                           (max(r1.x, r2.x) + 0.04, max(r1.y, r2.y) + 0.04, 0.12)),
        }

    def __init__(self, args):
        control_freq = 100
        home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)

        # Lightweight, training-free grasp/location estimation (replaces the Qwen
        # VLM). Cheap enough to run inline in the control loop -- no worker thread.
        self.grasp = GraspDetector()
        self.region_boxes = self._make_region_boxes()
        self._last_grasp_t = 0.0
        self._grasp = None

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
