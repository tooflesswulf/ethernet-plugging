from agent.utils.robot_utils import interrupt
import robot_execution
from env import URPose, GRIP_OPEN
import argparse
import numpy as np
import os

GRIP_WIDTH_MM = 15
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10


class Teleoperation(robot_execution.RobotExecution):
    port_xyz = URPose(x=-0.1269, y=0.5979, z=0.0730, rx=1.7296, ry=1.7668, rz=-0.760357)
    release_pose = URPose(x=-0.0297, y=0.7031, z=0.0813, rx=-2.1029, ry=-2.0230, rz=-0.4256)

    @staticmethod
    def add_args(parser):
        pass

    def args2metadata(self, args):
        meta = {}
        meta['id'] = args.id
        # RNG for target port selection
        # targ_port = random.randint(1, 4)
        # meta['target_port'] = targ_port
        return meta

    def pre_reset(self):
        """Print info to operator at start of teleop session."""
        # print(f'Target port = {self.data["target_port"]}')
        pass

    def get_action(self):
        if self.iface.dualsense.state.DpadDown:
            return self.move_to_port()
        if self.iface.dualsense.state.DpadRight:
            return self.release_cable()
        des_pose = URPose(*self.iface.target_pose)
        des_gripper = self.iface.gripper_state
        des_zforce = self.iface.target_zforce
        adaptive_mode = self.iface.adaptive_mode
        return des_pose, des_gripper, adaptive_mode, des_zforce

    def move_to_port(self):
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=.05)
        seq.move_to(self.port_xyz)
        return self.get_action()

    def release_cable(self):
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=.05)
        rel = np.array(self.release_pose)
        rel[:2] += np.random.uniform([-.04, -.08], [.04, .05])
        seq.move_to(URPose(*rel))
        seq.gripper(GRIP_OPEN)
        return self.get_action()

    def runtime_info(self):
        obs = self.last_obs
        st = obs['state']
        p = st['actual_pose']
        force = obs['state']['filtered_force']
        # print(f"Pose: {st['actual_pose']}", end='\r')
        print(f'URPose(x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}, rx={p.rx:.4f}, ry={p.ry:.4f}, rz={p.rz:.4f})', end='\r')

    def __init__(self, args):
        control_freq = 100
        home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)

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
