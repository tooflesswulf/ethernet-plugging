import robot_execution
from env import URPose
import argparse
import os

GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 5


class Teleoperation(robot_execution.RobotExecution):
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
