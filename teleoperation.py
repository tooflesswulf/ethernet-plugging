import argparse
import os
import interface
import time
from env import Env, URPose
import cv2
import random

GRIP_WIDTH_MM = 8 # 10 # 8
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10 # 20 # 10


def main(path=None, id=0, debug=False, meta={}):
    fps = 20 # 10  # saving data frequency
    controller_dt = 1 / 100

    # Home pose
    # ================================================================
    # home_pose = URPose(-0.125,0.545,0.305, 2.44,2.44,0.653, ) # high-position (cable too hard to to see)
    home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see)
    # home_pose = URPose(-0.147, 0.612, 0.184, 2.42, 2.42, 2.42)  # low-position (cable easy to see)

    # ================================================================
    # Initialize environment
    # ================================================================
    dataset_path = path
    env = Env(
        robot_ip="192.168.0.100",
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,
        dataset_path=dataset_path,
        save_interval=1.0 / fps,
        gforce=GRIP_FORCE_N,
        gwidth=GRIP_WIDTH_MM,
        gspeed=GRIP_SPEED_MMPS,
        gpullback=GRIP_PULLBACK_MM,
        metadata=meta
    )

    # ================================================================
    # Initialize joystick interface
    # ================================================================
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.08,
        rpyspeed=0.9,
        forcespeed=5.,
    )

    if 'rng' in meta:
        print(f"Target port = {meta['rng'] + 1}")

    env.reset(home_pose)  # start camera, robot go home, gripper open
    print("Starting teleoperation loop...")
    env.start()  # start threads

    while True:
        t0 = time.perf_counter()

        # ========================================================
        # Read joystick input
        # ========================================================
        flag = iface.update(controller_dt)
        if flag == -1:
            break
        des_pose = URPose(*iface.target_pose)
        des_gripper = iface.gripper_state

        # ========================================================
        # Step environment
        # ========================================================
        obs = env.step(
            des_pose=des_pose,
            des_gripper_state=des_gripper,
            des_zforce=iface.target_zforce,
            adaptive_mode=iface.adaptive_mode,
        )
        iface.store_obs(obs)

        # print(f"pos: {obs['state']['gripper_width']:7.2f} mm | force: {obs['state']['gripper_force']:7.2f} N" +
        #       f" | eef force: [{ff.x:5.2f}, {ff.y:5.2f}, {ff.z:5.2f}, {ff.rx:5.2f}, {ff.ry:5.2f}, {ff.rz:5.2f}]", end='\r')
        print(f"mode: {iface.adaptive_mode}, des zforce: {iface.target_zforce:7.2f} N | eef zforce: {obs['state']['filtered_force'].z:7.2f} N", end='\r')

        cv2.imshow('RGB', obs['image'])
        cv2.waitKey(1)

        sleep_time = max(0, controller_dt - (time.perf_counter() - t0))
        time.sleep(sleep_time)

    env.close()
    print('Env closed. Exiting.')


def create_metadata(args):
    meta = {}
    meta['id'] = args.id
    if args.rng is not None:
        rng = random.randint(0, args.rng - 1)
        meta['rng'] = rng
        print(f'=========== RNG={rng} ===========')
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--path', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_pluginv2_yiqi',
                        help='Base dataset directory')
    parser.add_argument('--id', type=int, default=None,
                        help='Episode ID (default: next available)')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--rng', type=int, default=None,
                        help='Generate a random number at the start')

    args = parser.parse_args()

    if args.id is None:
        indices = [
            int(d.removeprefix('episode'))
            for d in os.listdir(args.path)
            if d.startswith('episode') and d.removeprefix('episode').isdigit()
        ] if os.path.exists(args.path) else []
        args.id = max(indices, default=-1) + 1
        print(f'Auto-selected episode ID: {args.id}')

    meta = create_metadata(args)

    print(f"Saving data to: {args.path}, Episode {args.id}")
    os.makedirs(args.path, exist_ok=True)
    main(path=args.path, id=args.id, debug=args.debug, meta=meta)
