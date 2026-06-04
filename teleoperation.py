import argparse
import os
import imageio
import interface
import time
import numpy as np
from env import Env, URPose, blend
import cv2


def main(path=None, id=0, debug=False):
    fps = 20 # saving data frequency
    
    # Home pose
    # ================================================================
    # home_pose = URPose(-0.125,0.545,0.305, 2.44,2.44,0.653, ) # high-position (cable too hard to to see)
    home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633) # low-position (cable easy to see)

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
    )

    # ================================================================
    # Initialize joystick interface
    # ================================================================
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.05,
        rpyspeed=0.5,
    )

    env.reset(home_pose) # start camera, robot go home, gripper open

    print("Starting teleoperation loop...")
    env.start() # start threads
    while True:
        t0 = time.perf_counter()

        # ========================================================
        # Read joystick input
        # ========================================================
        flag = iface.update(env.dt)
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
        )

        print(f"pos: {obs['state']['gripper_width']:7.2f} mm | force: {obs['state']['gripper_force']:7.2f} N", end='\r')

        cv2.imshow('RGB', obs['image'])
        cv2.waitKey(1)

        sleep_time = max(0, env.dt - (time.perf_counter() - t0))
        time.sleep(sleep_time)

    env.close()
    print('Env closed. Exiting.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--path', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_unplug_red',
                        help='Base dataset directory')
    parser.add_argument('--id', type=int, default=None,
                        help='Episode ID (default: next available)')
    parser.add_argument('-d', '--debug', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()

    if args.id is not None:
        id = args.id
    else:
        indices = [
            int(d.removeprefix('episode'))
            for d in os.listdir(args.path)
            if d.startswith('episode') and d.removeprefix('episode').isdigit()
        ] if os.path.exists(args.path) else []
        id = max(indices, default=-1) + 1
        print(f'Auto-selected episode ID: {id}')
        
    print(f"Saving data to: {args.path}, Episode {id}")
    os.makedirs(args.path,  exist_ok=True)
    main(path=args.path, id=id, debug=args.debug)

c