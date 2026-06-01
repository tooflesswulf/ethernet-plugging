import argparse
import os
import imageio
import interface
import time
import numpy as np
from env import Env, URPose, blend


def main(dataset_path):
    # ================================================================
    # Home pose
    # ================================================================
    home_pose = URPose(
        -0.125,
        0.545,
        0.305,
        2.44,
        2.44,
        0.653,
    )

    # ================================================================
    # Initialize environment
    # ================================================================
    env = Env(
        robot_ip="192.168.0.100",
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,
        dataset_path=dataset_path,
    )

    # Reset robot
    env.reset(home_pose)

    # ================================================================
    # Initialize joystick interface
    # ================================================================
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.01,
        rpyspeed=0.1,
    )

    print("Starting teleoperation loop...")
    env.start()
    save_obs_interval = 0.1  # seconds
    last_save_time = time.time()
    try:
        while True:
            # ========================================================
            # Read joystick input
            # ========================================================

            iface.update(env.dt)
            des_pose = URPose(*iface.target_pose)
            des_gripper = iface.gripper_state

            # ========================================================
            # Step environment
            # ========================================================
            obs = env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
            )

            print(f'pos: {env.g_pos:7.2f} mm | force: {env.g_force:7.2f} N', end='\r')

    except:
        print("\nStopping teleoperation...")

    # save a gif of the RGB images for quick visualization
    if dataset_path is not None:
        image_dir = os.path.join(dataset_path, "images")
        images, N = [], len(os.listdir(image_dir))
        for idx in range(N):
            image_path = os.path.join(image_dir, f"{idx}.png")
            images.append(imageio.imread(image_path))
        gif_path = os.path.join(dataset_path, f"{id}.gif")
        # create gif wit 1/save_interval fps
        imageio.mimsave(gif_path,
                        images,
                        fps=1.0 / env.save_interval,
                        loop=0,  # infinite loop
                        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation data collection')
    parser.add_argument('--path', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_unplug',
                        help='Base dataset directory')
    parser.add_argument('--id', type=int, default=None,
                        help='Episode ID (default: next available)')
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

    dataset_path = os.path.join(args.path, f'episode{id}')

    main(dataset_path)
