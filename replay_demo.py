import os, imageio
import interface
import cv2, time, numpy as np
from env import Env, URPose, blend


def main():
    id = 0
    task = 'ethernet_unplug'
    dataset_path = f'/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_unplug/10'
    state_path = os.path.join(dataset_path, 'states.npz')
    des_poses = np.load(state_path)['actual_pose']
    des_grippers = np.load(state_path)['gripper_width']
    assert len(des_poses) == len(des_grippers), "Length of poses and gripper states must match"
    threshold = 30 # larger than threshold is 0, smaller than threshold is 1
    des_grippers = np.where(des_grippers > 20, 0.0, 1.0)
   
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
    )

    # Reset robot
    env.reset(home_pose)

    # ================================================================
    # Initialize joystick interface
    # ================================================================
    

    print("Starting replay loop...")
    env.start()

    for des_pose, des_gripper in zip(des_poses, des_grippers):
        t0 = time.perf_counter()
        des_pose = URPose(*des_pose)
        des_gripper =  des_gripper

        # ========================================================
        # Step environment
        # ========================================================
        obs = env.step(
            des_pose=des_pose,
            des_gripper_state=des_gripper,
        )
        cv2.imshow('RGB', obs['image'])
        cv2.waitKey(1)

        sleep_time = 0.1 #  max(0, env.dt - (time.perf_counter() - t0))
        time.sleep(sleep_time)

    # except:
    #     print("\nStopping teleoperation...")

        sleep_time = max(0, env.dt - (time.perf_counter() - t0))
        time.sleep(sleep_time)
        

    env.close()


if __name__ == "__main__":
    main()