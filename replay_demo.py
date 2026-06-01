import os, imageio
import interface
import time, numpy as np
from env import Env, URPose, blend


def main():
    id = 0
    task = 'ethernet_unplug'
    dataset_path = f'/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_unplug_test_success/0'
    state_path = os.path.join(dataset_path, 'states.npz')
    des_poses = np.load(state_path)['actual_pose']
    des_grippers = np.load(state_path)['gripper_width']
    assert len(des_poses) == len(des_grippers), "Length of poses and gripper states must match"
    threshold = 30 # larger than threshold is 0, smaller than threshold is 1
    des_grippers = (np.array(des_grippers) < threshold).astype(float).astype(int) # convert to binary open/close
   
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
    try:
        for des_pose, des_gripper in zip(des_poses, des_grippers):
           
            des_pose = URPose(*des_pose)
            des_gripper = des_gripper

            # ========================================================
            # Step environment
            # ========================================================
            obs = env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
            )
            time.sleep(0.1)

            print(f'pos: {env.g_pos:7.2f} mm | force: {env.g_force:7.2f} N', end='\r')
            
        
    except:
        print("\nStopping replay...")


    env.close()


if __name__ == "__main__":
    main()