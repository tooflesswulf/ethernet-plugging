import os, imageio
import interface
import time, numpy as np
from env import Env, URPose, blend


def main(id=0):
   
    fps = 20 # saving data frequency
    task = 'ethernet_unplug'
    
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
    dataset_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset'
    env = Env(
        robot_ip="192.168.0.100",
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,
        dataset_path=None,
        save_interval=1.0 / fps, 
    )

    # ================================================================
    # Initialize joystick interface
    # ================================================================
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.01,
        rpyspeed=0.1,
    )


    dataset_path = os.path.join(dataset_path, task, str(id))
    env.dataset_path = dataset_path

    env.reset(home_pose) # start camera, robot go home, gripper open

    

    print("Starting teleoperation loop...")
    env.start() # start threads
   
    # try:
    while True:
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

        print(f'pos: {env.g_pos:7.2f} mm | force: {env.g_force:7.2f} N', end='\r')
        
    # except:
    #     print("\nStopping teleoperation...")

    # save a gif of the RGB images for quick visualization
    # if dataset_path is not None:
    #     image_dir = os.path.join(dataset_path, "images")
    #     images, N = [], len(os.listdir(image_dir))
    #     for idx in range(N):
    #         image_path = os.path.join(image_dir, f"{idx}.png")
    #         images.append(imageio.imread(image_path))
    #     gif_path = os.path.join(dataset_path, f"{id}.gif")
    #     # create gif wit 1/save_interval fps
    #     imageio.mimsave(gif_path,
    #         images,
    #         fps=1.0 / env.save_interval,
    #         loop=0, # infinite loop
    #     )

    env.close()


if __name__ == "__main__":
    # accept an optional command line argument for the demo ID
    import argparse
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--id', type=int, default=0, help='ID for the demo (default: 0)')
    args = parser.parse_args()
    main(id=args.id)
     