import interface

from env import Env, URPose, blend


def main():
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
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.01,
        rpyspeed=0.1,
    )

    print("Starting teleoperation loop...")
    env.start()
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

           
    except:
        print("\nStopping teleoperation...")

    finally:
        env.close()


if __name__ == "__main__":
    main()