import interface
from env import Env, URPose
import cv2


class RobotExecution:
    def __init__(self,
                 path=None,
                 control_freq=100,
                 home_pose=URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633),
                 gforce=40,
                 gwidth=10,
                 gspeed=50,
                 gpullback=5,
                 env_metadata={},
                 show_image=True
                 ):
        fps = 20
        self.control_freq = control_freq
        self.control_dt = 1 / control_freq
        self.home_pose = home_pose  # low-position (cable easy to see)
        self.show_image = show_image

        # Robot control env
        self.env = Env(
            robot_ip="192.168.0.100",
            gripper_ip="192.168.0.20",
            camera_crop_mode=1,
            dataset_path=path,
            control_frequency=control_freq,
            save_interval=1.0 / fps,
            gforce=gforce,
            gwidth=gwidth,
            gspeed=gspeed,
            gpullback=gpullback,
            metadata=env_metadata
        )

        # Joystick interface
        self.iface = interface.DualSenseInterface(
            home_pose,
            xyzspeed=0.08,
            rpyspeed=0.9,
            forcespeed=5.,
        )

        self.pre_reset()
        self.env.reset(home_pose)  # start camera, robot go home, gripper open
        self.post_reset()
        self.pre_start()
        self.env.start()  # start threads
        self.post_start()

    def get_action(self):
        """ Returns the action to send to environment.
        Can return fewer than 4 values, in which des_zforce and adaptive_mode will be ignored.
        Returns:
            des_pose: URPose
            des_gripper: int
            des_zforce?: float
            adaptive_mode?: bool
        """
        des_pose = URPose(*self.iface.target_pose)
        des_gripper = self.iface.gripper_state
        des_zforce = self.iface.target_zforce
        adaptive_mode = self.iface.adaptive_mode
        return des_pose, des_gripper, adaptive_mode, des_zforce

    def run(self):
        self.pre_run()
        while True:
            self.env.init_period()

            # ========================================================
            # Read joystick input
            # ========================================================
            flag = self.iface.update(self.control_dt)
            if flag == -1:
                break

            act = self.get_action()
            if len(act) == 4:
                des_pose, des_gripper, adaptive_mode, des_zforce = act
            else:
                des_pose, des_gripper = act
                adaptive_mode = False
                des_zforce = 0.

            # ========================================================
            # Step environment
            # ========================================================
            self.last_obs = self.env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
                des_zforce=des_zforce,
                adaptive_mode=adaptive_mode,
            )
            self.iface.store_obs(self.last_obs)
            self.runtime_info()
            if self.show_image:
                cv2.imshow('RGB', self.last_obs['image'])
                cv2.waitKey(1)

            self.env.wait_period()
        self.close()

    def pre_reset(self):
        pass

    def post_reset(self):
        pass

    def pre_start(self):
        print("Starting teleoperation loop...")
        pass

    def post_start(self):
        pass

    def pre_run(self):
        pass

    def runtime_info(self):
        obs = self.last_obs
        print(f"g_width: {obs['state']['gripper_width']:7.2f}", end='\r')



    def close(self):
        self.env.close()
        print('Env closed. Exiting.')
