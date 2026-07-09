from env import Env, URPose, GRIP_OPEN
import interface
import threading
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
        self.home_pose = home_pose
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
        self.stop_event = threading.Event()

        self.pre_reset()
        self.env.reset(home_pose)  # start camera, robot go home, gripper open
        self.post_reset()
        self.pre_start()
        self.env.start()  # start threads
        self.post_start()
        self.last_action = (self.home_pose, GRIP_OPEN, 0., False)

    def get_action(self):
        """ Returns the action to send to _unshortcut_action.
        """
        des_pose = URPose(*self.iface.target_pose)
        des_gripper = self.iface.gripper_state
        des_zforce = self.iface.target_zforce
        adaptive_mode = self.iface.adaptive_mode
        return des_pose, des_gripper, adaptive_mode, des_zforce

    def _unshortcut_action(self, act):
        """ Converts action shortcuts into unified format.
        act: None             -> repeat last action
        act: (float[6], int)  -> pose + gripper (GRIP_OPEN=0, GRIP_CLOSED=1)
        act: (float[6], int, bool, float) -> pose, grip, zforce, mode
        """
        if act is None:
            return self.last_action
        if len(act) == 2:
            return (URPose(*act[0]), int(round(act[1])), 0, False)
        return (URPose(*act[0]), int(round(act[1])), bool(act[2]), act[3])

    def run(self):
        self.pre_run()
        while not self.stop_event.is_set():
            self.env.init_period()

            # ========================================================
            # Read joystick input
            # ========================================================
            flag = self.iface.update(self.control_dt)
            if flag == -1:
                break

            controller_state = self.iface.dualsense.state
            act = self.get_action()
            action = self._unshortcut_action(act)
            des_pose, des_gripper, adaptive_mode, des_zforce = action
            self.last_action = action

            # ========================================================
            # Step environment
            # ========================================================
            self.last_obs = self.env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
                des_zforce=des_zforce,
                adaptive_mode=adaptive_mode,
                dualsense=controller_state
            )
            self.post_step(self.last_obs, self.last_action)
            self.iface.store_obs(self.last_obs)
            self.runtime_info()
            if self.show_image:
                cv2.imshow('RGB', self.last_obs['image'])
                cv2.waitKey(1)

            self.env.wait_period()
        self.stop()
        self.close()

    def stop(self):
        self.stop_event.set()

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

    def post_step(self, obs, action):
        pass

    def runtime_info(self):
        obs = self.last_obs
        print(f"g_width: {obs['state']['gripper_width']:7.2f}", end='\r')

    def close(self):
        self.env.close()
        print('Env closed. Exiting.')
