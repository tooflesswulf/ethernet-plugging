import torch, numpy as np, collections, time, interface
from PIL import Image
from pathlib import Path
from einops import rearrange
import threading
from agent.utils.robot_utils import get_actions, build_states
from agent.evaluate.realtime_chunking import RealtimeActionChunkingBuffer
from agent.model.policy import DiffusionPolicy
from env import Env, URPose, GRIP_OPEN

class BasePolicy:
    def __init__(self, base_policy: DiffusionPolicy, env: Env, base_dt=1/20, weight_decay=0.5, device='cuda'):
        self.policy = base_policy
        self.env = env
        self.device = device
        self.base_dt = base_dt
        self.buffer = RealtimeActionChunkingBuffer(action_dt=base_dt, weight_decay=weight_decay)
        self.prediction_thread = None
        self.stop_event = None

    def get_base_action(self, t=None):
        if t is None:
            t = time.time()
        # des_pose, des_width, done (global frame)
        return self.buffer.get_action(t)

    def prediction_loop(self):
        action_horizon = self.policy.action_horizon
        obs_horizon = self.policy.obs_horizon

        obs_deque = collections.deque(maxlen=obs_horizon)
        while not self.stop_event.is_set():
            t_obs = time.time()  # observation time the chunk is anchored to
            obs_deque.append(self.env.get_obs())
            if len(obs_deque) < obs_horizon:
                continue

            # get_actions builds images + the obs_fields state vector from the deque.
            with torch.no_grad():
                des_poses, des_grips, des_done = get_actions(self.policy, obs_deque, self.device)

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs.
            # The done score rides through the buffer so it's ensembled and thresholded at
            # execution time in get_action (not averaged over the chunk here).
            start = obs_horizon - 1
            end = start + action_horizon
            self.buffer.add_chunk(t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end])
        self.buffer.clear()

    def reset(self):
        if self.prediction_thread is not None:
            self.stop()
            self.prediction_thread.join()
        self.start()

    def start(self):
        self.stop_event = threading.Event()
        self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        self.prediction_thread.start()

    def stop(self):
        self.stop_event.set()


class BasePolicyVecEnvWrapper:
    """
    Wraps our current environment with a base policy to enable standard RL training of residual policies.

    This wrapper:
    1. Takes raw observations from th environment
    2. Passes them through the base policy to get base actions
    3. Augments observations with base actions for the residual policy
    4. Combines base + residual actions before stepping the environment
    5. Returns augmented observations that include base actions
.
    """

    def __init__(
        self,
        env: Env,

        base_policy,
        image_size,
        lowdim_keys,
        device='cuda'
  
    ):
        """
        Args:
            env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy) to augment with residual actions
        """
        self.env = env; self.iface =  interface.DualSenseInterface( self.env.home_pose, xyzspeed=0.08, rpyspeed=0.9, forcespeed=5. )
        self.base_policy = BasePolicy(base_policy, env, device=device)
        self.device = device
        self.image_size = image_size
        self.lowdim_keys = lowdim_keys
        self.task_stage = 0; self.link_state = None
        # iface parameters
        control_freq = 20
        self.control_freq = control_freq
        self.control_dt = 1 / control_freq

        self.env.reset(self.env.home_pose)
        self.env.start()

    def _process_obs(self, obs_dict):
        rgb = np.array( Image.fromarray( obs_dict['image'] ).resize(self.image_size) )
        keys = [ 'actual_pose' if k == 'pose' else k for k in self.lowdim_keys]; state = obs_dict['state']
        if 'actual_pose' in keys:
            pose = np.array( state['actual_pose'] )
        if 'gripper_width' in keys:
            gripper_width = np.array([ state['gripper_width'] ])
        state = np.concatenate([pose, gripper_width])
        
        return {
            "observation.state": state, # (7, )
            "observation.rgb"  : rgb,   # (128, 128, 3)
            'network_status': obs_dict['network_status']
        }


    def reset(self, **kwargs):
        """Reset environment and base policy."""
 
       
        # Reset base policy (clear previously predicted actions)
        self.base_policy.reset()
        time.sleep(1)

        raw_obs = self._process_obs( self.env.get_obs() )
        # Get base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.get_base_action()
            base_dpose, gripper, done = base_action
            base_action = torch.tensor( np.concatenate([base_dpose, np.array([gripper]), np.array([0])]) )
           
        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_action)

        # Store for later use in step
        self._last_base_naction = base_action; self.task_stage = 0; self.link_state = None #  clear the task stage

        return augmented_obs, {}

    def step_task_stage(self, raw_obs):
        curr_state = raw_obs['network_status']
        prev_state = curr_state if self.link_state is None else self.link_state
        if prev_state != curr_state:
            self.task_stage += 1; self.link_state = curr_state

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment with residual action.

        Args:
            residual_action: The residual action from the residual policy

        Returns:
            augmented_obs: Observations augmented with base actions
            reward: Reward tensor
            terminated: Terminated tensor
            truncated: Truncated tensor
            info: Info dict
        """
        self.env.init_period()
       
        # Combine base and residual actions
        base_action = self.base_policy.get_base_action()
        base_dpose, gripper, done = base_action
        gripper = int(round(gripper))
        info = {}
        residual_pose = residual_naction[0][:6]
        dpose = ( torch.tensor(base_dpose).to(residual_naction.device) + residual_pose ).cpu().numpy()
        dpose = base_dpose
        combined_naction = torch.tensor( np.concatenate([dpose, np.array([gripper]), np.array([0])], -1) )
        # do we need clipping here?

        # Step the underlying environment
        self.env.step(URPose(*dpose), gripper)

        raw_obs = self._process_obs(self.env.get_obs())
        self.step_task_stage(raw_obs)
        reward = self.task_stage
        self.iface.update(self.control_dt)
        terminated = self.iface.dualsense.state.Cross # use iface for this
        
        # Store the scaled action for replay buffer (already computed above)
        info["scaled_action"] = combined_naction

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, combined_naction )

        self.env.wait_period()
        return augmented_obs, reward, terminated, info

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction

        return augmented_obs

    def _process_final_obs_in_info(self, info: dict, device: torch.device) -> dict:
        """Pad final_obs state with zeros to match augmented observation format."""
        if "final_obs" not in info or info["final_obs"] is None:
            return info

        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                # Pad with zeros (no action taken at terminal state)
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.action_dim, device=device, dtype=torch.float32
                )

        return info

    def close(self):
        """Close the environment."""
        return self.env.close()

