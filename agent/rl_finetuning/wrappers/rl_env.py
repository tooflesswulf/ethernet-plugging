import torch, numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange




def check_link( interface = "enx7cc2c6453f68"):
    state = Path(f"/sys/class/net/{interface}/operstate").read_text().strip()
    assert state in ['up', 'down'], 'a invalid link state is found: {state}'
    return state


class BasePolicy:
    def __init__(self, base_policy, device='cuda'):
        self.base_policy = base_policy
        self.device = device
        self.base_actions = []

    def get_action(self, image, state):

        # Get base action from base policy
        if len( self.base_actions) == 0:
            assert image.max() > 1, f"{image.max()}"
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float() 
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float() 
            raw_obs = {'rgb': rearrange( image.unsqueeze(0).unsqueeze(0).to(self.device) / 255.0, 'B T H W C -> B T C H W'), 'state': state.unsqueeze(0).unsqueeze(0).to(self.device),}
            with torch.no_grad():
                self.base_actions = self.base_policy.predict_action(raw_obs).squeeze(0).cpu().numpy()
        base_action = self.base_actions[0]
        self.base_actions =  self.base_actions[1:] if len(self.base_actions) > 1 else []
        return base_action

    def reset(self):
        self.base_actions = []


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
        env,
        iface,
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
        self.env = env; self.iface = iface
        self.base_policy = BasePolicy(base_policy, device=device)
        self.device = device
        self.image_size = image_size
        self.lowdim_keys = lowdim_keys
        self.task_stage = 0; self.link_state = 'down'
        # iface parameters
        control_freq = 20
        self.control_freq = control_freq
        self.control_dt = 1 / control_freq

    def _process_obs(self, obs_dict):
        rgb = np.array( Image.fromarray( obs_dict['image'] ).resize(self.image_size) )
        state = np.concatenate([ obs_dict[k] if obs_dict[k].ndim == 2 else obs_dict[k][:, None] for k in self.lowdim_keys], -1)
        assert False, f"{rgb.shape} {rgb.max()} {state.shape}"
        return {
            "observation.state" : state,
             "observation.rgb"  : rgb
        }


    def reset(self, **kwargs):
        """Reset environment and base policy."""
        # Reset the underlying environment
        self.env.reset()
        # Reset base policy (clear previously predicted actions)
        self.base_policy.reset()
        raw_obs = self._process_obs( self.env.get_obs() )
        # Get base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.get_action()

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_action)

        # Store for later use in step
        self._last_base_naction = base_naction
        self.task_stage = 0; self.link_state = 'down' #  clear the task stage

        return augmented_obs, {}

    def step_task_stage(self):
        prev_state = self.linK_state; curr_state = check_link()
        if prev_state != curr_state:
            self.task_stage += 1; prev_state = curr_state

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
        # Combine base and residual actions
        combined_naction = self._last_base_naction + residual_naction
        # do we need clipping here?

        # Step the underlying environment
        raw_obs = self._process_obs(self.env.step(combined_naction)); self.step_task_stage()
        reward = self.task_stage
        terminated = False if self.iface # use iface for this

        # Store the scaled action for replay buffer (already computed above)
        info["scaled_action"] = combined_naction

        # Get next base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Handle policy reset for terminated environments
        if terminated.any():
            # base policy will store a chunk of action. If reset, clear the stored action.
            self.base_policy.reset() 

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Handle final_obs in info dict to ensure consistent shapes
        if "final_obs" in info:
            info = self._process_final_obs_in_info(info, combined_naction.device)

        # Store for next step
        self._last_base_naction = base_naction

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

