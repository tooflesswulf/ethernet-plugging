import torch 

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
        base_policy,
        action_scaler,
        state_standardizer,
    ):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy) to augment with residual actions
            action_scaler: ActionScaler object for scaling/unscaling actions (REQUIRED)
            state_standardizer: StateStandardizer object for standardizing states (REQUIRED)
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"

        self.env =env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer

        # Get action dimension from the environment
        self.action_dim = 7

    def reset(self, **kwargs):
        """Reset environment and base policy."""
        # Reset the underlying environment
        self.env.reset()

        # Reset base policy (clear previously predicted actions)
        self.base_policy.reset()

        # Get base action from the base policy
        with torch.no_grad():
            base_action = None

        base_naction = self.action_scaler.scale(base_action)
        raw_obs = None 

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Store for later use in step
        self._last_base_naction = base_naction

        return augmented_obs, {}

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
        # Residual action is already scaled inside the Actor class
        # To ensure that we can use the same exploration for all dimensions,
        # we use the normalized actions as the action space
        # The normalized base action is stored as [-1, 1] in the replay buffer
        # and the residual action is predicted as action_scale * [-1, 1]
        combined_naction = self._last_base_naction + residual_naction

        # Unscale back to original action space for environment execution
        env_action = self.action_scaler.unscale(combined_naction)

        # Step the underlying vectorized environment
        raw_obs = self.env.step(env_action)
        reward, terminated, info = None, None, None # TODO

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
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])

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

