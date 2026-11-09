# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import math

import torch
from torch import nn

# Import vmap functionality (PyTorch 2.1+)
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional

from agent.finetuning.config.rlpd import CriticConfig
from agent.finetuning.off_policy.common_utils import utils


class Critic(nn.Module):
    def __init__(self,  prop_dim, action_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg.loss
        output_dim = 1

        # Number of Q-heads (ensemble size)
        num_q = getattr(cfg, "num_q", 2)
        input_dim = prop_dim + action_dim
        self.encoder = SingleMLP( input_dim, self.cfg.hidden_dim, self.cfg.hidden_dim, 2, cfg.use_layer_norm)

        # Build an ensemble that shares the spatial trunk and owns K independent heads
        self.q_ensemble = SpatialEmbQEnsemble(
            input_dim = self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim//2,
            orth=self.cfg.orth,
            output_dim=output_dim,
            num_heads=num_q,
            num_layers=cfg.num_layers,
            use_layer_norm=cfg.use_layer_norm,
        )


    def forward(self, prop, act, *, return_logits: bool = False):
        input = torch.cat([prop, act], -1);  feat = self.encoder(input)
        logits_per_head = self.q_ensemble(feat)

        # MSE case: outputs are already scalars per head → [num_q, B, 1]
        return logits_per_head

    def q_value(self, prop, act):
        """
        Returns the Q-value for a given feature, property, and action.
        I.e., gets all Q-values, subsets them, and returns the min.
        Used for critic loss computation (uses configurable min_q_heads).
        """
        # Get the Q-values for all heads
        q_out = self.forward( prop, act)

        # Take min over random subset of heads (configurable via min_q_heads)
        num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self,  prop, act):
        """
        Returns the Q-value for policy gradient computation.
        Supports different policy gradient types:
        - "ensemble_mean": mean over all heads (standard RED-Q approach)
        - "min_random_pair": min of configurable number of random heads (conservative variant)
        - "q1": just use q1 from ensemble (standard TD3 approach)
        """
        # Get the Q-values for all heads
        q_out = self.forward( prop, act)

        if self.cfg.policy_gradient_type == "ensemble_mean":
            # Take mean over all heads (standard RED-Q approach)
            return q_out.mean(dim=0)
        if self.cfg.policy_gradient_type == "min_random_pair":
            # Take min over random subset of heads (configurable via min_q_heads)
            num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
            idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
            return torch.min(q_out.index_select(0, idx), dim=0).values
        if self.cfg.policy_gradient_type == "q1":
            return q_out[0]
        raise ValueError(f"Unknown policy_gradient_type: {self.cfg.policy_gradient_type}")


class SpatialEmbQNet(nn.Module):
    def __init__(
        self,
        num_patch,
        patch_dim,
        prop_dim,
        action_dim,
        fuse_patch,
        emb_dim,
        hidden_dim,
        orth,
        output_dim=1,
        use_layer_norm=True,
    ):
        super().__init__()

        if fuse_patch:
            proj_in_dim = num_patch + action_dim + prop_dim
            num_proj = patch_dim
        else:
            proj_in_dim = patch_dim + action_dim + prop_dim
            num_proj = num_patch

        self.fuse_patch = fuse_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        # Build input projection layers
        input_layers = [nn.Linear(proj_in_dim, emb_dim)]
        if use_layer_norm:
            input_layers.append(nn.LayerNorm(emb_dim))
        input_layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*input_layers)

        self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
        nn.init.normal_(self.weight)

        # Build Q network layers
        q_layers = [nn.Linear(emb_dim + action_dim + prop_dim, hidden_dim)]
        if use_layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)])
        if use_layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim)])
        self.q = nn.Sequential(*q_layers)
        if orth:
            self.q.apply(utils.orth_weight_init)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor):
        assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"

        if self.fuse_patch:
            feat = feat.transpose(1, 2)

        repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
        all_feats = [feat, repeated_action]
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            all_feats.append(repeated_prop)

        x = torch.cat(all_feats, dim=-1)
        y: torch.Tensor = self.input_proj(x)
        z = (self.weight * y).sum(1)

        if self.prop_dim == 0:
            z = torch.cat((z, action), dim=-1)
        else:
            z = torch.cat((z, prop, action), dim=-1)

        q = self.q(z)
        # For MSE: [batch, 1], For HL-Gauss: [batch, K]
        return q  # noqa: RET504


class SingleMLP(nn.Module):
    """Single MLP head for the ensemble."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, use_layer_norm: bool = True):
        super().__init__()

        # Build layers dynamically based on num_layers
        layers = []
        current_dim = in_dim

        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())  # avoid inplace with vmap
            current_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(current_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            z: Input tensor [batch_size, in_dim]

        Returns:
            Output tensor [batch_size, out_dim]
        """
        return self.net(z)


class SpatialEmbQEnsemble(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        orth: int,
        output_dim: int = 1,
        num_heads: int = 2,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        # Trunk (shared across heads)
        self.input_dim = input_dim

        # vmap-based heads for efficient batched computation
        self.num_heads = num_heads
 
        # Create multiple module instances and stack their parameters/buffers
        heads = [SingleMLP(input_dim, hidden_dim, output_dim, num_layers, use_layer_norm) for _ in range(num_heads)]
        self.params, self.buffers = stack_module_state(heads)

        # Store a template head with no parameters (for structure only)
        self._head_template = SingleMLP(input_dim, hidden_dim, output_dim, num_layers, use_layer_norm)

        # Register params and buffers so they get moved with the module
        for name, param in self.params.items():
            self.register_parameter(f"_vmap_param_{name.replace('.', '_')}", nn.Parameter(param))
        for name, buffer in self.buffers.items():
            self.register_buffer(f"_vmap_buffer_{name.replace('.', '_')}", buffer)

        # Apply per-head initialization
        self._init_per_head_params(orth)

    def _init_per_head_params(self, orth: bool):
        """Apply per-head initialization to the stacked parameters."""
        with torch.no_grad():
            if orth:
                # Apply orthogonal initialization to all linear layers
                for key, param in self.params.items():
                    if "weight" in key and param.dim() >= 2:
                        # param shape: [num_heads, out_features, in_features] or similar
                        for h in range(self.num_heads):
                            if param[h].dim() >= 2:
                                utils.orth_weight_init(param[h])

    def extra_repr(self) -> str:
        return f"heads: {self.num_heads}"

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = feat

        # Reconstruct params/buffers from registered tensors (ensures proper device placement)
        current_params = {}
        current_buffers = {}

        for name in self.params:
            param_name = f"_vmap_param_{name.replace('.', '_')}"
            current_params[name] = getattr(self, param_name)

        for name in self.buffers:
            buffer_name = f"_vmap_buffer_{name.replace('.', '_')}"
            current_buffers[name] = getattr(self, buffer_name)

        def f_one_head(p, b, z_input):
            # p, b are the param/buffer for one head; z_input is [B, in_dim]
            return functional_call(self._head_template, (p, b), (z_input,))  # [B, out_dim]

        # Vectorize across heads; params/buffers carry leading H; z is shared across heads
        # Output: [num_heads, batch_size, output_dim]
        return vmap(f_one_head, in_dims=(0, 0, None))(current_params, current_buffers, z)