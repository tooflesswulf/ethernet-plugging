# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from agent.model.networks import TemporalConvEncoder
from agent.finetuning.config.rlpd import ActorConfig
from agent.finetuning.off_policy.common_utils import utils


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)



class Actor(nn.Module):
    def __init__(self, condition_length, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False):
        super().__init__()
        self.condition_length = condition_length
        self.prop_dim = prop_dim
        self.residual_actor = residual_actor
        self.cfg = cfg
        self.valid_dim = cfg.valid_dim
        self.valid_mask = torch.zeros((action_dim)); self.valid_mask[:self.valid_dim] = 1.0
        # temporal compressor (o_{t-w}, ..., o_t) -> emb0
        layers = [TemporalConvEncoder(input_dim=prop_dim, feature_dim=cfg.feature_dim, history_len=condition_length )]
        if cfg.use_layer_norm:
                    layers.append(nn.LayerNorm(cfg.feature_dim))
        self.temp_compress = nn.Sequential(*layers)

        # compressor: (o_t, a_t) -> emb1
        layers = [nn.Linear( prop_dim + action_dim, cfg.feature_dim)]
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.feature_dim))
        layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])
        self.compress = nn.Sequential(*layers)
       

        policy_in_dim = cfg.feature_dim * 2
        # Create policy network
        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            action_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )

        # Apply weight initialization
        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        """Apply weight initialization to all networks."""
        # Determine initialization distributions
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        # Initialize compression layers
        if cfg.orth:
            # Backward compatibility: use existing orthogonal initialization
            self.compress.apply(utils.orth_weight_init)
        else:
            # Use the specified distribution for compression layers
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        # Initialize policy network intermediate layers (exclude final layer)
        utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

        # Initialize final layer with specific configuration if provided
        if cfg.actor_last_layer_init_scale is not None:
            final_layer = None
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break

            if final_layer is not None:
                utils.initialize_layer_weights(
                    final_layer,
                    cfg.actor_last_layer_init_distribution,
                    cfg.actor_last_layer_init_scale,
                )

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        states = obs["observation.state"] # BxTxD
        base_actions = obs["observation.base_action"] # BxTXD
        cur_state, cur_base_action = states[:,-1], base_actions[:, -1]; cur_input = torch.cat([cur_state, cur_base_action], -1)
        temp_feat = self.temp_compress(states); feat = self.compress(cur_input)
        policy_input = torch.cat([temp_feat, feat], dim=-1)
       
        mu: torch.Tensor = self.policy(policy_input); valid_mask = self.valid_mask.to(mu.device).float().unsqueeze(0)
       
        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)
        action_dist.loc *= valid_mask; action_dist.scale *= valid_mask
        return action_dist  # noqa: RET504