from typing import Tuple, Sequence, Dict, Union, Optional, Callable
from einops import rearrange, repeat
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision

from agent.model.modules import RandomShiftsAug, ResidualMLP, SpatialEmb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8
                 ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module


############ For failur detection

class ImageEncoder(nn.Module):
    """
    Input:
        (B, 1, 128, 128, 3)

    Output:
        (B, 256)
    """

    def __init__(self, out_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),   # 64x64
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 8x8
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim
        self.proj = nn.Linear(256, out_dim)

    def forward(self, image):
        # image: (B,1,H,W,3)
        assert image.shape[1] == 1, f"Only support single image now"
        image = image[:, 0]                 # (B,H,W,3)
        image = image.permute(0, 3, 1, 2)   # (B,3,H,W)

        feat = self.backbone(image)
        feat = feat.flatten(1)

        return self.proj(feat)
    
class PoseEncoder(nn.Module):
    """
    pose:
        (B,1,6)

    gripper:
        (B,1,1)

    Output:
        (B,64)
    """

    def __init__(self, hidden=128, out_dim=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),

            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, pose):
        assert pose.shape[1] == 1, f"only support single pose now"
        x = pose[:, 0]
        return self.mlp(x)
    
class ForceEncoder(nn.Module):
    """
    Input:
        (B,8,6)

    Output:
        (B,64)
    """

    def __init__(self, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def forward(self, force):

        # (B,8,6) -> (B,6,8)
        force = force.transpose(1, 2)

        feat = self.net(force)
        feat = feat.squeeze(-1)

        return self.proj(feat)
    
class FusionEncoder(nn.Module):
    """
    image : (B,256)
    pose  : (B,64)
    force : (B,64)

    Output:
        (B,256)
    """

    def __init__(self, input_dim, out_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, out_dim),
        )

    def forward(self, image_feat, pose_feat=None, force_feat=None):
        x = image_feat 
        if pose_feat is not None:
            x = torch.cat([x, pose_feat], dim = -1)
        if force_feat is not None:
            x = torch.cat([x, force_feat], dim = -1)
       
        return self.net(x)

class MultiEncoder(nn.Module):
    """
    Multi-modal encoder for binary classification.

    Inputs
    ------
    image : (B, 1, 128, 128, 3)
    pose : (B, 1, 6)
    gripper_width : (B, 1, 1)
    force : (B, 8, 6)

    Outputs
    -------
    logits : (B,)
        Binary classification logits.

    embedding : (B, 256)
        Joint multimodal embedding.
    """

    def __init__(
        self,
        modality = ['image', 'pose', 'force'],
        image_dim=256,
        pose_dim=64,
        force_dim=64,
        fusion_dim=256,
    ):
        super().__init__()

        # Sub-encoders
        self.modality = modality
        sub_encoders = {
            'image': ImageEncoder(out_dim=image_dim),
            'pose':  PoseEncoder(out_dim=pose_dim),
            'force': ForceEncoder(out_dim=force_dim)
        }
        self.sub_encoders = nn.ModuleDict({k: sub_encoders[k] for k in modality})
        
        # Fuse multi-modal encodings
        input_dim = sum([ v.out_dim for k,v in self.sub_encoders.items()])
        self.fusion_encoder = FusionEncoder(input_dim, out_dim=fusion_dim)

        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def encode(
        self,
        image,
        pose=None,
        force=None,
    ):
        """
        Returns the fused embedding only.
        """

        image_feat = self.sub_encoders['image'](image)
        pose_feat = self.sub_encoders['pose'](pose) if 'pose' in self.modality else None
        force_feat = self.sub_encoders['force'](force) if 'force' in self.modality else None

        embedding = self.fusion_encoder( image_feat, pose_feat, force_feat, )

        return embedding

    def forward(
        self,
        image,
        pose,
        force,
    ):
        embedding = self.encode(
            image=image,
            pose=pose,
            force=force,
        )

        logits = self.classifier(embedding).squeeze(-1)

        return {
            "logits": logits,
            "embedding": embedding,
        }
"""
Double Unet with cross-attention
"""
class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for fixed-length observation token sequences.
    """

    def __init__(self, embed_dim: int, max_position: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_position = max_position

        position = torch.arange(max_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0) / embed_dim)))

        pe = torch.zeros(max_position, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positions.shape[:2] != x.shape[:2]:
            raise ValueError(f"positions shape {positions.shape} must match token shape {x.shape[:2]}")
        pos_embed = self.pe[positions.long()].to(dtype=x.dtype, device=x.device)
        if token_mask is not None:
            pos_embed = pos_embed * token_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x + pos_embed
        
class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention block where trajectory features attend to conditioning tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        cond_token_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 2,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.cond_proj = nn.Linear(cond_token_dim, embed_dim)
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        if self.use_temporal_pos_emb:
            self.temporal_pos_emb = TemporalPositionalEncoding(
                embed_dim=embed_dim,
                max_position=max_temporal_position,
            )
        if self.use_modality_emb:
            self.modality_emb = nn.Embedding(max_modalities, embed_dim)
        if self.use_range_emb:
            # 0: NULL, 1: LONG, 2: SHORT
            self.range_emb = nn.Embedding(max_ranges, embed_dim)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.cond_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        range_indices: Optional[torch.Tensor] = None,
        obs_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, C), cond_tokens: (B, N, D_cond) — already in unified_token_dim space
        cond = self.cond_proj(cond_tokens)

        if self.use_modality_emb:
            if modality_indices is None:
                raise ValueError("modality_indices required when use_modality_emb=True")
            cond = cond + self.modality_emb(modality_indices).to(dtype=cond.dtype)

        if self.use_range_emb:
            if range_indices is None:
                raise ValueError("range_indices required when use_range_emb=True")
            cond = cond + self.range_emb(range_indices).to(dtype=cond.dtype)

        if self.use_temporal_pos_emb:
            if temporal_positions is None:
                raise ValueError("temporal_positions required when use_temporal_pos_emb=True")
            cond = self.temporal_pos_emb(
                x=cond,
                positions=temporal_positions,
                token_mask=obs_token_mask,
            )

        cond_normed = self.cond_norm(cond)
        attn_out, _ = self.cross_attn(
            query=self.query_norm(x),
            key=cond_normed,
            value=cond_normed,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x
        
class AttentionConditionalResidualBlock1D(nn.Module):
    """
    Residual Conv1D block with cross-attention conditioning between the conv blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_token_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 2,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cross_attention = CrossAttentionConditioning(
            embed_dim=out_channels,
            cond_token_dim=cond_token_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            use_temporal_pos_emb=use_temporal_pos_emb,
            max_temporal_position=max_temporal_position,
            use_modality_emb=use_modality_emb,
            max_modalities=max_modalities,
            use_range_emb=use_range_emb,
            max_ranges=max_ranges,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        range_indices: Optional[torch.Tensor] = None,
        obs_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.blocks[0](x)
        out = self.cross_attention(
            x=out.transpose(1, 2).contiguous(),
            cond_tokens=cond_tokens,
            temporal_positions=temporal_positions,
            modality_indices=modality_indices,
            range_indices=range_indices,
            obs_token_mask=obs_token_mask,
        )
        out = out.transpose(1, 2).contiguous()
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
        
class CrossUnet1D2(nn.Module):
    """
    1D U-Net with attention conditioning for fixed, static observation horizons.

    backbone / ctx_backbone None: state-only; the observation tokens are the embedded states
    (cond_mlp / ctx_mlp) without the image features they otherwise compress into.
    """

    def __init__(
        self,
        backbone,
        ctx_backbone,
        input_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
        cond_steps=1,
        context_steps=4,
        cond_dim=None,
        cond_mlp_dims: list = [256],
        context_dim=None,
        short_range_dropout = 0.3,
        spatial_emb=128,
        num_img = 1,
        augment=False,
        ablate_context=False,
    ):
        super().__init__()
        self.short_range_dropout = short_range_dropout; self.num_img = num_img; self.ablate_context = ablate_context
        if self.ablate_context:
            short_range_dropout = 0.0
        self.backbone = backbone; self.ctx_backbone = ctx_backbone; self.cond_steps = cond_steps; self.context_steps = context_steps; self.cond_dim = cond_dim; self.context_dim = context_dim
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            visual_feature_dim = spatial_emb # * num_img
            if num_img > 1:
                assert False, f"Not implemented"
            elif backbone is not None:  # TODO: clean up
                self.compress,  self.ctx_compress = [SpatialEmb(
                    num_patch=bkbone.num_patch,
                    patch_dim=bkbone.patch_repr_dim,
                    prop_dim=spatial_emb,
                    proj_dim=spatial_emb,
                    dropout=0,
                ) for bkbone in [self.backbone, self.ctx_backbone] ]

            assert num_img == 1, f"Not supporting num_img > 1 yet, num_img to token dimension better?"
      
        self.visual_feature_dim = visual_feature_dim; self.short_range_null_token = nn.Parameter(torch.zeros(1, self.visual_feature_dim))
        self.cond_mlp, self.ctx_mlp = [ResidualMLP( # embed state to visual feature dim
                dim_list=[_cond_dim] + cond_mlp_dims + [self.visual_feature_dim],
                activation_type="Mish",
                out_activation_type="Identity",
            ) for _cond_dim in [cond_dim, context_dim] ]
        
        global_cond_dim = visual_feature_dim  # assume visual feature from short/long context the same.
        self.global_cond_dim = global_cond_dim
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        # Unified token dimension: all conditioning tokens (obs + timestep) are projected
        # to this dimension ONCE here in the UNet, then shared across every attention block.
        # This means the raw encoder output passes through global_to_token exactly once,
        # so backward gradients flow through one matrix rather than ~12 independent cond_proj
        # matrices, preventing gradient amplification that drives the encoder toward NaN.
        unified_token_dim = down_dims[0]
        self.unified_token_dim = unified_token_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dim=diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        # Both projections map into unified_token_dim so all tokens share one space
        self.timestep_to_token = nn.Linear(diffusion_step_embed_dim, unified_token_dim)
        self.global_to_token = nn.Linear(global_cond_dim, unified_token_dim)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                AttentionConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_token_dim=unified_token_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    use_temporal_pos_emb=use_temporal_pos_emb,
                    max_temporal_position=max_temporal_position,
                    use_modality_emb=use_modality_emb,
                    max_modalities=max_modalities,
                    use_range_emb=use_range_emb,
                    max_ranges=max_ranges,
                ),
                AttentionConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_token_dim=unified_token_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    use_temporal_pos_emb=use_temporal_pos_emb,
                    max_temporal_position=max_temporal_position,
                    use_modality_emb=use_modality_emb,
                    max_modalities=max_modalities,
                    use_range_emb=use_range_emb,
                    max_ranges=max_ranges,
                ),
            ]
        )

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        AttentionConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_token_dim=unified_token_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        AttentionConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_token_dim=unified_token_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        AttentionConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_token_dim=unified_token_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        AttentionConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_token_dim=unified_token_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout,
                            use_temporal_pos_emb=use_temporal_pos_emb,
                            max_temporal_position=max_temporal_position,
                            use_modality_emb=use_modality_emb,
                            max_modalities=max_modalities,
                            use_range_emb=use_range_emb,
                            max_ranges=max_ranges,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.up_modules = up_modules
        self.down_modules = down_modules

    def _expand_timesteps(
        self,
        timesteps: Union[torch.Tensor, float, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize the diffusion timestep into a (batch_size,) long tensor.

        Accepts a scalar (Python int/float), a 0-d tensor, or an already-batched
        1-d tensor, and returns a 1-d tensor of length batch_size on the correct
        device — ready to be passed to the timestep embedding.
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(device)
        else:
            timesteps = timesteps.to(device)
        return timesteps.expand(batch_size)

    def _prepare_cond_tokens(
        self,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor],
        temporal_positions: Optional[torch.Tensor],
        modality_indices: Optional[torch.Tensor],
        range_indices: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple:
        # Observation tokens: project from encoder output dim to unified_token_dim ONCE here.
        # All attention blocks then receive tokens already in unified_token_dim space, so
        # each block's cond_proj only lifts from unified_token_dim → embed_dim (not from
        # the raw encoder dim). This keeps backward gradients to the encoder bounded.
        if global_cond is None:
            model_dtype = self.timestep_to_token.weight.dtype
            obs_tokens = torch.zeros(batch_size, 0, self.unified_token_dim, device=device, dtype=model_dtype)
            obs_temporal_positions = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_modality_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_range_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        else:
            if global_cond.ndim == 2:
                global_cond = global_cond.unsqueeze(1)
            if global_cond.ndim != 3:
                raise ValueError(f"global_cond must have shape (B, N, D) or (B, D), got {global_cond.shape}")
            if global_cond.shape[0] != batch_size:
                raise ValueError(f"global_cond batch {global_cond.shape[0]} != sample batch {batch_size}")
            if global_cond.shape[-1] != self.global_cond_dim:
                raise ValueError(f"global_cond dim {global_cond.shape[-1]} != expected {self.global_cond_dim}")
            obs_tokens = self.global_to_token(global_cond)  # (B, N, unified_token_dim)
            n_obs_tokens = obs_tokens.shape[1]

            if self.use_temporal_pos_emb:
                if temporal_positions is None:
                    raise ValueError("temporal_positions is required when use_temporal_pos_emb=True")
                obs_temporal_positions = temporal_positions
            else:
                obs_temporal_positions = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            if self.use_modality_emb:
                if modality_indices is None:
                    raise ValueError("modality_indices is required when use_modality_emb=True")
                obs_modality_indices = modality_indices
            else:
                obs_modality_indices = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            if self.use_range_emb:
                if range_indices is None:
                    raise ValueError("range_indices is required when use_range_emb=True")
                obs_range_indices = range_indices
            else:
                obs_range_indices = torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)

        # Diffusion timestep token
        timesteps = self._expand_timesteps(timestep, batch_size, device)
        timestep_embed = self.diffusion_step_encoder(timesteps)
        timestep_token = self.timestep_to_token(timestep_embed).unsqueeze(1)
        timestep_zeros = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        timestep_obs_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Keep token ordering consistent with conditional_unet1d_attention:
        # [timestep_token, obs_tokens...]
        cond_tokens = torch.cat([timestep_token, obs_tokens], dim=1)
        temporal_positions = torch.cat([timestep_zeros, obs_temporal_positions], dim=1)
        modality_indices = torch.cat([timestep_zeros, obs_modality_indices], dim=1)
        range_indices = torch.cat([timestep_zeros, obs_range_indices], dim=1)
        # Mask that indicates which tokens are true observations (vs diffusion timestep)
        obs_token_mask = torch.cat(
            [
                timestep_obs_mask,
                torch.ones(batch_size, obs_tokens.shape[1], dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        return cond_tokens, temporal_positions, modality_indices, range_indices, obs_token_mask

    def _encode_obs(self, cond: Dict[str, torch.Tensor], B: int) -> tuple:
        """
        Encode n_obs_steps observation timesteps into a cross-attention token sequence.

        With dual encoders (short_range_obs_horizon not None): the oldest
        (n_obs_steps - short_range_obs_horizon) frames are encoded by obs_encoder as
        LONG-range tokens, and the most recent short_range_obs_horizon frames are encoded
        by short_range_encoder as SHORT-range tokens. During training, short-range tokens
        are replaced with a learned null token with probability short_range_dropout.

        Args:
            cond: normalized obs dict, each value shape (B, To, ...)
            B:    batch size

        Returns:
            tokens:     (B, To, obs_feature_dim) — one token per obs timestep
            positions:  (B, To) — temporal index; 0 = oldest, To-1 = most recent
            modalities: (B, To) — all 1 (single hybrid-encoder modality)
            ranges:     (B, To) — 1 = LONG, 2 = SHORT
        """
        cond_B = cond["state"].shape[0]
        duplicates = B // cond_B # use for sample multiple action plans from the same observation

        # encode state: B x (Ts or Th) x d -> BTs x D or BTh x D
        ctx_state = rearrange(  cond["state"], 'B T d -> (B T) d'); state = rearrange(cond["state"][:,-self.cond_steps:], 'B T d -> (B T) d')
        ctx_state = self.ctx_mlp(ctx_state ); state = self.cond_mlp(state )
       
        if self.backbone is None:  # state-only: one token per embedded state
            feat, ctx_feat = state, ctx_state
        else:
            # encode images: B x T x C x H x W -> B x D
            rgb = cond["rgb"][:, -self.cond_steps:]; ctx_rgb = cond["rgb"][:, -self.context_steps:]
            if self.num_img > 1:
                assert False, f"Not implemented"
            else:
                rgb = rearrange(rgb, "b t c h w -> (b t) c h w"); ctx_rgb = rearrange(ctx_rgb, "b t c h w -> (b t) c h w")
            # convert rgb to float32 for augmentation
            rgb = rgb.float(); ctx_rgb = ctx_rgb.float()
        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            assert False, f"Not Implemented"
        else:  # single image
            if self.backbone is not None:
                if self.augment and self.training:
                    rgb = self.aug(rgb); ctx_rgb = self.aug(ctx_rgb)
                # feat: BxtokensxD; ctx_feat: (BxT)xtokensxD
                feat = self.backbone(rgb); ctx_feat = self.ctx_backbone(ctx_rgb)
                # compress
                feat = self.compress.forward(feat, state); ctx_feat = self.ctx_compress.forward(ctx_feat, ctx_state)
            if self.training and self.short_range_dropout > 0.0:
                drop_mask = torch.bernoulli(
                    torch.full((cond_B,), self.short_range_dropout, device=feat.device)
                ).bool()  # (B,) — True means replace this sample's short-range tokens
                if drop_mask.any():
                    # feat is (B cond_steps) x d: drop all of a sample's short-range frames together
                    drop_mask = drop_mask.repeat_interleave(self.cond_steps)
                    null = self.short_range_null_token.view(1, -1).expand(feat.shape[0], -1)
                    feat = torch.where(drop_mask.view(-1, 1), null, feat)
        feat = rearrange(feat, '(B T) d -> B T d', T=self.cond_steps); ctx_feat = rearrange(ctx_feat, '(B T) d -> B T d', T=self.context_steps)
        cond_encoded = torch.cat([ctx_feat, feat], dim=1) if not self.ablate_context else feat
        if duplicates > 1:
            cond_encoded =  repeat(cond_encoded, "B N d -> (B duplicates) N d", duplicates = duplicates)
        tokens = cond_encoded
        long_positions = torch.arange(self.context_steps, device=tokens.device).unsqueeze(0).expand(B, -1)
        short_positions = torch.arange(self.context_steps-self.cond_steps, self.context_steps, device=tokens.device).unsqueeze(0).expand(B, -1)
        positions = torch.cat([long_positions, short_positions], dim=1) if not self.ablate_context else short_positions
        long_ranges = torch.ones(B, self.context_steps, dtype=torch.long, device=tokens.device)
        short_ranges = torch.full((B, self.cond_steps), 2, dtype=torch.long, device=tokens.device)
        ranges = torch.cat([long_ranges, short_ranges], dim=1) if not self.ablate_context else short_ranges
        modalities = torch.ones(B, tokens.shape[1], dtype=torch.long, device=tokens.device)
        
        return tokens, positions, modalities, ranges

    def forward(
        self,
        sample,
        timestep,
        cond,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            timestep: (B,) or scalar diffusion step
            cond: dicionary of observation
           
        Returns:
            (B, T, input_dim)
        """

        global_cond, temporal_positions, modality_indices, range_indices = self._encode_obs(cond, B=len(sample))
      
        """
        global_cond: (B, N, global_cond_dim) observation tokens
        temporal_positions: (B, N) temporal indices for each observation token
        modality_indices: (B, N) modality indices for each observation token
        range_indices: (B, N) short/long range indices for each observation token
        """

        # Run the entire UNet in fp32.
        #
        # Why only this UNet and not the FiLM version: cross-attention adds TWO extra
        # residual connections per block (attn residual + FFN residual) on top of the
        # standard ResNet skip. With 12 blocks stacked, gradients at deep conv layers
        # (e.g. Conv1d 512→1024, k=5) accumulate contributions from all downstream
        # residuals and can exceed bf16 range (~65504), producing NaN. FiLM modulates
        # via scale/shift, which does not add an independent gradient path, so the same
        # conv layers stay stable in bf16 under FiLM conditioning.
        orig_dtype = sample.dtype
        with torch.autocast(device_type=sample.device.type, enabled=False):
            sample = sample.float()
            if global_cond is not None:
                global_cond = global_cond.float()
        
            sample = sample.permute(0, 2, 1).contiguous()  # "b h t -> b t h"
            batch_size = sample.shape[0]
            # Remember the original sequence length so we can restore it after up-sampling.
            # With down_dims=[d0, d1, d2] the UNet performs two stride-2 downsamples and two
            # stride-2 upsamples. For odd horizons (e.g. 17), ceil-division during the
            # downsamples followed by exact 2x upsamples produces an output one element longer
            # than the input (e.g. 17 -> 9 -> 5 -> 10 -> 18). We truncate at the end to undo this.
            original_horizon = sample.shape[-1]

            (
                cond_tokens,
                cond_temporal_positions,
                cond_modality_indices,
                cond_range_indices,
                cond_obs_token_mask,
            ) = self._prepare_cond_tokens(
                timestep=timestep,
                global_cond=global_cond,
                temporal_positions=temporal_positions,
                modality_indices=modality_indices,
                range_indices=range_indices,
                batch_size=batch_size,
                device=sample.device,
            )
            cond_tokens = cond_tokens.to(dtype=sample.dtype, device=sample.device)

            h_local = []
            x = sample
            h = []
            for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                x = resnet(
                    x,
                    cond_tokens,
                    temporal_positions=cond_temporal_positions,
                    modality_indices=cond_modality_indices,
                    range_indices=cond_range_indices,
                    obs_token_mask=cond_obs_token_mask,
                )
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(
                    x,
                    cond_tokens,
                    temporal_positions=cond_temporal_positions,
                    modality_indices=cond_modality_indices,
                    range_indices=cond_range_indices,
                    obs_token_mask=cond_obs_token_mask,
                )
                h.append(x)
                x = downsample(x)

            for mid_module in self.mid_modules:
                x = mid_module(
                    x,
                    cond_tokens,
                    temporal_positions=cond_temporal_positions,
                    modality_indices=cond_modality_indices,
                    range_indices=cond_range_indices,
                    obs_token_mask=cond_obs_token_mask,
                )

            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                h_pop = h.pop()
                if x.shape[-1] != h_pop.shape[-1]:
                    x = x[..., :h_pop.shape[-1]]
                x = torch.cat((x, h_pop), dim=1)
                x = resnet(
                    x,
                    cond_tokens,
                    temporal_positions=cond_temporal_positions,
                    modality_indices=cond_modality_indices,
                    range_indices=cond_range_indices,
                    obs_token_mask=cond_obs_token_mask,
                )
                if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(
                    x,
                    cond_tokens,
                    temporal_positions=cond_temporal_positions,
                    modality_indices=cond_modality_indices,
                    range_indices=cond_range_indices,
                    obs_token_mask=cond_obs_token_mask,
                )
                x = upsample(x)

            if x.shape[-1] != original_horizon:
                x = x[..., :original_horizon]

            x = self.final_conv(x)
            x = x.permute(0, 2, 1).contiguous()  # "b t h -> b h t"
        return x.to(orig_dtype)
        
if __name__ == '__main__':
    from agent.dataset.sequence import FailureDetectDataset
    dataset_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc'
    label_path = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc/label.csv'
    dataset = FailureDetectDataset(dataset_path, label_path)

    from torch.utils.data import  DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
    )
    (data, label) = next(iter(dataloader) )
    encoder = MultiEncoder().to('cuda')
    image = data['image'] # B x T x H x W x C
    gripper = data['gripper_width'].unsqueeze(-1)
    pose = torch.cat( [data['pose'], gripper], dim=-1) # B x T x D
    force = data['force'] # B x T x D
    out = encoder(image, pose, force)
    assert False, f"{out['logits'].shape} {out['embedding'].shape}"
