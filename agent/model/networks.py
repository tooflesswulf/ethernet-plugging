from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision


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