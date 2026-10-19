"""Failure-detection classifier (WIP): encoders over image, pose and force."""
import torch
import torch.nn as nn


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
    from agent.failure.dataset import FailureDetectDataset
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
