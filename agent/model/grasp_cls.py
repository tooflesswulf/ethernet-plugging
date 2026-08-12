"""
Trained crop classifier for "is the cable clamped in the gripper fingers?".

This is the learned counterpart to the hand-tuned cues in `agent/model/grasp.py`,
whose docstring calls for exactly this: the ROI carries the signal, the hue/specular
thresholds were just the cheap version and have to be recalibrated per connector.
A single image in, a single logit out.
"""
import torch
import torch.nn as nn

from agent.model.networks import get_resnet, replace_bn_with_gn

FEATURE_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}


class GraspClassifier(nn.Module):
    """ResNet trunk + linear head. forward() returns raw logits, shape (B,)."""

    def __init__(self, backbone='resnet18', pretrained=True, image_size=128, dropout=0.0, roi=None):
        super().__init__()
        if backbone not in FEATURE_DIMS:
            raise ValueError(f'Unsupported backbone {backbone!r}; expected one of {list(FEATURE_DIMS)}')
        # Kept so from_checkpoint can rebuild the model, and so eval code knows the
        # preprocessing (image size / ROI) the weights were trained with.
        self.config = dict(backbone=backbone, pretrained=pretrained, image_size=image_size,
                           dropout=dropout, roi=list(roi) if roi is not None else None)

        weights = 'IMAGENET1K_V1' if pretrained else None
        # GroupNorm instead of BatchNorm: EMA tracks parameters but not BN running
        # stats, and mixing EMA'd weights with live BN statistics wrecks accuracy
        # (same reason DiffusionPolicy does this to its vision encoder).
        self.encoder = replace_bn_with_gn(get_resnet(backbone, weights=weights))
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(FEATURE_DIMS[backbone], 1),
        )

    def forward(self, images):
        # images: (B, 3, H, W), already normalized by the dataset transform
        return self.head(self.encoder(images)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, images):
        """P(cable in fingers), shape (B,)."""
        return torch.sigmoid(self(images))

    @classmethod
    def from_checkpoint(cls, ckpt_path, device='cpu'):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert 'config' in checkpoint, f"Checkpoint {ckpt_path} has no 'config' entry"
        config = dict(checkpoint['config'])
        config['pretrained'] = False  # weights come from the checkpoint, not ImageNet
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Checkpoint] Loaded grasp classifier from {ckpt_path} (config: {checkpoint['config']})")
        return model.to(device).eval()
