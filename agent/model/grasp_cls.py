"""
Trained crop classifier for "is the cable clamped in the gripper fingers?".

This is the learned counterpart to the hand-tuned cues in `agent/model/grasp.py`,
whose docstring calls for exactly this: the ROI carries the signal, the hue/specular
thresholds were just the cheap version and have to be recalibrated per connector.
A single image in, a single logit out.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from agent.dataset.grasp_images import eval_transform
from agent.model.networks import get_resnet, replace_bn_with_gn

FEATURE_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}


class GraspClassifier(nn.Module):
    """ResNet trunk + linear head. forward() returns raw logits, shape (B,)."""

    def __init__(self, backbone='resnet18', pretrained=True, image_size=128, dropout=0.0,
                 roi=None, frame_shape=None):
        super().__init__()
        if backbone not in FEATURE_DIMS:
            raise ValueError(f'Unsupported backbone {backbone!r}; expected one of {list(FEATURE_DIMS)}')
        # Kept so from_checkpoint can rebuild the model, and so eval code knows the
        # preprocessing (image size / ROI, and the frame size the ROI is in pixels of)
        # the weights were trained with.
        self.config = dict(backbone=backbone, pretrained=pretrained, image_size=image_size,
                           dropout=dropout, roi=list(roi) if roi is not None else None,
                           frame_shape=list(frame_shape) if frame_shape is not None else None)

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


class NeuralGraspDetector:
    """Streaming wrapper with the same interface as `agent.model.grasp.GraspDetector`
    (`held` / `update` / `reset`), so it drops straight into the teleop loop.

    Preprocessing is taken from the checkpoint config -- same ROI crop, image size and
    normalization the model was trained with -- and the debounce is the same
    majority vote over the last `debounce` frames, for the same reason: a single
    blurred or occluded frame must not flip the state machine.

    Usage:
        det = NeuralGraspDetector('logs/grasp-cls/ckpt_best.pth', device='cuda:0')
        held = det.update(rgb_frame)   # bool
        p = det.last_proba             # P(held) of the most recent frame
    """

    def __init__(self, ckpt_path, device='cuda:0', threshold=0.5, debounce=5):
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f'[GraspDetector] {device} unavailable, falling back to CPU')
            device = 'cpu'
        self.device = device
        self.threshold = threshold
        self.model = GraspClassifier.from_checkpoint(ckpt_path, device=device)
        self.image_size = self.model.config['image_size']
        self.roi = self.model.config.get('roi')
        self.frame_shape = self.model.config.get('frame_shape')
        self.transform = eval_transform(self.image_size)
        self._buf: deque = deque(maxlen=debounce)
        self._warned_shape = False
        self.last_proba: float | None = None

        # Burn the lazy CUDA init / cuDNN autotune now rather than inside the control loop.
        size = self.image_size
        with torch.no_grad():
            self.model(torch.zeros(1, 3, size, size, device=device))

    def _crop_box(self, width, height):
        """ROI in this frame's pixels, rescaled if the live camera resolution differs
        from the frames the model was trained on."""
        if self.roi is None:
            return None
        if self.frame_shape is None or (height, width) == tuple(self.frame_shape[:2]):
            return tuple(self.roi)
        sx = width / self.frame_shape[1]
        sy = height / self.frame_shape[0]
        if not self._warned_shape:
            print(f'[GraspDetector] live frame {width}x{height} differs from the trained '
                  f'{self.frame_shape[1]}x{self.frame_shape[0]}; rescaling ROI by ({sx:.3f}, {sy:.3f})')
            self._warned_shape = True
        x0, y0, x1, y1 = self.roi
        return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

    @torch.no_grad()
    def proba(self, image) -> float:
        """P(cable in fingers) for one RGB frame."""
        img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        img = img.convert('RGB')
        box = self._crop_box(*img.size)
        if box is not None:
            img = img.crop(box)
        x = self.transform(img).unsqueeze(0).to(self.device)
        self.last_proba = float(torch.sigmoid(self.model(x))[0])
        return self.last_proba

    def held(self, image) -> bool:
        """Instantaneous grasp decision (no debounce)."""
        return self.proba(image) > self.threshold

    def update(self, image) -> bool:
        """Debounced grasp state -- majority vote over the last `debounce` frames."""
        self._buf.append(self.held(image))
        return sum(self._buf) > len(self._buf) // 2

    def reset(self):
        self._buf.clear()
        self.last_proba = None
