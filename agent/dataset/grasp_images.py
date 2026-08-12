"""
Image-only dataset for the "is the cable in the gripper fingers?" classifier.

Reads the aggregated dataset written by `scripts/build_grasp_dataset.py`: a
`dataset.h5` whose `images` is a virtual dataset pointing back into the raw
episode files, plus a per-frame `label` (1 = cable held, 0 = not).

Only the image is used as input -- no pose, force or gripper width -- so the
classifier has to key on what is actually visible between the fingers, which is
the point: at eval time the gripper width alone cannot tell a held cable from a
gripper that closed on nothing.
"""
from dataclasses import dataclass
import pathlib

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import h5py

IMAGE_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_h5(dataset_path) -> pathlib.Path:
    """Accept either a dataset directory or the dataset.h5 itself."""
    path = pathlib.Path(dataset_path)
    return path if path.suffix == '.h5' else path / 'dataset.h5'


def source_image_shape(dataset_path) -> tuple:
    """(H, W, C) of the frames the dataset links to."""
    with h5py.File(resolve_h5(dataset_path), 'r') as f:
        return f['images'].shape[1:]


def expand_roi(roi, frac, width, height, roi_frame_size=None):
    """Grow an (x0, y0, x1, y1) box by `frac` on each side, clipped to the image.

    `roi_frame_size` is the frame width/height the box was measured in (GRASP_ROI is
    quoted for the 480x480 wrist frame); pass it to rescale the box when the frames
    on hand are a different size.
    """
    x0, y0, x1, y1 = roi
    if roi_frame_size is not None:
        sx, sy = width / roi_frame_size, height / roi_frame_size
        x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    box = (max(0, int(x0 - dx)), max(0, int(y0 - dy)),
           min(width, int(x1 + dx)), min(height, int(y1 + dy)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'ROI {roi} does not land inside a {width}x{height} frame (got {box}); '
                         f'pass roi_frame_size, or recalibrate the box for this camera')
    return box


@dataclass
class Split:
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_episodes: np.ndarray
    val_episodes: np.ndarray


def episode_split(dataset_path, val_frac=0.2, seed=0) -> Split:
    """Hold out whole EPISODES, never individual frames.

    Consecutive frames of an episode are near-duplicates (the wrist camera barely
    moves between them), so a frame-level split puts near-copies of validation
    frames in the training set and reports validation numbers that are far too
    optimistic. Success and empty episodes are split separately so both kinds land
    in each half.
    """
    with h5py.File(resolve_h5(dataset_path), 'r') as f:
        lengths = f['metadata/length'][:]
        is_empty = f['metadata/is_empty'][:]

    starts = np.concatenate([[0], np.cumsum(lengths)])
    rng = np.random.default_rng(seed)

    val_episodes = []
    for flag in (False, True):
        eps = np.flatnonzero(is_empty == flag)
        if len(eps) == 0:
            continue
        rng.shuffle(eps)
        n_val = min(len(eps) - 1, max(1, round(val_frac * len(eps))))
        val_episodes.extend(eps[:n_val])
    val_episodes = np.sort(np.array(val_episodes, dtype=int))
    train_episodes = np.setdiff1d(np.arange(len(lengths)), val_episodes)

    def frames(episodes):
        if len(episodes) == 0:
            return np.zeros(0, dtype=int)
        return np.concatenate([np.arange(starts[e], starts[e + 1]) for e in episodes])

    return Split(frames(train_episodes), frames(val_episodes), train_episodes, val_episodes)


class GraspImageDataset(torch.utils.data.Dataset):
    """Single frames -> binary label.

    Args:
        dataset_path: dataset directory (or dataset.h5) from build_grasp_dataset.py
        indices:      frame indices to use; None means every frame
        image_size:   square size the network sees
        augment:      random crop + colour jitter (train split only)
        roi:          (x0, y0, x1, y1) crop applied before resizing, in source-image
                      pixels. The camera is wrist-mounted, so the finger gap sits at a
                      fixed spot in frame and cropping to it throws away background
                      the classifier could otherwise latch onto. None = full frame.
    """

    def __init__(self, dataset_path, indices=None, image_size=IMAGE_SIZE, augment=False, roi=None):
        self.h5_path = resolve_h5(dataset_path)
        self.image_size = image_size
        self.roi = tuple(roi) if roi is not None else None

        with h5py.File(self.h5_path, 'r') as f:
            all_labels = f['label'][:]
            self.source_shape = f['images'].shape[1:]
            # build_grasp_dataset links the raw BGR frames without reordering channels
            self.bgr = f['images'].attrs.get('color_order', 'rgb') == 'bgr'

        self.indices = np.arange(len(all_labels)) if indices is None else np.asarray(indices)
        self.labels = all_labels[self.indices].astype(np.float32)
        self.h5 = None  # opened lazily, per worker process

        if augment:
            # No horizontal flip: the wrist camera is rigidly mounted and the fingers
            # are not left/right symmetric, so a mirrored frame is a view the model
            # will never see at eval time.
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.indices)

    @property
    def class_counts(self) -> tuple[int, int]:
        """(n_negative, n_positive)"""
        n_pos = int(self.labels.sum())
        return len(self.labels) - n_pos, n_pos

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        frame = self.h5['images'][self.indices[i]]
        if self.bgr:
            frame = frame[..., ::-1]
        img = Image.fromarray(np.ascontiguousarray(frame))
        if self.roi is not None:
            img = img.crop(self.roi)
        return self.transform(img), torch.tensor(self.labels[i])
