"""Lightweight cable-grasp + coarse-location estimators for the wrist camera.

Replaces the Qwen VLM for the two things it was bad at / overkill for:

  * "is the cable clamped in the gripper fingers?"  -> GraspDetector
  * "roughly where is the cable in the scene?"      -> cable_region (kinematics)

Design, and why it is this simple (from the qwen-debug investigation):

  * The camera is WRIST-MOUNTED, so the gripper -- and a plug held in it --
    appears at a STABLE image location, GRASP_ROI. Grasp detection is not "find
    the cable"; it is "is a plug in this fixed box". A plug on a nearby stand
    falls OUTSIDE the box and is ignored (this broke every VLM prompt).

  * Key on the CONNECTOR, not the background. Neither brightness nor edges work
    across setups: a grasped plug and empty cardboard have the same intensity;
    edges fire on cluttered-switch backgrounds whether or not a plug is held
    (that is why held-in-clutter cases defeat an edge threshold). What is stable
    is the plug's own appearance in the gap:
      - a coloured connector boot  -> saturated pixels in a known hue band ("warm")
      - a bare metal shield        -> small specular highlight ("spec")
    Everything else in the gap (cardboard, gripper, switch plastic) is dull/matte.
    On the debug frames this separates held from empty by a huge margin
    (warm 0.18-0.27 held vs <=0.001 empty) in BOTH plain and cluttered scenes.

  * Robustness during autonomous exploration: debounce the per-frame decision so
    blur/occlusion cannot flip it; a grasped->empty transition you did not command
    is your 'dropped' signal.

  * LIMITATION: these thresholds are tuned to THIS connector's colour/finish. If
    you change to a differently-coloured plug or the lighting shifts a lot,
    recalibrate hue_band / thresholds (see calibrate()) or move to a small trained
    crop classifier -- the crop carries the signal, hand thresholds are just the
    cheap version. A plug half-swallowed by the switch (dark, occluded) reads
    'not held'; that case is treated as acceptable/ambiguous.
"""
from collections import deque

import numpy as np
from PIL import Image

# Tight finger-gap ROI (x0, y0, x1, y1) in the 480x480 wrist frame -- just where a
# grasped plug sits. Valid because the camera is rigidly wrist-mounted.
GRASP_ROI = (263, 185, 322, 250)

# Connector-boot hue band in PIL HSV units (0-255; ~21-63 deg == yellow/gold),
# matching this cable's warm boot. Recalibrate for a differently coloured plug.
WARM_HUE = (15, 45)


def _as_pil(img) -> Image.Image:
    return img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))


def connector_cues(img, roi=GRASP_ROI, hue_band=WARM_HUE) -> tuple[float, float]:
    """(warm, spec) fractions in the ROI.

    warm: fraction of saturated pixels in the connector-boot hue band (coloured
          plug body). spec: fraction of bright, low-saturation pixels (bare metal
          shield highlight). A held plug lights up one or the other; an empty gap
          lights up neither, regardless of what the background is.
    """
    c = _as_pil(img).convert("RGB").crop(roi)
    hsv = np.asarray(c.convert("HSV"), np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = hue_band
    warm = float(((S > 60) & (V > 60) & (H > lo) & (H < hi)).mean())
    spec = float(((V > 200) & (S < 60)).mean())
    return warm, spec


class GraspDetector:
    """Connector-cue grasp detector over the fixed finger-gap ROI.

    held = (warm > warm_thr) OR (spec > spec_thr), debounced. No training.

    Usage (streaming):
        det = GraspDetector()
        held = det.update(frame)          # bool

    Calibrate once with calibrate(): thresholds sit between empty (~0) and held
    (warm ~0.2 / spec ~0.03). Defaults leave a wide margin.
    """

    def __init__(self, roi=GRASP_ROI, warm_thr: float = 0.05, spec_thr: float = 0.02,
                 hue_band=WARM_HUE, debounce: int = 5):
        self.roi = roi
        self.warm_thr = warm_thr
        self.spec_thr = spec_thr
        self.hue_band = hue_band
        self._buf: deque = deque(maxlen=debounce)

    def cues(self, image) -> tuple[float, float]:
        """(warm, spec) for one frame -- useful for calibrating thresholds."""
        return connector_cues(image, self.roi, self.hue_band)

    def held(self, image) -> bool:
        """Instantaneous grasp decision (no debounce)."""
        warm, spec = self.cues(image)
        return warm > self.warm_thr or spec > self.spec_thr

    def update(self, image) -> bool:
        """Debounced grasp state -- majority vote over the last `debounce` frames.

        A grasped->empty transition you did not command is your 'dropped' signal.
        """
        self._buf.append(self.held(image))
        return sum(self._buf) > len(self._buf) // 2

    def reset(self):
        self._buf.clear()

    def calibrate(self, empty_imgs, held_imgs, margin: float = 0.5):
        """Set thresholds from a few labelled frames (midpoint of the two clusters)."""
        we = [connector_cues(i, self.roi, self.hue_band)[0] for i in empty_imgs]
        wh = [connector_cues(i, self.roi, self.hue_band)[0] for i in held_imgs]
        se = [connector_cues(i, self.roi, self.hue_band)[1] for i in empty_imgs]
        sh = [connector_cues(i, self.roi, self.hue_band)[1] for i in held_imgs]
        # only set a cue's threshold if that cue actually separates the clusters
        if max(wh, default=0) > max(we, default=0):
            self.warm_thr = margin * (max(we, default=0) + min(wh))
        if max(sh, default=0) > max(se, default=0):
            self.spec_thr = margin * (max(se, default=0) + min(sh))
        return self


def cable_region(pose, grasped, boxes: dict, default: str = "transit") -> str:
    """Coarse cable location from KINEMATICS, not pixels.

    When grasped, the cable is rigidly at the end effector, so its region is just
    which named box the EE pose falls in -- far more reliable than any wrist-cam
    readout. When not grasped, the cable is where it was last released; track that
    from grasp transitions.

    Args:
        pose:  object with .x/.y/.z (e.g. URPose) in robot base frame.
        grasped: output of GraspDetector.
        boxes: {name: (lo_xyz, hi_xyz)} axis-aligned regions.
        default: region when grasped but in no named box.
    """
    if not grasped:
        return "released"
    p = np.array([pose.x, pose.y, pose.z], float)
    for name, (lo, hi) in boxes.items():
        if np.all(p >= np.asarray(lo, float)) and np.all(p <= np.asarray(hi, float)):
            return name
    return default
