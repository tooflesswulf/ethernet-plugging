import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


# --- score_state schema ---------------------------------------------------
# The cable's location and grasp are scored on two orthogonal axes so that
# predicates can co-occur (e.g. held AND plugged during an A->B insertion),
# which is exactly what happens at the transitions we care about. See
# score_state().
GRIP_STATUS = "gripper is holding the cable"
STATE_SURFACE = "the ethernet cable is lying on the cardboard surface"
STATE_HOLDING = "the robot gripper is holding the ethernet cable"
STATE_VISIBLE_ELSE = "the ethernet cable is visible but not resting on the surface or held in the gripper"
STATE_NOT_VISIBLE = "the ethernet cable is not visible"

# Order: A (holding), B (plugged), C (on surface), D (not visible).
ETHERNET_STATES = (GRIP_STATUS, STATE_SURFACE, STATE_HOLDING, STATE_VISIBLE_ELSE, STATE_NOT_VISIBLE)

# IMPORTANT: the prompts below deliberately never say "ethernet cable" (or
# "RJ45"). On this wrist view the model does not recognize the plug as an
# ethernet cable and, whenever that phrase appears, collapses every readout to
# ~0 (measured P(holding) 0.01-0.15 on clearly-held frames). Asking about a
# generic "object" instead restores the signal (P(holding) 0.4-0.9). We attach
# the task label ("ethernet cable") ourselves via the STATE_* keys, since the
# cable is the only manipulated object in this task. See git history / the
# qwen-debug investigation for the ablation.
#
# Mutually-exclusive options for the cable-end LOCATION head. The order here is
# the A/B/C/D letter order shown to the model; indices are referenced by
# score_state (0=port, 1=surface, 2=mid-air, 3=not-visible).
#
# NOTE: the "inserted into a socket or port" option (-> STATE_PLUGGED) is the
# weak one -- a single wrist frame cannot reliably tell plugged from
# held-just-outside-the-port. For a trustworthy plugged signal, fuse force /
# proximity-to-port_pose here rather than relying on this logit.
_LOCATION_OPTIONS = (
    "resting on the flat surface",
    "held up in the air by the gripper",
    "object is visible but not resting on the surface or held in the gripper",
    "no such object is in view",
)


class QwenClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_input_length: int = 32768,
        min_pixels: int = 384 * 384,
        max_pixels: int = 1024 * 1024,
        sharp_window: int = 4,
    ):

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.model_name = model_name
        self.max_input_length = max_input_length

        # Resolution budget handed to the image processor for score_state. The
        # RJ45 connector is tiny in the wrist view, so we let single frames use a
        # generous token budget rather than the video-sized default.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # How many of the most recent frames score_state considers when picking
        # the sharpest one to score (blurred frames give garbage readouts).
        self.sharp_window = sharp_window

        # Answer tokens for the readouts. The prompt is prefilled with
        # "The answer is:" so each continuation carries a leading space.
        # Yes/No (grasp head) and A/B/C/D (location head) answer tokens.
        tok = self.processor.tokenizer
        self._yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
        self._no_id = tok(" No", add_special_tokens=False).input_ids[0]
        self._letter_ids = [
            tok(f" {chr(ord('A') + i)}", add_special_tokens=False).input_ids[0]
            for i in range(len(_LOCATION_OPTIONS))
        ]

    # --- single-frame state estimation ------------------------------------

    @staticmethod
    def _sharpness(img) -> float:
        """Cheap focus measure (gradient energy); higher == sharper."""
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        gx = np.diff(a, axis=1)
        gy = np.diff(a, axis=0)
        return float(gx.var() + gy.var())

    def _select_frame(self, frames, pick_sharpest: bool):
        """Pick the frame to score: the sharpest of the recent tail, or the last."""
        frames = list(frames)
        frame = frames[-1]
        if pick_sharpest and len(frames) > 1:
            frame = max(frames[-self.sharp_window:], key=self._sharpness)
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame))
        return frame

    def _image_content(self, frame, min_pixels, max_pixels) -> dict:
        return {"type": "image", "image": frame,
                "min_pixels": min_pixels, "max_pixels": max_pixels}

    def _last_token_logits(self, content) -> torch.Tensor:
        """Logits over the vocab at the position right after 'The answer is:'."""
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = f"{text}The answer is:"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            return self.model(**inputs).logits[0, -1].float()

    def _grasp_prob(self, frame, min_pixels, max_pixels) -> float:
        """P(gripper is holding the cable) from a Yes/No readout on one frame.

        Phrased as a generic "object" on purpose -- see the note by
        _LOCATION_OPTIONS about why "ethernet cable" must not appear here.
        """
        prompt = (
            "Is the robot gripper holding an object?\n"
            "Answer Yes or No."
        )
        content = [self._image_content(frame, min_pixels, max_pixels),
                   {"type": "text", "text": prompt}]
        logits = self._last_token_logits(content)
        pair = torch.stack([logits[self._yes_id], logits[self._no_id]])
        return torch.softmax(pair, dim=0)[0].item()

    def _location_probs(self, frame, min_pixels, max_pixels) -> list:
        """Normalized distribution over _LOCATION_OPTIONS for the cable end."""
        options = "\n".join(
            f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(_LOCATION_OPTIONS)
        )
        prompt = (
            "The robot gripper is working with a small cable-like object in an assembly task.\n"
            "Where is that object right now?\n"
            f"{options}\n"
            "Answer with a single letter."
        )
        content = [self._image_content(frame, min_pixels, max_pixels),
                   {"type": "text", "text": prompt}]
        logits = self._last_token_logits(content)
        opt_logits = torch.stack([logits[i] for i in self._letter_ids])
        return torch.softmax(opt_logits, dim=0).tolist()

    def score_state(
        self,
        frames,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        pick_sharpest: bool = True,
    ) -> dict:
        """Estimate the cable's state on a single (sharp) frame.

        Queries two orthogonal heads:

          * grasp (Yes/No)  -> P(holding)
          * cable-end location (A/B/C/D multiple choice) -> P(plugged / on-surface
            / mid-air / not-visible)

        Options *compete within* the location head (fixing the muddiness of three
        independent predicates) while grasp stays independent, so holding can be
        high alongside plugged or on-surface -- the co-occurrence that defines the
        A->B / C->A / A->C transitions. A single frame cannot tell C->A from A->C
        (both are grasp=yes, location=surface); recover direction from the state
        sequence if you need it.

        Args:
            frames: recent frames (PIL Images / arrays), oldest first. Only one is
                scored; a rolling buffer just lets pick_sharpest avoid blur.
            min_pixels, max_pixels: per-frame resolution budget (defaults set on
                the client).
            pick_sharpest: score the sharpest of the last `sharp_window` frames
                instead of blindly the last one.

        Returns:
            {STATE_HOLDING, STATE_PLUGGED, STATE_SURFACE, STATE_NOT_VISIBLE: prob}.
            The mid-air location mass is not returned directly; it is
            1 - (plugged + surface + not_visible) and shows up as "held but not yet
            placed/plugged".
        """
        self.model.eval()
        min_pixels = self.min_pixels if min_pixels is None else min_pixels
        max_pixels = self.max_pixels if max_pixels is None else max_pixels

        frame = self._select_frame(frames, pick_sharpest)
        p_grasp = self._grasp_prob(frame, min_pixels, max_pixels)
        loc = self._location_probs(frame, min_pixels, max_pixels)  # port/surface/mid-air/not-visible
        return {
            GRIP_STATUS: p_grasp,
            STATE_SURFACE: loc[0],
            STATE_HOLDING: loc[1],
            STATE_VISIBLE_ELSE: loc[2],
            STATE_NOT_VISIBLE: loc[3],
        }
