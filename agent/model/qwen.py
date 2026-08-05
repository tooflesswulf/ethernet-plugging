import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


# --- score_state schema ---------------------------------------------------
# Qwen is VISION-ONLY here: it answers WHERE the cable is (which region of the
# workspace) and WHETHER it is visible. Grasp/holding is NOT asked -- the wrist
# camera cannot resolve finger-closure (t0-empty and t60-held are the same
# picture to the model), so that signal comes from the gripper width/force
# outside this module. Likewise "plugged vs approaching" and "dropped vs ok"
# are not pixel-separable and are left to force/proprioception.
#
# The three states are chosen to match visually-separable regimes and, on the
# labelled debug frames, separate cleanly (each ~1.0 on the correct class):
#   near_gripper : cable up by the tool against the plain cardboard workspace
#   at_switch    : cable over at the blue network switch / port area
#   not_visible  : cable out of frame / occluded
STATE_NEAR_GRIPPER = "the cable is up near the gripper against the plain workspace"
STATE_AT_SWITCH = "the cable is over at the network switch"
STATE_NOT_VISIBLE = "the cable is not visible in the frame"

# Order matches the A/B/C option letters shown to the model.
ETHERNET_STATES = (STATE_NEAR_GRIPPER, STATE_AT_SWITCH, STATE_NOT_VISIBLE)

# IMPORTANT: never say "ethernet cable" or "RJ45" in the prompt. On this wrist
# view the model does not recognize the plug as an ethernet cable and, whenever
# that phrase appears, collapses every readout to ~0. "cable/connector" is fine
# and we attach the task label via the STATE_* keys.
#
# The prompt is also deliberately NEUTRAL: an earlier version framed it as "the
# gripper is working with the object, where is that object", which presupposed
# a grasp and collapsed every frame onto "held in the air" (0.57-0.92 even when
# the cable was out of frame). Asking "where is the cable/connector" plainly is
# what makes the three regions separate. Options are mutually exclusive; index
# order (0=near-gripper, 1=at-switch, 2=not-visible) is referenced by score_state.
_LOCATION_OPTIONS = (
    "up near the robot gripper, against a plain background",
    "over at the blue network switch with ports",
    "not visible in the frame",
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
            "Look at the whole image. Where is the cable/connector?\n"
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
        """Estimate WHERE the cable is from a single (sharp) frame.

        Vision-only: a neutral multiple-choice location head returns a normalized
        distribution over the three visually-separable regions
        (near_gripper / at_switch / not_visible). Grasp/holding is intentionally
        not scored here -- see the schema note at the top of the module -- and is
        expected to come from the gripper width/force.

        Args:
            frames: recent frames (PIL Images / arrays), oldest first. Only one is
                scored; a rolling buffer just lets pick_sharpest avoid blur.
            min_pixels, max_pixels: per-frame resolution budget (defaults set on
                the client).
            pick_sharpest: score the sharpest of the last `sharp_window` frames
                instead of blindly the last one.

        Returns:
            {STATE_NEAR_GRIPPER, STATE_AT_SWITCH, STATE_NOT_VISIBLE: prob},
            summing to 1.
        """
        self.model.eval()
        min_pixels = self.min_pixels if min_pixels is None else min_pixels
        max_pixels = self.max_pixels if max_pixels is None else max_pixels

        frame = self._select_frame(frames, pick_sharpest)
        loc = self._location_probs(frame, min_pixels, max_pixels)  # near-gripper / at-switch / not-visible
        return {
            STATE_NEAR_GRIPPER: loc[0],
            STATE_AT_SWITCH: loc[1],
            STATE_NOT_VISIBLE: loc[2],
        }
