import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


# --- score_state schema ---------------------------------------------------
# Qwen is VISION-ONLY here: it answers WHERE the cable is (which region of the
# workspace) and WHETHER it is visible. Grasp/holding is NOT asked -- the wrist
# camera cannot resolve finger-closure, so that signal comes from the gripper width/force
# outside this module. Likewise "plugged vs approaching" and "dropped vs ok"
# are not pixel-separable and are left to force/proprioception.
#
# The three states come from two robust, independent heads (see score_state):
#   * visibility (Yes/No): is the cable/connector visible at all?
#   * scene (plain vs cluttered): what is behind the gripper?
# factored as:
#   not_visible  = 1 - P(visible)                 cable out of frame / occluded
#   near_gripper = P(visible) * P(plain scene)    up by the tool, plain workspace
#   at_switch    = P(visible) * P(cluttered scene) at the port / work area
STATE_NEAR_GRIPPER = "the cable is up near the gripper against the plain workspace"
STATE_AT_SWITCH = "the cable is over at the network switch"
STATE_NOT_VISIBLE = "the cable is not visible in the frame"

ETHERNET_STATES = (STATE_NEAR_GRIPPER, STATE_AT_SWITCH, STATE_NOT_VISIBLE)

# IMPORTANT: never say "ethernet cable" or "RJ45" in the prompt. On this wrist
# view the model does not recognize the plug as an ethernet cable and, whenever
# that phrase appears, collapses every readout to ~0. "cable/connector" is fine
# and we attach the task label via the STATE_* keys.
#
# WHY SCENE-CLUTTER, NOT THE SWITCH: the region head used to key off "the blue
# network switch". When the switch was swapped for a small BLACK one, that
# anchor died -- the switch is now tiny, dark, and edge-of-frame, and the word
# "blue" actively pointed the model at the blue bench clamps instead (at_switch
# dropped to 0.18-0.38, port-jack detection to <0.35). What still separates
# transport from the port region robustly is the BACKGROUND: plain cardboard vs
# a cluttered bench. That scene question hits ~0.99 on both classes regardless
# of the switch's colour/size, so the region is inferred from clutter, not the
# switch. Keep these options switch-agnostic.
_SCENE_OPTIONS = (
    "a plain, mostly empty cardboard or desk surface",
    "a cluttered bench of tools, clamps, wires and electronic equipment",
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
        # Yes/No (visibility head) and A/B/... (multiple-choice) answer tokens.
        tok = self.processor.tokenizer
        self._yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
        self._no_id = tok(" No", add_special_tokens=False).input_ids[0]
        self._letter_ids = [
            tok(f" {chr(ord('A') + i)}", add_special_tokens=False).input_ids[0]
            for i in range(4)
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

    def _visible_prob(self, frame, min_pixels, max_pixels) -> float:
        """P(the cable/connector is visible somewhere in the frame)."""
        prompt = (
            "Is a cable or its connector visible anywhere in this image?\n"
            "Answer Yes or No."
        )
        content = [self._image_content(frame, min_pixels, max_pixels),
                   {"type": "text", "text": prompt}]
        logits = self._last_token_logits(content)
        pair = torch.stack([logits[self._yes_id], logits[self._no_id]])
        return torch.softmax(pair, dim=0)[0].item()

    def _scene_cluttered_prob(self, frame, min_pixels, max_pixels) -> float:
        """P(scene behind the gripper is the cluttered port/work area vs plain).

        Region proxy that is robust to the switch's colour/size -- see the WHY
        SCENE-CLUTTER note by _SCENE_OPTIONS.
        """
        options = "\n".join(
            f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(_SCENE_OPTIONS)
        )
        prompt = (
            "What is in the scene behind the robot gripper?\n"
            f"{options}\n"
            "Answer with a single letter."
        )
        content = [self._image_content(frame, min_pixels, max_pixels),
                   {"type": "text", "text": prompt}]
        logits = self._last_token_logits(content)
        pair = torch.stack([logits[self._letter_ids[0]], logits[self._letter_ids[1]]])
        return torch.softmax(pair, dim=0)[1].item()  # index 1 == cluttered

    def score_state(
        self,
        frames,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        pick_sharpest: bool = True,
    ) -> dict:
        """Estimate WHERE the cable is from a single (sharp) frame.

        Vision-only, from two robust independent heads (see the schema note at
        the top of the module) factored into three regions:

            not_visible  = 1 - P(visible)
            near_gripper = P(visible) * P(plain scene)
            at_switch    = P(visible) * P(cluttered scene)

        Grasp/holding is intentionally not scored here -- it comes from the
        gripper width/force outside this module.

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
        p_visible = self._visible_prob(frame, min_pixels, max_pixels)
        p_cluttered = self._scene_cluttered_prob(frame, min_pixels, max_pixels)
        return {
            STATE_NEAR_GRIPPER: p_visible * (1.0 - p_cluttered),
            STATE_AT_SWITCH: p_visible * p_cluttered,
            STATE_NOT_VISIBLE: 1.0 - p_visible,
        }
