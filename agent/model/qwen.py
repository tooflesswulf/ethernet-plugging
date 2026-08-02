from collections.abc import Sequence
from typing import cast
import time
import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


# Default event predicates for the ethernet plugging task. Each is phrased as a
# statement that is either true or false of the most recent frame.
ETHERNET_EVENTS = (
    "the robot gripper is holding the ethernet cable",
    "the ethernet cable is fully plugged into the ethernet switch",
    "the ethernet cable is lying on the carboard surface",
)

# --- score_state schema ---------------------------------------------------
# The cable's location and grasp are scored on two orthogonal axes so that
# predicates can co-occur (e.g. held AND plugged during an A->B insertion),
# which is exactly what happens at the transitions we care about. See
# score_state() and StateFilter.
STATE_HOLDING = "the robot gripper is holding the ethernet cable"
STATE_PLUGGED = "the ethernet cable is fully plugged into the ethernet switch"
STATE_SURFACE = "the ethernet cable is lying on the cardboard surface"
STATE_NOT_VISIBLE = "the ethernet cable is not visible"

# Order: A (holding), B (plugged), C (on surface), D (not visible).
ETHERNET_STATES = (STATE_HOLDING, STATE_PLUGGED, STATE_SURFACE, STATE_NOT_VISIBLE)

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
# StateFilter (0=port, 1=surface, 2=mid-air, 3=not-visible).
#
# NOTE: the "inserted into a socket or port" option (-> STATE_PLUGGED) is the
# weak one -- a single wrist frame cannot reliably tell plugged from
# held-just-outside-the-port. For a trustworthy plugged signal, fuse force /
# proximity-to-port_pose here rather than relying on this logit.
_LOCATION_OPTIONS = (
    "inserted into a socket or port",
    "resting on the flat surface",
    "held up in the air by the gripper",
    "no such object is in view",
)

def _subsample(frames: Sequence, count: int) -> list:
    """Uniformly pick at most `count` frames, always keeping the last one."""
    if count >= len(frames):
        return list(frames)
    ix = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[i] for i in ix]


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

        # Answer tokens for the True/False readout. The prompt is prefilled with
        # "The answer is:" so the continuation carries a leading space.
        tok = self.processor.tokenizer
        self._true_id = tok(" True", add_special_tokens=False).input_ids[0]
        self._false_id = tok(" False", add_special_tokens=False).input_ids[0]
        # Yes/No (grasp head) and A/B/C/D (location head) answer tokens, same
        # leading-space convention as above.
        self._yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
        self._no_id = tok(" No", add_special_tokens=False).input_ids[0]
        self._letter_ids = [
            tok(f" {chr(ord('A') + i)}", add_special_tokens=False).input_ids[0]
            for i in range(len(_LOCATION_OPTIONS))
        ]

    def _true_prob(self, frames, event: str) -> float:
        """P(True) for `event` given the frames, normalized against P(False)."""
        prompt = (
            "The video above shows a robot arm performing an ethernet cable plugging task.\n"
            f"Statement: {event}\n"
            "Decide whether the statement is true at the moment of the LAST frame of the video. "
            "Answer True or False."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": list(frames)},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = f"{text}The answer is:"
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[misc]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()

        # Normalize over just the two answer tokens so the score is a probability
        # that is comparable across differently-worded events.
        pair = torch.stack([logits[self._true_id], logits[self._false_id]])
        return torch.softmax(pair, dim=0)[0].item()

    def score_window(self, frames, events: Sequence[str] = ETHERNET_EVENTS) -> dict:
        """P(True) for each event given a single window of frames.

        The real-time analog of score_events: instead of striding along a whole
        trajectory, it scores one window (the last frame is treated as "now") and
        returns {event: P(True)}. Intended for live use where a caller keeps a
        rolling buffer of recent frames and wants the current event probabilities.

        Args:
            frames: recent frames (>= 2) as PIL Images / arrays, oldest first.
            events: statements to score. Defaults to ETHERNET_EVENTS.
        """
        self.model.eval()
        return {event: self._true_prob(frames, event) for event in events}

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
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[misc]
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
            "The robot gripper is working with a small object in an assembly task.\n"
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

        Unlike score_window's independent True/False predicates, this queries two
        orthogonal heads:

          * grasp (Yes/No)  -> P(holding)
          * cable-end location (A/B/C/D multiple choice) -> P(plugged / on-surface
            / mid-air / not-visible)

        Options *compete within* the location head (fixing the muddiness of three
        independent predicates) while grasp stays independent, so holding can be
        high alongside plugged or on-surface -- the co-occurrence that defines the
        A->B / C->A / A->C transitions. A single frame cannot tell C->A from A->C
        (both are grasp=yes, location=surface); use StateFilter to recover
        direction from the state sequence.

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
            STATE_HOLDING: p_grasp,
            STATE_PLUGGED: loc[0],
            STATE_SURFACE: loc[1],
            STATE_NOT_VISIBLE: loc[3],
        }

    def score_events(
        self,
        frames: Sequence,
        events: Sequence[str] = ETHERNET_EVENTS,
        stride: int = 10,
        max_video_frames: int = 16,
        min_frames: int = 2,
        verbose: bool = False,
    ):
        """Score each event predicate at strided timesteps along a trajectory.

        At timestep t the model sees the trajectory so far (frames [0, t], uniformly
        subsampled to `max_video_frames`) and answers True/False for each event.

        Cost is one forward pass per (timestep, event), so it scales as
        len(frames)/stride * len(events) -- raise `stride` to trade resolution for speed.

        Args:
            frames: full list of PIL Images / arrays for the episode.
            events: statements to score. Defaults to ETHERNET_EVENTS.
            stride: evaluate every `stride`-th frame. The final frame is always scored.
            max_video_frames: frames handed to the model per query.
            min_frames: smallest window the model is queried on (video needs >= 2).
            verbose: print per-timestep scores and timing as they are computed.

        Returns:
            (scores, timesteps) where `timesteps` is an int array of evaluated frame
            indices and `scores` maps each event string to a float array of P(True)
            aligned with `timesteps`.
        """
        timesteps = list(range(min_frames - 1, len(frames), stride))
        if not timesteps:
            raise ValueError(f"Need at least {min_frames} frames, got {len(frames)}.")
        if timesteps[-1] != len(frames) - 1:
            timesteps.append(len(frames) - 1)

        self.model.eval()
        scores = {event: [] for event in events}
        for t in timesteps:
            window = _subsample(frames[:t + 1], max_video_frames)

            t0 = time.time()
            for event in events:
                scores[event].append(self._true_prob(window, event))
            if verbose:
                summary = "  ".join(f"{scores[e][-1]:.2f}" for e in events)
                print(f"frame {t:>5d}  {summary}  ({time.time() - t0:.1f}s)")

        return {e: np.array(v) for e, v in scores.items()}, np.array(timesteps)

    def detect_events(self, frames, events=ETHERNET_EVENTS, threshold: float = 0.5, **kwargs):
        """First timestep at which each event's P(True) crosses `threshold`.

        Thin wrapper over score_events. Accepts the same keyword arguments.

        Returns:
            (onsets, scores, timesteps) where `onsets` maps each event to the frame
            index of its first crossing, or None if it never crosses.
        """
        scores, timesteps = self.score_events(frames, events=events, **kwargs)
        onsets = {}
        for event, probs in scores.items():
            hit = np.flatnonzero(probs >= threshold)
            onsets[event] = int(timesteps[hit[0]]) if len(hit) else None
        return onsets, scores, timesteps

    def compute_instruction_reward(
        self,
        instruction,
        frames,
        scale=0.5,
        reduction: str = "mean",
        add_chat_template: bool = False,
    ):
        """Compute a log-likelihood reward for an instruction conditioned on a trajectory of frames.

        This implements the instruction reward approach from "Vision Language Models are
        In-Context Value Learners", measuring how well the trajectory matches the given
        instruction by computing the log-probability of generating the instruction text.

        Note: this returns an unnormalized log-prob, which is not comparable across
        different instruction strings. For thresholding events, prefer score_events.

        Args:
            frames: List of images representing the trajectory (at least 2 frames).
            instruction: Instruction text to evaluate.
            reduction: Reduction to apply to token log probabilities ("mean" or "sum").
            use_video_description: If True, generate instruction-agnostic description of
                                  the robot manipulation trajectory, then prepend it as context
                                  before evaluating instruction likelihood. This avoids circular
                                  dependencies that would artificially inflate scores.
            add_chat_template: If True, wrap the full prompt (including instruction) with
                               the chat template before tokenization.

        Returns:
            the computed reward
        """

        N = int(len(frames) * scale)
        v_indice = np.linspace(0, len(frames) - 1, min(N, len(frames)), dtype=int)
        frames = [frames[i] for i in v_indice]
        pil_frames = frames

        # Optionally generate trajectory description for augmented context
        prompt_text = "The above video shows a robot manipulation trajectory that completes the following task: "

        content = [
            {"type": "video", "video": pil_frames},
            {"type": "text", "text": prompt_text},
        ]
        user_messages = [{"role": "user", "content": content}]
        eos_token = self.processor.tokenizer.eos_token

        instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is: True"
        prompt_chat = self.processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if eos_token is not None:
            prompt_chat = prompt_chat.split(eos_token)[0]
        full_text = f"{prompt_chat}{instruction_suffix}"
        image_inputs, video_inputs = process_vision_info(user_messages)  # type: ignore[misc]
        self.model.eval()

        rewards = []
        times = []
        for n in range(N - 1, N):
            _video_inputs = video_inputs[0][:n]
            inputs = self.processor(
                text=[full_text],
                images=image_inputs,
                videos=[_video_inputs],
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            labels = inputs["input_ids"].clone()

            # Mask the prompt so we only compute loss on the instruction + "True" part
            prompt_length = inputs["input_ids"].shape[1] - 1
            labels[:, :prompt_length] = -100
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

            t1 = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
            t2 = time.time()
            # Compute per-token log probabilities
            logits = outputs.logits[:, :-1, :]
            target_labels = labels[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            mask = target_labels != -100
            safe_targets = target_labels.masked_fill(~mask, 0)
            token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            masked_log_probs = token_log_probs[mask]

            # Apply reduction
            reward = masked_log_probs.sum().item() if reduction == "sum" else masked_log_probs.mean().item()
            rewards.append(reward)
            times.append(t2 - t1)

        return float(rewards[-1]), frames


class StateFilter:
    """Sticky HMM that smooths score_state() output over time and recovers the
    transition *direction* a single frame can't resolve.

    Hidden phases and the per-frame observation each expects:

        phase     grasp  cable-end     meaning
        surface    no    surface       C: resting on cardboard
        grab       yes   surface       C->A: being picked up
        held       yes   mid-air       A: held, in transit
        insert     yes   port          A->B: being plugged in
        plugged    no    port          B: plugged, released
        place      yes   surface       A->C: being set down
        hidden      -     not-visible   D: occluded / out of frame

    `grab` and `place` share the same instantaneous (grasp, location); only the
    allowed transitions -- grab is reachable from `surface`, place from `held` --
    tell them apart. That is precisely the C->A vs A->C ambiguity that motivates
    the filter. The collapsed A/B/C/D probabilities returned by update() can
    co-occur (e.g. `insert` lights up both holding and plugged).
    """

    PHASES = ("surface", "grab", "held", "insert", "plugged", "place", "hidden")
    # (expected grasp: 1 / 0 / None, cable-end location index) per phase.
    # location index matches _LOCATION_OPTIONS: 0=port, 1=surface, 2=mid-air, 3=not-visible.
    _EMIT = ((0, 1), (1, 1), (1, 2), (1, 0), (0, 0), (1, 1), (None, 3))
    # Directed neighbours (besides the self-loop) reachable from each phase.
    _EDGES = {
        "surface": ("grab",),
        "grab":    ("held", "surface"),   # grasp succeeds -> held, or slips back
        "held":    ("insert", "place"),
        "insert":  ("plugged", "held"),   # seats -> plugged, or pulls back out
        "plugged": ("insert",),           # start of an unplug
        "place":   ("surface",),
        "hidden":  (),                    # recovers to anywhere (handled below)
    }

    def __init__(self, p_self: float = 0.80, p_hidden: float = 0.02, floor: float = 1e-3):
        n = len(self.PHASES)
        idx = {p: i for i, p in enumerate(self.PHASES)}
        T = np.full((n, n), floor)
        for p, i in idx.items():
            T[i, i] += p_self
            nbrs = self._EDGES[p]
            if nbrs:
                share = max(0.0, 1.0 - p_self - p_hidden) / len(nbrs)
                for q in nbrs:
                    T[i, idx[q]] += share
            T[i, idx["hidden"]] += p_hidden  # occlusion can strike from any phase
        T[idx["hidden"], :] += 1.0 / n      # ... and clears back to anywhere
        T /= T.sum(axis=1, keepdims=True)

        self.T = T
        self._idx = idx
        self.belief = np.full(n, 1.0 / n)

    def reset(self):
        self.belief = np.full(len(self.PHASES), 1.0 / len(self.PHASES))

    def update(self, scored: dict) -> dict:
        """Advance one step given a score_state() dict; return smoothed A/B/C/D."""
        p_grasp = scored[STATE_HOLDING]
        loc = np.array([
            scored[STATE_PLUGGED],        # 0 port
            scored[STATE_SURFACE],        # 1 surface
            0.0,                          # 2 mid-air (filled below)
            scored[STATE_NOT_VISIBLE],    # 3 not-visible
        ], dtype=float)
        loc[2] = max(0.0, 1.0 - loc[0] - loc[1] - loc[3])

        emit = np.empty(len(self.PHASES))
        for i, (g, l) in enumerate(self._EMIT):
            e = loc[l]
            if g is not None:
                e *= p_grasp if g == 1 else (1.0 - p_grasp)
            emit[i] = e + 1e-6

        prior = self.belief @ self.T          # predict
        post = prior * emit                    # correct
        s = post.sum()
        self.belief = post / s if s > 0 else np.full(len(self.PHASES), 1.0 / len(self.PHASES))
        return self.states()

    def states(self) -> dict:
        """Collapse the phase belief into co-occurring A/B/C/D probabilities."""
        b, ix = self.belief, self._idx
        return {
            STATE_HOLDING:     float(b[ix["grab"]] + b[ix["held"]] + b[ix["insert"]] + b[ix["place"]]),
            STATE_PLUGGED:     float(b[ix["insert"]] + b[ix["plugged"]]),
            STATE_SURFACE:     float(b[ix["surface"]] + b[ix["grab"]] + b[ix["place"]]),
            STATE_NOT_VISIBLE: float(b[ix["hidden"]]),
        }

    @property
    def phase(self) -> str:
        """MAP phase label, e.g. 'insert' (A->B), 'grab' (C->A), 'place' (A->C)."""
        return self.PHASES[int(self.belief.argmax())]
