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
    ):

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.model_name = model_name
        self.max_input_length = max_input_length

        # Answer tokens for the True/False readout. The prompt is prefilled with
        # "The answer is:" so the continuation carries a leading space.
        tok = self.processor.tokenizer
        self._true_id = tok(" True", add_special_tokens=False).input_ids[0]
        self._false_id = tok(" False", add_special_tokens=False).input_ids[0]

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
