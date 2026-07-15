from collections.abc import Sequence
from typing import cast
import time
import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


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
