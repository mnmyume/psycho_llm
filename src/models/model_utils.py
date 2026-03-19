"""
Model loading and inference utilities with dual backend support.

Supports two backends:
  - "unsloth": Uses FastVisionModel for optimized loading + 4-bit quantization.
               Best for Qwen3-VL models (dense architectures).
  - "hf":      Uses standard HuggingFace transformers + PEFT.
               Required for Qwen3.5 MoE models (BnB can't quantize MoE experts).

The backend is selected via the `backend` field in the training recipe YAML.
"""

import json
import os
import torch
from PIL import Image as PILImage
from typing import Optional, Tuple
import transformers

from transformers import TextStreamer


# ============================================================
# Backend: Unsloth (for Qwen3-VL dense models)
# ============================================================

def _load_unsloth(model_name: str, load_in_4bit: bool = True) -> Tuple:
    """Load model via Unsloth's FastVisionModel."""
    from unsloth import FastVisionModel

    print(f"Loading base model [unsloth]: {model_name} (4-bit={load_in_4bit})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _apply_lora_unsloth(
    model,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
):
    """Apply LoRA via Unsloth's FastVisionModel."""
    from unsloth import FastVisionModel

    print(f"Applying LoRA adapters [unsloth] (r={r}, alpha={alpha}, dropout={dropout})")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    FastVisionModel.for_training(model)
    return model


def _load_unsloth_inference(model_path: str, load_in_4bit: bool = True) -> Tuple:
    """Load model for inference via Unsloth."""
    from unsloth import FastVisionModel

    print(f"Loading model for inference [unsloth]: {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded and set to inference mode.")
    return model, tokenizer


def _generate_unsloth(model, tokenizer, image, prompt, **gen_kwargs) -> str:
    """Generate response using Unsloth tokenizer API."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = tokenizer(
        images=[image],
        text=[input_text],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    stream = gen_kwargs.pop("stream", False)
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            streamer=streamer,
            **gen_kwargs,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


# ============================================================
# Backend: HuggingFace (for Qwen3.5 MoE models)
# ============================================================

def _load_hf(model_name: str, load_in_4bit: bool = False) -> Tuple:
    """Load model via standard HuggingFace transformers in bf16.

    Note: 4-bit quantization via BitsAndBytes is NOT effective for MoE
    models (expert layers use custom classes that BnB can't quantize).
    We always load in bf16 regardless of load_in_4bit.
    """
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoHFVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoHFVisionModel

    print(f"Loading base model [hf]: {model_name} (bf16)")
    try:
        model = AutoHFVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        if isinstance(exc, KeyError) and "qwen3_5" in str(exc):
            raise RuntimeError(
                "Current transformers build does not support Qwen3.5 model_type 'qwen3_5'. "
                f"Installed transformers={transformers.__version__}. "
                "Upgrade transformers in the training environment or use an Unsloth Qwen3-VL recipe."
            ) from exc
        from transformers import AutoModelForCausalLM
        print(f"Vision model auto-loader failed ({type(exc).__name__}); falling back to AutoModelForCausalLM.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, processor


def _apply_lora_hf(
    model,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
):
    """Apply LoRA via standard PEFT library."""
    from peft import LoraConfig, get_peft_model

    if not finetune_vision_layers and not finetune_language_layers:
        raise ValueError(
            "At least one of finetune_vision_layers or finetune_language_layers must be True."
        )

    module_suffixes = []
    if finetune_attention_modules:
        module_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if finetune_mlp_modules:
        module_suffixes.extend(["gate_proj", "up_proj", "down_proj"])

    if not module_suffixes:
        raise ValueError(
            "At least one of finetune_attention_modules or finetune_mlp_modules must be True."
        )

    def _is_vision_module(name: str) -> bool:
        return any(tok in name for tok in ("visual", "vision_tower", "vision_model"))

    def _is_language_module(name: str) -> bool:
        return any(tok in name for tok in ("model.layers", "language_model", "lm_head"))

    target_modules = []
    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(f".{suffix}") for suffix in module_suffixes):
            continue

        is_vision = _is_vision_module(module_name)
        is_language = _is_language_module(module_name)

        if finetune_vision_layers and finetune_language_layers:
            target_modules.append(module_name)
        elif finetune_vision_layers and is_vision:
            target_modules.append(module_name)
        elif finetune_language_layers and is_language:
            target_modules.append(module_name)

    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError(
            "No LoRA target modules matched. Check finetune_* flags and model architecture names."
        )

    print(f"Applying LoRA adapters [hf] (r={r}, alpha={alpha}, dropout={dropout})")
    print(f"  Matched {len(target_modules)} target modules.")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    return model


def _load_hf_inference(model_path: str, load_in_4bit: bool = False) -> Tuple:
    """Load model for inference via HuggingFace + PEFT."""
    from peft import PeftModel
    from transformers import AutoProcessor

    print(f"Loading model for inference [hf]: {model_path}")

    is_lora = os.path.isfile(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", model_path)

        print(f"  Detected LoRA adapter. Base model: {base_model_name}")
        model, _ = _load_hf(base_model_name)
        model = PeftModel.from_pretrained(model, model_path)
        # Prefer processor/tokenizer from adapter dir so chat template changes
        # (including thinking behavior) are preserved after fine-tuning.
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print("  Loaded processor from LoRA adapter directory.")
        except Exception:
            processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
            print("  Adapter processor not found; using base model processor.")
        print("  LoRA adapter loaded and applied.")
    else:
        model, processor = _load_hf(model_path)

    model.eval()
    print("Model loaded and set to inference mode.")
    return model, processor


def _generate_hf(model, processor, image, prompt, **gen_kwargs) -> str:
    """Generate response using HuggingFace processor API."""
    import re

    # Extract thinking controls.
    # `thinking_budget` should also be forwarded to model.generate() when supported.
    thinking_budget = gen_kwargs.pop("thinking_budget", None)
    enable_thinking = thinking_budget is not None and thinking_budget > 0
    show_thinking = bool(gen_kwargs.pop("show_thinking", False))

    def _strip_markdown_fences(text: str) -> str:
        text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        return text.strip()

    messages = []
    if enable_thinking:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use a short reasoning trace when thinking is enabled: "
                    "at most 2 short sentences or about 40 words. "
                    "Then give the final answer directly. "
                    "Do not use markdown code fences."
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    )

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    # Some processor/chat-template combos ignore `enable_thinking`.
    # Ensure a thinking prefill exists when explicitly requested.
    if enable_thinking and "<think>" not in input_text:
        input_text = input_text + "<think>\n"

    inputs = processor(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Ensure float tensors (like pixel_values) match the model's dtype (e.g., bfloat16)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)

    stream = gen_kwargs.pop("stream", False)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True) if stream else None

    # Fix: generation_config may store eos_token_id as a set,
    # which causes "unhashable type: 'set'" during generate().
    if hasattr(model, "generation_config"):
        gc = model.generation_config
        if isinstance(getattr(gc, "eos_token_id", None), set):
            gc.eos_token_id = list(gc.eos_token_id)
        if isinstance(getattr(gc, "pad_token_id", None), set):
            gc.pad_token_id = list(gc.pad_token_id)

    model_gen_kwargs = dict(gen_kwargs)
    if enable_thinking:
        model_gen_kwargs["thinking_budget"] = int(thinking_budget)

    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs,
                streamer=streamer,
                **model_gen_kwargs,
            )
        except (TypeError, ValueError) as exc:
            # Some model implementations reject unsupported kwargs as either
            # TypeError (unexpected kwarg) or ValueError (unused model_kwargs).
            err = str(exc)
            unsupported_thinking_budget = (
                "thinking_budget" in err
                and (
                    "unexpected keyword argument" in err
                    or "not used by the model" in err
                )
            )
            if unsupported_thinking_budget:
                model_gen_kwargs.pop("thinking_budget", None)
                output_ids = model.generate(
                    **inputs,
                    streamer=streamer,
                    **model_gen_kwargs,
                )
            else:
                raise

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=not show_thinking,
    )[0]
    response = re.sub(r"<\|[^|]+?\|>", "", response).strip()
    response = _strip_markdown_fences(response)

    if show_thinking:
        # The chat template prefills "<think>\n" in the prompt, so the decoded
        # continuation may omit the opening tag. Reconstruct it for display.
        if enable_thinking and "<think>" not in response and not response.lstrip().startswith("{"):
            if "</think>" in response:
                reasoning, answer = response.split("</think>", 1)
                reasoning = reasoning.strip()
                answer = answer.lstrip()
                if reasoning:
                    response = f"<think>\n{reasoning}\n</think>\n{answer}"
                else:
                    response = answer
            else:
                json_start = response.find("{")
                if json_start > 0:
                    reasoning = response[:json_start].strip()
                    answer = response[json_start:].lstrip()
                    if reasoning:
                        response = f"<think>\n{reasoning}\n</think>\n{answer}"
                    else:
                        response = answer
        return response.strip()

    # Hide reasoning by default. When the opening <think> tag was part of the
    # prompt prefill, generated text may begin with reasoning and only emit the
    # closing tag, so strip that implied reasoning span as well.
    if enable_thinking and "</think>" in response and "<think>" not in response:
        response = response.split("</think>", 1)[1]
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    response = response.replace("<think>", "").replace("</think>", "")
    return _strip_markdown_fences(response)


# ============================================================
# Public API — dispatches to the selected backend
# ============================================================

def load_base_model(
    model_name: str,
    backend: str = "hf",
    load_in_4bit: bool = True,
) -> Tuple:
    """Load a base model and its tokenizer/processor.

    Args:
        model_name: HuggingFace model ID or local path.
        backend: "unsloth" or "hf".
        load_in_4bit: Use 4-bit quantization (only effective with unsloth backend).

    Returns:
        Tuple of (model, tokenizer_or_processor).
    """
    if backend == "unsloth":
        return _load_unsloth(model_name, load_in_4bit)
    else:
        return _load_hf(model_name, load_in_4bit)


def apply_lora_for_training(
    model,
    backend: str = "hf",
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
):
    """Apply LoRA adapters for training.

    Args:
        model: The base model from load_base_model().
        backend: "unsloth" or "hf".
        r, alpha, dropout: LoRA hyperparameters.
        finetune_*: Which layers to apply LoRA to.

    Returns:
        Model with LoRA adapters, set to training mode.
    """
    kwargs = dict(
        r=r, alpha=alpha, dropout=dropout,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
    )
    if backend == "unsloth":
        return _apply_lora_unsloth(model, **kwargs)
    else:
        return _apply_lora_hf(model, **kwargs)


def load_model_for_inference(
    model_path: str,
    backend: str = "hf",
    load_in_4bit: bool = True,
) -> Tuple:
    """Load a model for inference (auto-detects LoRA adapters).

    Args:
        model_path: Base model ID or path to a LoRA adapter directory.
        backend: "unsloth" or "hf".
        load_in_4bit: Use 4-bit quantization (only effective with unsloth backend).

    Returns:
        Tuple of (model, tokenizer_or_processor).
    """
    if backend == "unsloth":
        return _load_unsloth_inference(model_path, load_in_4bit)
    else:
        return _load_hf_inference(model_path, load_in_4bit)


def generate_response(
    model,
    tokenizer,
    image: PILImage.Image,
    prompt: str,
    backend: str = "hf",
    max_new_tokens: int = 4096,
    temperature: float = 1.5,
    min_p: float = 0.1,
    stream: bool = False,
    thinking_budget: Optional[int] = 64,
    show_thinking: bool = False,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate a response for an image and text prompt.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer (unsloth) or processor (hf).
        image: A PIL Image object.
        prompt: Text instruction for the model.
        backend: "unsloth" or "hf".
        max_new_tokens, temperature, min_p: Generation parameters.
        stream: If True, streams output to stdout.
        thinking_budget: Max tokens for Qwen3.5 thinking (hf backend only).
            Set to 0 or None to disable thinking entirely. Default: 64.
        show_thinking: If True, keep <think>...</think> text in output (hf only).
        repetition_penalty: Penalizes repeated tokens (>1.0 = less repetition).
            Default: 1.2.

    Returns:
        The generated text response.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
        stream=stream,
        repetition_penalty=repetition_penalty,
    )
    if backend == "unsloth":
        return _generate_unsloth(model, tokenizer, image, prompt, **gen_kwargs)
    else:
        gen_kwargs["thinking_budget"] = thinking_budget
        gen_kwargs["show_thinking"] = show_thinking
        return _generate_hf(model, tokenizer, image, prompt, **gen_kwargs)
