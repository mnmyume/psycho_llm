"""
Model loading and inference utilities with dual backend support.

Supports two backends:
  - "unsloth": Uses FastVisionModel for optimized loading + 4-bit quantization.
               Best for Qwen3-VL models (dense architectures).
  - "hf":      Uses standard HuggingFace transformers + PEFT.
               Supports bf16 loading by default and can optionally use
               bitsandbytes 4-bit QLoRA when the model architecture supports it.

The backend is selected via the `backend` field in the training recipe YAML.
"""

import json
import os
import torch
from PIL import Image as PILImage
from typing import Optional, Tuple
import transformers

from transformers import TextStreamer
from chat_template_utils import apply_chat_template_with_fallback


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


def _generate_unsloth(model, tokenizer, images, prompt, **gen_kwargs) -> str:
    """Generate response using Unsloth tokenizer API.

    Args:
        images: List of PIL Image objects.
    """
    user_content = [{"type": "text", "text": prompt}]
    for img in images:
        user_content.append({"type": "image", "image": img})

    messages = [{"role": "user", "content": user_content}]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = tokenizer(
        images=images,
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

def _hf_validate_4bit_target(model_name: str):
    """Reject known HF model families that do not fit this repo's 4-bit path."""
    model_name_l = model_name.lower()
    if "fp8" in model_name_l:
        raise RuntimeError(
            f"HF 4-bit QLoRA cannot be combined with an FP8 checkpoint: {model_name}. "
            "Use the non-FP8 base model instead."
        )
    if "a3b" in model_name_l:
        raise RuntimeError(
            f"HF 4-bit QLoRA is not supported in this repo for MoE A3B models: {model_name}. "
            "Use bf16 on larger hardware or an Unsloth Qwen3-VL recipe instead."
        )


def _hf_quantization_config(model_name: str, load_in_4bit: bool):
    """Build a bitsandbytes quantization config for HF QLoRA loads."""
    if not load_in_4bit:
        return None

    _hf_validate_4bit_target(model_name)
    if not torch.cuda.is_available():
        raise RuntimeError("HF 4-bit loading requires a CUDA GPU.")

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _hf_training_load_kwargs(model_name: str, load_in_4bit: bool) -> dict:
    """Build from_pretrained kwargs for HF training loads."""
    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    quantization_config = _hf_quantization_config(model_name, load_in_4bit)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        # For training, keep the model on the current device rather than
        # relying on auto-sharding/offload behavior intended for inference.
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        kwargs["device_map"] = "auto"
    return kwargs


def _hf_inference_load_kwargs(model_name: str, load_in_4bit: bool) -> dict:
    """Build from_pretrained kwargs for HF inference loads."""
    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    quantization_config = _hf_quantization_config(model_name, load_in_4bit)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    kwargs["device_map"] = "auto"
    return kwargs


def _load_hf(
    model_name: str,
    load_in_4bit: bool = False,
) -> Tuple:
    """Load model via HuggingFace transformers in bf16 or 4-bit QLoRA mode."""
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoHFVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoHFVisionModel

    mode_label = "4-bit NF4 QLoRA" if load_in_4bit else "bf16"
    print(f"Loading base model [hf]: {model_name} ({mode_label})")
    load_kwargs = _hf_training_load_kwargs(model_name, load_in_4bit)
    try:
        model = AutoHFVisionModel.from_pretrained(
            model_name,
            **load_kwargs,
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
            **load_kwargs,
        )

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except ImportError as exc:
        err = str(exc)
        if "ReasoningEffort" in err and "mistral_common" in err:
            raise RuntimeError(
                "Gemma 4 processor loading requires a newer mistral_common package. "
                "Please upgrade the training environment to mistral_common>=1.10.0 "
                "and retry."
            ) from exc
        raise
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not finetune_vision_layers and not finetune_language_layers:
        raise ValueError(
            "At least one of finetune_vision_layers or finetune_language_layers must be True."
        )

    module_suffixes = []
    if finetune_attention_modules:
        module_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if finetune_mlp_modules:
        module_suffixes.extend(["gate_proj", "up_proj", "down_proj"])
    if finetune_vision_layers:
        # Gemma 4 exposes a separate multimodal bridge as `embed_vision`.
        # Adapting that projection alongside the vision tower is useful for
        # image-heavy tasks and is not covered by the standard attn/MLP suffixes.
        module_suffixes.extend(["embedding_projection"])

    if not module_suffixes:
        raise ValueError(
            "At least one of finetune_attention_modules or finetune_mlp_modules must be True."
        )

    def _is_vision_module(name: str) -> bool:
        return any(
            tok in name
            for tok in (
                "visual",
                "vision_tower",
                "vision_model",
                "embed_vision",
                "multimodal_projector",
                "multi_modal_projector",
                "image_projection",
                "vision_proj",
            )
        )

    def _is_language_module(name: str) -> bool:
        return any(tok in name for tok in ("model.layers", "language_model", "lm_head"))

    def _resolve_supported_lora_target(module_name: str, module) -> Tuple[str, bool]:
        """Redirect wrapper modules to a PEFT-supported leaf module when possible."""
        if isinstance(module, torch.nn.Linear):
            return module_name, False

        inner_linear = getattr(module, "linear", None)
        if isinstance(inner_linear, torch.nn.Linear):
            return f"{module_name}.linear", True

        return module_name, False

    target_modules = []
    redirected_modules = []
    for module_name, module in model.named_modules():
        if not any(module_name.endswith(f".{suffix}") for suffix in module_suffixes):
            continue

        is_vision = _is_vision_module(module_name)
        is_language = _is_language_module(module_name)
        resolved_module_name, was_redirected = _resolve_supported_lora_target(module_name, module)

        is_selected = False
        if finetune_vision_layers and finetune_language_layers:
            target_modules.append(resolved_module_name)
            is_selected = True
        elif finetune_vision_layers and is_vision:
            target_modules.append(resolved_module_name)
            is_selected = True
        elif finetune_language_layers and is_language:
            target_modules.append(resolved_module_name)
            is_selected = True

        if is_selected and was_redirected:
            redirected_modules.append((module_name, resolved_module_name))

    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError(
            "No LoRA target modules matched. Check finetune_* flags and model architecture names."
        )

    is_kbit_model = bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
    )
    if is_kbit_model:
        print("Preparing HF model for k-bit training.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    print(f"Applying LoRA adapters [hf] (r={r}, alpha={alpha}, dropout={dropout})")
    print(f"  Matched {len(target_modules)} target modules.")
    if redirected_modules:
        print(
            f"  Redirected {len(set(redirected_modules))} wrapped modules to supported leaf linears "
            "(for example Gemma4ClippableLinear -> .linear)."
        )

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


def _load_hf_inference(
    model_path: str,
    load_in_4bit: bool = False,
) -> Tuple:
    """Load model for inference via HuggingFace + PEFT."""
    from peft import PeftModel
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoHFVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoHFVisionModel

    print(f"Loading model for inference [hf]: {model_path}")

    is_lora = os.path.isfile(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", model_path)

        print(f"  Detected LoRA adapter. Base model: {base_model_name}")
        # Auto-disable 4-bit for MoE A3B models (not supported)
        effective_4bit = load_in_4bit
        if load_in_4bit and "a3b" in base_model_name.lower():
            print("  Note: A3B (MoE) model detected — disabling 4-bit, using bf16.")
            effective_4bit = False
        model, _ = _load_hf(base_model_name, load_in_4bit=effective_4bit)
        try:
            model = PeftModel.from_pretrained(model, model_path)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to apply LoRA adapter to base model '{base_model_name}'. "
                "This typically happens when the model is partially CPU-offloaded "
                "due to insufficient GPU VRAM. Try a node with more GPU memory "
                "(e.g. watgpu508/708/808/1008 with >100GB VRAM)."
            ) from exc
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
        mode_label = "4-bit NF4" if load_in_4bit else "bf16"
        print(f"  Inference precision: {mode_label}")
        load_kwargs = _hf_inference_load_kwargs(model_path, load_in_4bit)
        try:
            model = AutoHFVisionModel.from_pretrained(
                model_path,
                **load_kwargs,
            )
        except Exception as exc:
            if isinstance(exc, KeyError) and "qwen3_5" in str(exc):
                raise RuntimeError(
                    "Current transformers build does not support Qwen3.5 model_type 'qwen3_5'. "
                    f"Installed transformers={transformers.__version__}. "
                    "Upgrade transformers in the inference environment or use an Unsloth Qwen3-VL model."
                ) from exc
            from transformers import AutoModelForCausalLM
            print(
                f"Vision model auto-loader failed ({type(exc).__name__}); "
                "falling back to AutoModelForCausalLM."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    print("Model loaded and set to inference mode.")
    return model, processor


def _generate_hf(model, processor, images, prompt, **gen_kwargs) -> str:
    """Generate response using HuggingFace processor API.

    Args:
        images: List of PIL Image objects.
    """
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

    user_content = [{"type": "text", "text": prompt}]
    for img in images:
        user_content.append({"type": "image", "image": img})
    messages.append({"role": "user", "content": user_content})

    input_text = apply_chat_template_with_fallback(
        processor,
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
        images=images if images else None,
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
    load_in_4bit: bool = False,
) -> Tuple:
    """Load a base model and its tokenizer/processor.

    Args:
        model_name: HuggingFace model ID or local path.
        backend: "unsloth" or "hf".
        load_in_4bit: Use 4-bit quantization. For HF this enables a
            bitsandbytes NF4 QLoRA load when supported.

    Returns:
        Tuple of (model, tokenizer_or_processor).
    """
    if backend == "unsloth":
        return _load_unsloth(model_name, load_in_4bit)
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
    load_in_4bit: bool = False,
) -> Tuple:
    """Load a model for inference (auto-detects LoRA adapters).

    Args:
        model_path: Base model ID or path to a LoRA adapter directory.
        backend: "unsloth" or "hf".
        load_in_4bit: Use 4-bit quantization. For HF this enables a
            bitsandbytes NF4 load when supported.

    Returns:
        Tuple of (model, tokenizer_or_processor).
    """
    if backend == "unsloth":
        return _load_unsloth_inference(model_path, load_in_4bit)
    return _load_hf_inference(model_path, load_in_4bit)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    image: Optional[PILImage.Image] = None,
    images: Optional[list] = None,
    backend: str = "hf",
    max_new_tokens: int = 4096,
    temperature: float = 1.5,
    min_p: float = 0.1,
    stream: bool = False,
    thinking_budget: Optional[int] = 64,
    show_thinking: bool = False,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate a response for image(s) and a text prompt.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer (unsloth) or processor (hf).
        prompt: Text instruction for the model.
        image: A single PIL Image (backward compat — wrapped into a list).
        images: List of PIL Image objects for multi-image prompts.
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
    # Resolve image list: prefer explicit images param, fallback to single image
    resolved_images = images if images else ([image] if image else [])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
        stream=stream,
        repetition_penalty=repetition_penalty,
    )
    if backend == "unsloth":
        return _generate_unsloth(model, tokenizer, resolved_images, prompt, **gen_kwargs)
    else:
        gen_kwargs["thinking_budget"] = thinking_budget
        gen_kwargs["show_thinking"] = show_thinking
        return _generate_hf(model, tokenizer, resolved_images, prompt, **gen_kwargs)
