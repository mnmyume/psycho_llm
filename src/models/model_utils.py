"""
Model loading and inference utilities for Qwen3-VL with LoRA support.

This module centralizes all model lifecycle operations:
  - Loading the base Qwen3-VL model via unsloth
  - Applying LoRA adapters for training
  - Loading a fine-tuned LoRA model for inference
  - Running generation with proper message formatting

The unsloth library provides optimized model loading and LoRA injection
that is compatible with the standard Hugging Face + PEFT ecosystem.
"""

import torch
from PIL import Image as PILImage
from typing import Optional, Tuple

from unsloth import FastVisionModel
from transformers import TextStreamer


def load_base_model(
    model_name: str,
    load_in_4bit: bool = True,
) -> Tuple:
    """Load a base Qwen3-VL model and its tokenizer.

    Uses unsloth's FastVisionModel for optimized loading with optional
    4-bit quantization (bitsandbytes NF4) to reduce VRAM usage.

    Args:
        model_name: HuggingFace model ID or local path.
                    Example: "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        load_in_4bit: If True, loads model weights in 4-bit NF4 quantization.
                      This reduces VRAM from ~16GB to ~5GB for the 8B model.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading base model: {model_name} (4-bit={load_in_4bit})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def apply_lora_for_training(
    model,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
):
    """Apply LoRA (Low-Rank Adaptation) adapters to the model for training.

    LoRA works by injecting pairs of small trainable matrices (A and B) into
    existing model layers. Instead of updating a full weight matrix W (d×d),
    LoRA learns W + BA where B (d×r) and A (r×d), with r << d.

    This means we only train r*d*2 parameters per layer instead of d*d,
    achieving ~99% parameter reduction while maintaining quality.

    Args:
        model: The base model returned by load_base_model().
        r: LoRA rank. Controls capacity of the low-rank matrices.
           Higher = more expressive but more parameters. 16 is a good default.
        alpha: Scaling factor for LoRA. The LoRA output is scaled by alpha/r.
               Setting alpha = r gives a scaling of 1.0.
        dropout: Dropout rate on LoRA layers. 0 is fine for most fine-tuning.
        finetune_vision_layers: Apply LoRA to vision encoder (for image understanding).
        finetune_language_layers: Apply LoRA to language model layers.
        finetune_attention_modules: Apply LoRA to Q, K, V, O attention projections.
        finetune_mlp_modules: Apply LoRA to feed-forward network layers.

    Returns:
        The model with LoRA adapters applied, set to training mode.
    """
    print(f"Applying LoRA adapters (r={r}, alpha={alpha}, dropout={dropout})")
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


def load_model_for_inference(
    model_path: str,
    load_in_4bit: bool = True,
) -> Tuple:
    """Load a model for inference, with automatic LoRA detection.

    If model_path points to a saved LoRA adapter directory (containing
    adapter_config.json), unsloth will automatically load the base model
    and merge the LoRA weights. If it points to a base model, it loads
    that directly.

    Args:
        model_path: Path to a saved LoRA adapter directory or a base model ID.
                    Example LoRA: "lora_model/qwen3_vl_8b_emoart_5k_v1"
                    Example base: "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    print(f"Loading model for inference: {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded and set to inference mode.")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    image: PILImage.Image,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 1.5,
    min_p: float = 0.1,
    stream: bool = False,
) -> str:
    """Generate a psychological analysis response for an image and prompt.

    Constructs a multi-modal chat message with the image and text prompt,
    tokenizes it using the model's chat template, and runs generation.

    Args:
        model: The loaded model (base or base + LoRA).
        tokenizer: The model's tokenizer.
        image: A PIL Image object (the sandbox drawing or artwork).
        prompt: The text instruction for the model.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        min_p: Minimum probability threshold for sampling.
        stream: If True, streams output to stdout during generation.

    Returns:
        The generated text response as a string.
    """
    # Build the multi-modal message in Qwen3-VL chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply the chat template to format the message with special tokens
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # Tokenize both the image and formatted text together
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    # Set up optional streaming
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p,
        )

    # Decode only the newly generated tokens (skip the input prompt tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()
