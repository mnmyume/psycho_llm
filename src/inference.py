"""
CLI inference script for Qwen3-VL psychological image analysis.

Loads a trained LoRA model (or base model) and runs inference on a
single image with a text prompt. Outputs the analysis to stdout.

Usage:
    # With a fine-tuned LoRA model:
    python src/inference.py \
        --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 \
        --image path/to/sandbox_drawing.png \
        --prompt "Analyze the psychological themes in this sandbox image."

    # With the base model (no LoRA):
    python src/inference.py \
        --model_path unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit \
        --image path/to/image.jpg \
        --prompt "Describe the emotions expressed in this artwork."

    # With streaming output:
    python src/inference.py \
        --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 \
        --image path/to/image.jpg \
        --prompt "What career might suit the creator of this drawing?" \
        --stream
"""

import argparse
import os
import sys

# Ensure the src/ directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from models.model_utils import load_model_for_inference, generate_response


# Default prompt for psychological sandbox analysis
DEFAULT_PROMPT = (
    "Analyze this sandbox image. Describe the psychological themes, "
    "emotional state, and potential personality traits of the creator. "
    "Your answer should use JSON format."
)


def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Run psychological analysis on an image using Qwen3-VL."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to LoRA adapter directory or base model ID.",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Text prompt / instruction for the model.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help="Sampling temperature (higher = more creative).",
    )
    parser.add_argument(
        "--min_p", type=float, default=0.1,
        help="Minimum probability threshold for sampling.",
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization (use 16-bit instead).",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream output tokens to stdout as they are generated.",
    )
    return parser.parse_args()


def run_inference(args):
    """Load model and run inference on a single image."""
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Load the image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    # Load the model (automatically detects if it's a LoRA adapter or base model)
    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        load_in_4bit=not args.no_4bit,
    )

    # Generate the analysis
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        image=image,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        stream=args.stream,
    )

    if not args.stream:
        print(response)

    print("-" * 60)
    return response


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)