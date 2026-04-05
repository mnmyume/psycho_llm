"""
Gradio web UI for Psycho LLM — Interactive sandbox image analysis.

Provides a user-friendly interface where users can:
  1. Upload a sandbox drawing or artwork image
  2. Optionally upload a second image (brush texture) for grid-003 workflows
  3. Enter a text prompt (or use a preset analysis prompt)
  4. Get a detailed psychological analysis from the multi-modal LLM

The model is loaded once on startup for fast repeated inference.

Usage:
    # With a fine-tuned LoRA model:
    python src/app.py --model_path lora_model/sandbox_001_qwen3vl32b_v1
    
    # With the base model (no LoRA):
    python src/app.py --model_path unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit

    # Custom port and shareable link:
    python src/app.py --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 --port 7861 --share

    Wait until you see: Launching Gradio server on port 7860...
    Then run based on the node on your local machine:
    ssh -N -L 7860:watgpu[NODE_NUM]:7860 h5dai@watgpu

    Then open http://localhost:7860 in your browser.
"""

import argparse
import gc
import os
import sys
import torch

# Ensure the src/ directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use a user-specific temp dir to avoid PermissionError on shared systems
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.expanduser("~"), ".gradio_tmp")

import gradio as gr

from models.model_utils import load_model_for_inference, generate_response
from prompts import (
    EXPERT_REASONING_PROMPT,
    GRID_TEST_PROMPT,
    GRID_003_SYSTEM_PROMPT,
    GRID_003_USER_PROMPT_TEMPLATE,
)


# ----- Default Prompts -----
PROMPTS = {
    "Sandbox Scoring": EXPERT_REASONING_PROMPT,
    "Grid Cell Selection": GRID_TEST_PROMPT,
    "Grid-003 Texture Mapping": GRID_003_USER_PROMPT_TEMPLATE,
    "Custom Prompt": "",
}

# Prompt types that require a second image (brush texture)
MULTI_IMAGE_PROMPTS = {"Grid-003 Texture Mapping"}

# Prompt types that require pixel boundary metadata
BOUNDARY_PROMPTS = {"Grid-003 Texture Mapping"}


# ----- Available Models -----
# Each entry maps a display name to (model_path, backend)
MODELS = {
    "LoRA (Grid-003-Qwen3.5-35B)": ("lora_model/grid_003_qwen3.5_35b_v1", "hf"),
    "LoRA (Grid-003-Qwen3-VL-32B)": ("lora_model/grid_003_qwen3vl32b_v1", "unsloth"),
    "LoRA (Grid-002-Qwen3-VL-32B)": ("lora_model/grid_002_qwen3vl32b_v1", "unsloth"),
    "LoRA (Grid-001-Qwen3.5-9B)": ("lora_model/grid_001_qwen3.5-9b_v1", "hf"),
    "Base Model (Qwen3.5-9B)": ("Qwen/Qwen3.5-9B", "hf"),
    "Base Model (Qwen3.5-35B-A3B)": ("Qwen/Qwen3.5-35B-A3B", "hf"),
    "Base Model (Qwen3-VL-32B)": ("unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit", "unsloth"),
    "Base Model (Qwen3-VL-8B)": ("unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", "unsloth"),
}

# Global variables to cache the currently loaded model and its path
current_model = None
current_tokenizer = None
current_model_path = ""
current_backend = ""

def load_or_get_model(model_path: str, backend: str = "hf", load_in_4bit: bool = False):
    global current_model, current_tokenizer, current_model_path, current_backend
    if current_model is None or current_model_path != model_path:
        # Clear old model from memory before loading the new one
        if current_model is not None:
            print("Clearing previous model from VRAM...")
            del current_model
            del current_tokenizer
            current_model = None
            current_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Loading model: {model_path} (backend={backend})...")
        current_model, current_tokenizer = load_model_for_inference(
            model_path=model_path,
            backend=backend,
            load_in_4bit=load_in_4bit,
        )
        current_model_path = model_path
        current_backend = backend
    return current_model, current_tokenizer, current_backend


def create_app(initial_model_path):
    """Build and return the Gradio interface.

    Returns:
        A Gradio Blocks app ready to launch.
    """

    def analyze_image(
        image, brush_image, pixel_boundary_text,
        prompt_type, custom_prompt,
        temperature, max_tokens, thinking_budget, show_thinking,
        selected_model_key,
    ):
        """Process uploaded image(s) and return the model's analysis."""
        if image is None:
            return "Please upload an image first."

        # Determine the prompt text
        if prompt_type == "Custom Prompt":
            prompt = custom_prompt
        elif prompt_type == "Grid-003 Texture Mapping":
            # Substitute {pixel_boundary} placeholder
            boundary_str = pixel_boundary_text.strip() if pixel_boundary_text else ""
            prompt = GRID_003_USER_PROMPT_TEMPLATE.format(pixel_boundary=boundary_str)
        else:
            prompt = PROMPTS[prompt_type]

        if not prompt.strip():
            return "Please enter a prompt."

        try:
            # Dynamically load the model if the selection has changed
            model_entry = MODELS.get(selected_model_key, (initial_model_path, "hf"))
            model_path, backend = model_entry
            model, tokenizer, backend = load_or_get_model(model_path, backend)

            # Build image list
            images = [image]
            if prompt_type in MULTI_IMAGE_PROMPTS and brush_image is not None:
                images.append(brush_image)

            # Determine system prompt for grid-003
            system_prompt = None
            if prompt_type == "Grid-003 Texture Mapping":
                system_prompt = GRID_003_SYSTEM_PROMPT

            # For grid-003, prepend system prompt into the prompt
            # (since generate_response doesn't have a system_prompt param,
            #  we use the full prompt text which already contains the instructions)
            final_prompt = prompt
            if system_prompt:
                final_prompt = system_prompt + "\n\n" + prompt

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=final_prompt,
                images=images,
                backend=backend,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                thinking_budget=int(thinking_budget) if thinking_budget else None,
                show_thinking=show_thinking,
            )
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during analysis: {str(e)}"

    def update_prompt_visibility(prompt_type):
        """Show/hide the custom prompt textbox and multi-image inputs."""
        show_custom = prompt_type == "Custom Prompt"
        show_brush = prompt_type in MULTI_IMAGE_PROMPTS
        show_boundary = prompt_type in BOUNDARY_PROMPTS
        return (
            gr.update(visible=show_custom),
            gr.update(visible=show_brush),
            gr.update(visible=show_boundary),
        )

    # ----- Build the UI -----
    with gr.Blocks(
        title="Psycho LLM — Sandbox Image Analyzer",
    ) as app:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>Psycho LLM</h1>
                <p>Upload a sandbox drawing or artwork for AI-powered psychological analysis</p>
            </div>
        """)

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image (Scene)",
                    type="pil",
                    height=400,
                )

                brush_image_input = gr.Image(
                    label="Upload Brush Texture (Grid-003 only)",
                    type="pil",
                    height=150,
                    visible=True,
                )

                pixel_boundary_input = gr.Textbox(
                    label="Pixel Boundary (Grid-003 only)",
                    placeholder="e.g. 278, 221, 37, 36",
                    visible=True,
                    info="Comma-separated values: originalX, originalY, width, height",
                )

                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="Selected Model",
                    info="If you change this, the new model will be loaded on the first request (which takes a minute).",
                )

                prompt_type = gr.Dropdown(
                    choices=list(PROMPTS.keys()),
                    value="Grid-003 Texture Mapping",
                    label="Analysis Type",
                )

                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter your custom analysis prompt...",
                    lines=3,
                    visible=False,
                )

                with gr.Accordion("Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.1, step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused",
                    )
                    max_tokens = gr.Slider(
                        minimum=256, maximum=32768, value=8192, step=256,
                        label="Max Tokens",
                        info="Maximum length of the generated response",
                    )
                    thinking_budget = gr.Slider(
                        minimum=0, maximum=256, value=64, step=32,
                        label="Thinking Budget (Qwen3.5 only)",
                        info="Max thinking tokens. Default 64 keeps reasoning short. 0 disables thinking.",
                    )
                    show_thinking = gr.Checkbox(
                        value=True,
                        label="Show Thinking Text (Qwen3.5 only)",
                    )

                analyze_btn = gr.Button(
                    "Analyze Image",
                    variant="primary",
                    size="lg",
                )

            # Right column: output
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Analysis Result",
                    lines=25,
                )

        # Event handlers
        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt, brush_image_input, pixel_boundary_input],
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[
                image_input, brush_image_input, pixel_boundary_input,
                prompt_type, custom_prompt,
                temperature, max_tokens, thinking_budget, show_thinking,
                model_selector,
            ],
            outputs=[output],
        )

    return app


def main():
    """Parse arguments, load model, and launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the Psycho LLM web UI.")
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to LoRA adapter directory or base model ID. Defaults to the first model in the MODELS list.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the Gradio server on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a shareable public link.",
    )
    args = parser.parse_args()

    # Resolve initial model: use --model_path if provided, otherwise first entry in MODELS
    if args.model_path:
        initial_model_path = args.model_path
        initial_backend = "hf"
    else:
        first_key = list(MODELS.keys())[0]
        initial_model_path, initial_backend = MODELS[first_key]

    # Load the initial model
    print(f"Loading initial model for the web UI: {initial_model_path}")
    load_or_get_model(initial_model_path, initial_backend)

    # Build and launch the app
    print("Building Gradio UI...")
    app = create_app(initial_model_path=initial_model_path)
    print(f"Launching Gradio server on port {args.port}...")
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
    print("Gradio server exited.")


if __name__ == "__main__":
    main()
