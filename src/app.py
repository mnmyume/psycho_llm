"""
Gradio web UI for Psycho LLM — Interactive sandbox image analysis.

Provides a user-friendly interface where users can:
  1. Upload a sandbox drawing or artwork image
  2. Enter a text prompt (or use the default psychological analysis prompt)
  3. Get a detailed psychological analysis from the multi-modal LLM

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
from prompts import EXPERT_REASONING_PROMPT, GRID_TEST_PROMPT


# ----- Default Prompts -----
PROMPTS = {
    "Sandbox Scoring": EXPERT_REASONING_PROMPT,
    "Grid Cell Selection": GRID_TEST_PROMPT,
    "Custom Prompt": "",
}


# ----- Available Models -----
# Each entry maps a display name to (model_path, backend)
MODELS = {
    "LoRA (Grid-001-Qwen3.5-9B)": ("lora_model/grid_001_qwen3.5-9b_v1", "hf"),
    "LoRA (Grid-001-Qwen3-VL-32B)": ("lora_model/grid_001_qwen3vl32b_v1", "unsloth"),
    "Base Model (Qwen3.5-35B-A3B)": ("Qwen/Qwen3.5-35B-A3B", "hf"),
    "Base Model (Qwen3-VL-32B)": ("unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit", "unsloth"),
    "Base Model (Qwen3-VL-8B)": ("unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", "unsloth"),
}

# Global variables to cache the currently loaded model and its path
current_model = None
current_tokenizer = None
current_model_path = ""
current_backend = ""

def load_or_get_model(model_path: str, backend: str = "hf", load_in_4bit: bool = True):
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


def create_app(initial_model_path, load_in_4bit):
    """Build and return the Gradio interface.

    Args:
        model: The loaded Qwen3-VL model (base or with LoRA).
        tokenizer: The model's tokenizer.

    Returns:
        A Gradio Blocks app ready to launch.
    """

    def analyze_image(image, prompt_type, custom_prompt, temperature, max_tokens, thinking_budget, selected_model_key):
        """Process an uploaded image and return the model's analysis."""
        if image is None:
            return "Please upload an image first."

        # Use custom prompt if selected, otherwise use the preset
        prompt = custom_prompt if prompt_type == "Custom Prompt" else PROMPTS[prompt_type]

        if not prompt.strip():
            return "Please enter a prompt."

        try:
            # Dynamically load the model if the selection has changed
            model_entry = MODELS.get(selected_model_key, (initial_model_path, "hf"))
            model_path, backend = model_entry
            model, tokenizer, backend = load_or_get_model(model_path, backend, load_in_4bit)

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                image=image,
                prompt=prompt,
                backend=backend,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                thinking_budget=int(thinking_budget) if thinking_budget else None,
            )
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during analysis: {str(e)}"

    def update_prompt_visibility(prompt_type):
        """Show/hide the custom prompt textbox based on selection."""
        if prompt_type == "Custom Prompt":
            return gr.update(visible=True)
        return gr.update(visible=False)

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
                    label="Upload Image",
                    type="pil",
                    height=400,
                )

                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="Selected Model",
                    info="If you change this, the new model will be loaded on the first request (which takes a minute).",
                )

                prompt_type = gr.Dropdown(
                    choices=list(PROMPTS.keys()),
                    value="Grid Cell Selection",
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
                        minimum=0, maximum=2048, value=0, step=64,
                        label="Thinking Budget (Qwen3.5 only)",
                        info="Max thinking tokens. 0 = disable thinking for fastest response.",
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
            outputs=[custom_prompt],
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, prompt_type, custom_prompt, temperature, max_tokens, thinking_budget, model_selector],
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
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization.",
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
    load_or_get_model(initial_model_path, initial_backend, not args.no_4bit)

    # Build and launch the app
    print("Building Gradio UI...")
    app = create_app(initial_model_path=initial_model_path, load_in_4bit=not args.no_4bit)
    print(f"Launching Gradio server on port {args.port}...")
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
    print("Gradio server exited.")


if __name__ == "__main__":
    main()
