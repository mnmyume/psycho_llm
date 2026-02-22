"""
Gradio web UI for Psycho LLM — Interactive sandbox image analysis.

Provides a user-friendly interface where users can:
  1. Upload a sandbox drawing or artwork image
  2. Enter a text prompt (or use the default psychological analysis prompt)
  3. Get a detailed psychological analysis from the multi-modal LLM

The model is loaded once on startup for fast repeated inference.

Usage:
    python src/app.py --model_path lora_model/qwen3_vl_8b_emoart_5k_v1

    # Custom port and shareable link:
    python src/app.py --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 --port 7861 --share
"""

import argparse
import os
import sys

# Ensure the src/ directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from models.model_utils import load_model_for_inference, generate_response


# ----- Default Prompts -----
PROMPTS = {
    "🧠 Psychological Analysis": (
        "Analyze this sandbox image. Describe the psychological themes, "
        "emotional state, and potential personality traits of the creator. "
        "Your answer should use JSON format."
    ),
    "🎨 Artistic Style & Emotion": (
        "Describe the artistic style and emotional content of this image. "
        "Your answer should use JSON format."
    ),
    "💼 Career Recommendation": (
        "Based on this sandbox drawing, analyze the creator's personality traits, "
        "interests, and strengths. Then recommend 3 suitable careers with explanations. "
        "Your answer should use JSON format."
    ),
    "❤️ Mental Health Assessment": (
        "Analyze this sandbox image for indicators of the creator's current mental state. "
        "Consider the use of color, space, symbols, and composition. "
        "Provide observations and gentle suggestions. "
        "Your answer should use JSON format."
    ),
    "✏️ Custom Prompt": "",
}


def create_app(model, tokenizer):
    """Build and return the Gradio interface.

    Args:
        model: The loaded Qwen3-VL model (base or with LoRA).
        tokenizer: The model's tokenizer.

    Returns:
        A Gradio Blocks app ready to launch.
    """

    def analyze_image(image, prompt_type, custom_prompt, temperature, max_tokens):
        """Process an uploaded image and return the model's analysis."""
        if image is None:
            return "⚠️ Please upload an image first."

        # Use custom prompt if selected, otherwise use the preset
        prompt = custom_prompt if prompt_type == "✏️ Custom Prompt" else PROMPTS[prompt_type]

        if not prompt.strip():
            return "⚠️ Please enter a prompt."

        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                image=image,
                prompt=prompt,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
            )
            return response
        except Exception as e:
            return f"❌ Error during analysis: {str(e)}"

    def update_prompt_visibility(prompt_type):
        """Show/hide the custom prompt textbox based on selection."""
        if prompt_type == "✏️ Custom Prompt":
            return gr.update(visible=True)
        return gr.update(visible=False)

    # ----- Build the UI -----
    with gr.Blocks(
        title="Psycho LLM — Sandbox Image Analyzer",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css="""
            .main-header {
                text-align: center;
                padding: 1rem 0;
            }
            .main-header h1 {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
            }
            .main-header p {
                color: #6b7280;
                font-size: 1.1rem;
            }
        """,
    ) as app:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🧠 Psycho LLM</h1>
                <p>Upload a sandbox drawing or artwork for AI-powered psychological analysis</p>
            </div>
        """)

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Upload Image",
                    type="pil",
                    height=400,
                )

                prompt_type = gr.Dropdown(
                    choices=list(PROMPTS.keys()),
                    value="🧠 Psychological Analysis",
                    label="🔮 Analysis Type",
                )

                custom_prompt = gr.Textbox(
                    label="✏️ Custom Prompt",
                    placeholder="Enter your custom analysis prompt...",
                    lines=3,
                    visible=False,
                )

                with gr.Accordion("⚙️ Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.5, step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused",
                    )
                    max_tokens = gr.Slider(
                        minimum=256, maximum=8192, value=4096, step=256,
                        label="Max Tokens",
                        info="Maximum length of the generated response",
                    )

                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg",
                )

            # Right column: output
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="📋 Analysis Result",
                    lines=25,
                    show_copy_button=True,
                )

        # Event handlers
        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt],
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, prompt_type, custom_prompt, temperature, max_tokens],
            outputs=[output],
        )

    return app


def main():
    """Parse arguments, load model, and launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the Psycho LLM web UI.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to LoRA adapter directory or base model ID.",
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

    # Load model once at startup
    print("Loading model for the web UI...")
    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        load_in_4bit=not args.no_4bit,
    )

    # Build and launch the app
    app = create_app(model, tokenizer)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
