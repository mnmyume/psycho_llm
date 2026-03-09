"""
Configuration module for the Psycho LLM project.

Provides a dataclass-based configuration system with YAML loading support.
Training recipes (YAML files in recipes/) define model, dataset, and
hyperparameter settings that can be overridden via CLI arguments.
"""

import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning of Qwen3-VL models.

    Attributes:
        model_name: HuggingFace model ID or local path (e.g. unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit).
        dataset_path: Local path to the dataset annotation JSON file.
        dataset_dir: Directory containing the dataset images.
        run_name: Unique identifier for this training run, used for checkpoint and LoRA save dirs.
        output_dir: Directory where training checkpoints are saved during training.
        lora_save_dir: Directory where the final LoRA adapter weights are saved after training.
        load_in_4bit: Whether to load the base model in 4-bit quantization (saves VRAM).

        # LoRA configuration
        lora_r: LoRA rank — controls the dimensionality of the low-rank matrices.
                Higher values = more capacity but more parameters. 16 is a good default.
        lora_alpha: LoRA scaling factor. Typically set equal to lora_r.
                    The effective learning rate for LoRA layers is scaled by (alpha / r).
        lora_dropout: Dropout applied to LoRA layers. 0 is fine for most cases.
        finetune_vision_layers: Whether to apply LoRA to the vision encoder layers.
        finetune_language_layers: Whether to apply LoRA to the language model layers.
        finetune_attention_modules: Whether to apply LoRA to attention modules (Q, K, V, O projections).
        finetune_mlp_modules: Whether to apply LoRA to MLP/FFN modules.

        # Training hyperparameters
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps before an optimizer update.
        learning_rate: Peak learning rate for the optimizer.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        max_length: Maximum sequence length for tokenization.
        lr_scheduler_type: Learning rate scheduler type (e.g. 'linear', 'cosine').
        weight_decay: Weight decay coefficient for AdamW.
        seed: Random seed for reproducibility.

        # Inference settings (used by inference.py and app.py)
        inference_max_tokens: Maximum number of new tokens to generate during inference.
        inference_temperature: Sampling temperature — higher = more creative, lower = more deterministic.
        inference_min_p: Minimum probability threshold for nucleus sampling.
    """

    # --- Model ---
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    backend: str = "hf"  # "unsloth" or "hf"
    load_in_4bit: bool = True

    # --- Dataset ---
    dataset_path: str = "dataset/EmoArt-130k/Annotation.json"
    dataset_dir: str = "dataset/EmoArt-130k"

    # --- Run ---
    run_name: str = "default_run"
    output_dir: str = "outputs/{run_name}"
    lora_save_dir: str = "lora_model/{run_name}"

    # --- LoRA Configuration ---
    # LoRA (Low-Rank Adaptation) injects small trainable matrices into
    # frozen model layers. This lets us fine-tune with <1% of parameters.
    lora_r: int = 16               # Rank of the low-rank decomposition
    lora_alpha: int = 16           # Scaling factor (alpha / r scales the LoRA output)
    lora_dropout: float = 0.0      # Dropout on LoRA layers (0 = no dropout)
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    # --- Training Hyperparameters ---
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 5
    max_length: int = 2048
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.001
    seed: int = 3407

    # --- Inference ---
    inference_max_tokens: int = 8192
    inference_temperature: float = 1.5
    inference_min_p: float = 0.1

    def __post_init__(self):
        """Resolve template placeholders in output paths."""
        self.output_dir = self.output_dir.format(run_name=self.run_name)
        self.lora_save_dir = self.lora_save_dir.format(run_name=self.run_name)

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "TrainingConfig":
        """Load configuration from a YAML recipe file, with optional overrides.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Keyword arguments that override values from the YAML file.
                         Only fields defined in the dataclass are accepted.

        Returns:
            A TrainingConfig instance with values from YAML merged with overrides.

        Example:
            config = TrainingConfig.from_yaml(
                "recipes/Qwen3-VL-8B.yaml",
                run_name="my_experiment",
                num_epochs=3,
            )
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Filter to only known fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_yaml = {k: v for k, v in yaml_data.items() if k in valid_fields}
        filtered_overrides = {k: v for k, v in overrides.items() if k in valid_fields and v is not None}

        # Overrides take precedence over YAML values
        merged = {**filtered_yaml, **filtered_overrides}
        return cls(**merged)
