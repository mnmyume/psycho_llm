"""
Training script for LoRA fine-tuning of multimodal models.

Supports two backends:
  - "unsloth": Optimized loading via FastVisionModel + UnslothVisionDataCollator
  - "hf":      Standard HuggingFace transformers + PEFT

The backend is selected via the `backend` field in the recipe YAML.

Usage:
    # Unsloth backend (Qwen3-VL):
    python src/train.py --config recipes/Qwen3-VL-32B.yaml --run_name qwen3_vl_32B_v1

    # HuggingFace backend (Qwen3.5 MoE):
    python src/train.py --config recipes/sandbox-001-qwen3.5-35b.yaml --run_name sandbox_001_v1

    # Via SLURM:
    sbatch train.sh
"""

import argparse
import os
import sys

# Ensure the src/ directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parse args early so we can import unsloth before trl/transformers if needed
def _maybe_import_unsloth():
    """Import unsloth before other ML libraries if the recipe uses the unsloth backend."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config":
            config_path = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            if config_path:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if cfg.get("backend") == "unsloth":
                    import unsloth  # noqa: F401 — must be imported before trl/transformers
            break

_maybe_import_unsloth()

from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

from config import TrainingConfig
from models.model_utils import load_base_model, apply_lora_for_training
from data_loaders.emoart import EmoArt
from data_loaders.sandbox import SandboxDataset


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a multimodal model with LoRA on an image dataset."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML recipe file (e.g. recipes/sandbox-001-qwen3.5-35b.yaml).",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Override model name.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override annotation JSON path.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset image directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Unique name for this training run.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha.")
    return parser.parse_args()


def resolve_optimizer(config: TrainingConfig) -> str:
    """Select an optimizer suited to the selected backend and precision mode."""
    if config.backend == "hf" and config.load_in_4bit:
        return "paged_adamw_32bit"
    return "adamw_8bit"


def train(config: TrainingConfig):
    """Run the full LoRA fine-tuning pipeline.

    Steps:
        1. Load the dataset
        2. Load the base model (via selected backend)
        3. Apply LoRA adapters for training
        4. Configure and run the SFTTrainer
        5. Save the trained LoRA adapter weights
    """
    print("=" * 60)
    print(f"  Psycho LLM — LoRA Fine-Tuning")
    print(f"  Backend:  {config.backend}")
    print(f"  Model:    {config.model_name}")
    print(f"  Dataset:  {config.dataset_path}")
    print(f"  Run:      {config.run_name}")
    print(f"  Output:   {config.output_dir}")
    print(f"  LoRA dir: {config.lora_save_dir}")
    print(f"  4-bit:    {config.load_in_4bit}")
    print("=" * 60)

    # --- Step 1: Load Dataset ---
    print("\n[1/5] Loading dataset...")
    if config.dataset_path.endswith(".jsonl"):
        dataset = SandboxDataset(
            annotation_path=config.dataset_path,
            data_dir=config.dataset_dir,
        )
    else:
        dataset = EmoArt(
            annotation_path=config.dataset_path,
            data_dir=config.dataset_dir,
        )

    # Unsloth's SFTTrainer path can misclassify VLM models when a top-level
    # 'image'/'images' column exists. Keep only chat messages for this backend.
    if config.backend == "unsloth":
        cols_to_remove = [
            col for col in ("image", "images")
            if col in dataset.data.column_names
        ]
        if cols_to_remove:
            print(f"  Removing columns for unsloth compatibility: {cols_to_remove}")
            dataset.data = dataset.data.remove_columns(cols_to_remove)

    print(f"  Loaded {len(dataset)} training samples.")

    # --- Step 2: Load Base Model ---
    print("\n[2/5] Loading base model...")
    model, tokenizer = load_base_model(
        model_name=config.model_name,
        backend=config.backend,
        load_in_4bit=config.load_in_4bit,
    )

    # --- Step 3: Apply LoRA ---
    print("\n[3/5] Applying LoRA adapters...")
    model = apply_lora_for_training(
        model,
        backend=config.backend,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
    )

    # --- Step 4: Configure Trainer ---
    print("\n[4/5] Starting training...")

    # Check for existing checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint:
            print(f"  Resuming from checkpoint: {last_checkpoint}")

    # Backend-specific trainer kwargs
    trainer_kwargs = {}
    if config.backend == "unsloth":
        from unsloth.trainer import UnslothVisionDataCollator
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(model, tokenizer)
    else:
        from data_loaders.vision_collator import VisionDataCollator
        trainer_kwargs["data_collator"] = VisionDataCollator(tokenizer, max_length=config.max_length)
    trainer_kwargs["processing_class"] = tokenizer
    optimizer_name = resolve_optimizer(config)
    print(f"  Optimizer: {optimizer_name}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset.data,
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            logging_steps=1,
            optim=optimizer_name,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            output_dir=config.output_dir,
            report_to="none",
            remove_unused_columns=False,
            bf16=True,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.max_length,
        ),
        **trainer_kwargs,
    )

    # --- Run Training ---
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f"\n  Training complete! Stats: {trainer_stats.metrics}")

    # --- Step 5: Save LoRA Weights ---
    print(f"\n[5/5] Saving LoRA adapter to: {config.lora_save_dir}")
    os.makedirs(config.lora_save_dir, exist_ok=True)
    model.save_pretrained(config.lora_save_dir)
    tokenizer.save_pretrained(config.lora_save_dir)
    print("  Done! LoRA adapter saved successfully.")

    # --- Print Peak GPU Memory Usage ---
    import torch
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\n[Stats] Peak GPU Memory Allocated: {max_memory:.2f} GB")


if __name__ == "__main__":
    args = parse_args()

    # Build configuration from YAML + CLI overrides
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }

    if args.config:
        config = TrainingConfig.from_yaml(args.config, **cli_overrides)
    else:
        config = TrainingConfig(**cli_overrides)

    train(config)
