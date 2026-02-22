"""
Training script for LoRA fine-tuning of Qwen3-VL models.

This module fine-tunes a multi-modal Qwen3-VL model on an emotion/style
dataset (e.g. EmoArt) using Parameter-Efficient Fine-Tuning (PEFT) via LoRA.
Only the small LoRA adapter weights are saved — the base model is never modified.

Usage:
    # Using a YAML recipe:
    python src/train.py --config recipes/Qwen3-VL-8B.yaml --run_name my_exp_v1

    # With CLI overrides:
    python src/train.py --config recipes/Qwen3-VL-8B.yaml --run_name my_exp_v2 --num_epochs 3

    # Via SLURM:
    sbatch train.sh
"""

import argparse
import os
import sys

# Ensure the src/ directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers.trainer_utils import get_last_checkpoint

from config import TrainingConfig
from models.model_utils import load_base_model, apply_lora_for_training
from data_loaders.emoart import EmoArt


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with LoRA on an image-emotion dataset."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML recipe file (e.g. recipes/Qwen3-VL-8B.yaml).",
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


def train(config: TrainingConfig):
    """Run the full LoRA fine-tuning pipeline.

    Steps:
        1. Load the EmoArt dataset
        2. Load the base Qwen3-VL model
        3. Apply LoRA adapters for training
        4. Configure and run the SFTTrainer
        5. Save the trained LoRA adapter weights
    """
    print("=" * 60)
    print(f"  Psycho LLM — LoRA Fine-Tuning")
    print(f"  Model:    {config.model_name}")
    print(f"  Dataset:  {config.dataset_path}")
    print(f"  Run:      {config.run_name}")
    print(f"  Output:   {config.output_dir}")
    print(f"  LoRA dir: {config.lora_save_dir}")
    print("=" * 60)

    # --- Step 1: Load Dataset ---
    print("\n[1/5] Loading dataset...")
    dataset = EmoArt(
        annotation_path=config.dataset_path,
        data_dir=config.dataset_dir,
    )
    print(f"  Loaded {len(dataset)} training samples.")

    # --- Step 2: Load Base Model ---
    print("\n[2/5] Loading base model...")
    model, tokenizer = load_base_model(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
    )

    # --- Step 3: Apply LoRA ---
    print("\n[3/5] Applying LoRA adapters...")
    model = apply_lora_for_training(
        model,
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset.data,
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            output_dir=config.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.max_length,
        ),
    )

    # --- Run Training ---
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f"\n  Training complete! Stats: {trainer_stats.metrics}")

    # --- Step 5: Save LoRA Weights ---
    # IMPORTANT: We save only the LoRA adapter weights, NOT the full base model.
    # This typically saves ~50-200MB instead of 16-60GB.
    print(f"\n[5/5] Saving LoRA adapter to: {config.lora_save_dir}")
    os.makedirs(config.lora_save_dir, exist_ok=True)
    model.save_pretrained(config.lora_save_dir)
    tokenizer.save_pretrained(config.lora_save_dir)
    print("  Done! LoRA adapter saved successfully.")


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