"""
Project path constants.

All paths are resolved relative to the project root directory,
making them work correctly regardless of the working directory.
"""

from pathlib import Path

# Project root — two levels up from src/constants/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- EmoArt Dataset Paths ---
EMOART_5K_DATA_DIR = str(PROJECT_ROOT / "dataset" / "EmoArt-5k")
EMOART_130K_DATA_DIR = str(PROJECT_ROOT / "dataset" / "EmoArt-130k")

EMOART_5K_ANNOTATION = str(PROJECT_ROOT / "dataset" / "EmoArt-5k" / "annotation.json")
EMOART_130K_ANNOTATION = str(PROJECT_ROOT / "dataset" / "EmoArt-130k" / "Annotation.json")

# --- Model Paths ---
LORA_MODEL_DIR = str(PROJECT_ROOT / "lora_model")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
