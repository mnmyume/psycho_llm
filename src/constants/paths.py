import os
from os.path import join, dirname

# EmoArt Dataset
EMOART_5K_DATA_DIR = join(
    "dataset",
    "EmoArt-5k",
)

EMOART_130K_DATA_DIR = join(
    "dataset",
    "EmoArt-130k",
)

EMOART_5K_DATA_PATH = join(
    EMOART_5K_DATA_DIR,
    "annotation.json"
)

EMOART_130K_DATA_PATH = join(
    EMOART_130K_DATA_DIR,
    "Annotation.json"
)
