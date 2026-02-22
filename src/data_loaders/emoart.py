"""
EmoArt dataset loader for emotion-annotated artwork images.

The EmoArt dataset (5k and 130k variants) pairs artwork images with
structured emotional and stylistic descriptions. This loader formats
the data into the multi-turn chat message format required by Qwen3-VL
and the SFTTrainer for LoRA fine-tuning.

Dataset structure:
    annotation.json — Each entry has:
        - image_path: relative path to the artwork image
        - description: emotional/stylistic analysis (dict or string)
"""

import json
import os
from datasets import load_dataset, Image

from .base_dataset import BaseDataset


# Default instruction prompt used during training
DEFAULT_INSTRUCTION = (
    "Describe the artistic style and emotional content of this image. "
    "Your answer should use JSON format."
)


class EmoArt(BaseDataset):
    """Loader for the EmoArt emotion-in-art dataset.

    Loads the annotation JSON, resolves image paths, and formats each
    sample into a multi-modal chat message for supervised fine-tuning.

    Args:
        annotation_path: Path to the annotation JSON file.
        data_dir: Directory containing the dataset images (used to
                  resolve relative image paths from annotations).
        instruction: The instruction prompt prepended to each training sample.
                     Can be customized for different analysis tasks.

    Example:
        dataset = EmoArt(
            annotation_path="dataset/EmoArt-5k/annotation.json",
            data_dir="dataset/EmoArt-5k",
        )
        print(len(dataset))       # Number of samples
        print(dataset[0])         # First sample with 'messages' and 'image' columns
    """

    def __init__(
        self,
        annotation_path: str,
        data_dir: str,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        super().__init__(path=annotation_path)
        self.data_dir = data_dir
        self.instruction = instruction
        self.load()

    def load(self):
        """Load and preprocess the EmoArt dataset.

        Steps:
            1. Load the annotation JSON via Hugging Face datasets
            2. Resolve relative image paths to absolute paths
            3. Format each sample into a user/assistant chat message pair
            4. Cast the image column for automatic PIL loading
        """
        hf_dataset = load_dataset(
            "json",
            data_files=self.path,
            split="train",
        )

        data_dir = self.data_dir
        instruction = self.instruction

        def _format_sample(example):
            # Resolve image path: "Images\\Abstract..." -> "/abs/path/Images/Abstract..."
            rel_path = example["image_path"].replace("\\", "/")
            abs_path = os.path.join(data_dir, rel_path)
            example["image_path"] = abs_path
            example["image"] = abs_path

            # Format the description as a JSON string if it's a dict
            desc = example["description"]
            if isinstance(desc, dict):
                text_content = json.dumps(desc, ensure_ascii=False)
            else:
                text_content = str(desc)

            # Build the chat message pair for SFTTrainer
            example["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": abs_path},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_content}],
                },
            ]
            return example

        self.data = hf_dataset.map(_format_sample)
        self.data = self.data.cast_column("image", Image())
