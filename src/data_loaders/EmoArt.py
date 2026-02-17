import json
from .Dataset import Dataset
from datasets import load_dataset, Image
from constants.paths import *


class EmoArt(Dataset):
    def __init__(
            self,
            path: str = EMOART_130K_DATA_PATH,
    ):
        super().__init__(path)
        self.messages = None;

        self.load()

    def load(self):
        hf_dataset = load_dataset(
            "json",
            data_files=self.path,
            split="train"
        )

        def process(example):
            # Fix image path: "Images\Abs..." -> "/root/data/Images/Abs..."
            rel_path = example["image_path"].replace("\\", "/")
            abs_path = os.path.join(EMOART_130K_DATA_DIR, rel_path)
            example["image_path"] = abs_path
            example["image"] = abs_path

            instruction = "Describe the artistic style and emotional content of this image. Your answer should use JSON format."
            desc = example["description"]

            if isinstance(desc, dict):
                text_content = json.dumps(desc, ensure_ascii=False)
            else:
                text_content = desc

            example["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": abs_path}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(text_content)}]
                }
            ]
            return example

        self.data = hf_dataset.map(process)
        self.data = self.data.cast_column("image", Image())
        self.messages = self.data.select_columns(["messages"])


    @staticmethod
    def get_prompt(item):
        if "prompt" in item:
            return f"{item['prompt']}"
        elif "text" in item:
            return f"{item['text']}"
        else:
            raise Exception("No prompt or text in item")
