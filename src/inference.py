import os
from unsloth import FastVisionModel
from datasets import load_dataset, Image
from transformers import TextStreamer

dataset = load_dataset(
    "json",
    data_files="dataset/EmoArt-5k/annotation.json",
    split="train"
)
DATASET_ROOT = os.path.abspath("dataset/EmoArt-5k")
def fix_path(example):
    example["image_path"] = os.path.join(
        DATASET_ROOT,
        example["image_path"].replace("\\", "/")
    )
    return example
dataset = dataset.map(fix_path)
dataset = dataset.cast_column("image_path", Image())

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_model/qwen3_vl_8b_emoart_5k_v1", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference

image = dataset[0]["image_path"]

instruction = "Describe the artistic style and emotional content of this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 4096,
                   use_cache = True, temperature = 1.5, min_p = 0.1)