import os
from unsloth import FastVisionModel
from datasets import load_dataset, Image
from transformers import TextStreamer

from data_loaders.EmoArt import EmoArt

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_model/qwen3_vl_8b_emoart_130k_v1", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference

train_dataset = EmoArt()
sample = train_dataset[0]
image = sample["image"]
instruction = "Describe the artistic style and emotional content of this image. Your answer should use JSON format."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "Sample image."},
            {"type": "text", "text": instruction}
        ]
    }
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