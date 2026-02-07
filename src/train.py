import argparse
import os
import torch
from PIL import Image
from unsloth import FastVisionModel
from datasets import load_dataset, Image
from huggingface_hub import hf_hub_download
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", help="Model name")
parser.add_argument("--dataset_name", type=str, default="printblue/EmoArt-5k", help="HuggingFace dataset path")
parser.add_argument("--run_name", type=str, required=True, help="Unique name for this training run (e.g., emoart_v1)")
args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name
run_name = args.run_name

output_dir = f"outputs/{run_name}"  # Checkpoints
final_save_path = f"models/{run_name}"

instruction = "Describe the artistic style and emotional content of this image."

# load local dataset
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
    model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Configure PEFT
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Conversion Function
def convert_to_conversation(sample):
    output_text = sample["description"]
    if isinstance(output_text, dict):
        output_text = output_text.get('text', str(output_text))

    image_obj = sample["image_path"]

    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": image_obj}]
         },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": output_text}]
         },
    ]
    return {"messages": conversation}


# Apply conversion
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

FastVisionModel.for_training(model)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=30,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

# Start Training
trainer_stats = trainer.train()

# Save
print(f"Saving model to {final_save_path}...")
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)