# Save this as download_training_model.py and run it
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

# NOTE: We are downloading 'bnb-4bit', NOT 'GGUF'
# This version contains the necessary config.json and safetensors
snapshot_download(
    repo_id   = "unsloth/Qwen3-VL-30B-A3B-Instruct-bnb-4bit",
    local_dir = "unsloth/Qwen3-VL-30B-A3B-Instruct-bnb-4bit",
    # We do NOT use allow_patterns like *Q4_K_M* here because
    # we need the full safetensors files, not a single GGUF file.
)