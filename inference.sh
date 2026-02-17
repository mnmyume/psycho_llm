#!/bin/bash

LLAMA_PATH="./llama.cpp"
# Point this to the F16 folder we just created
BASE_MODEL_DIR="model_f16_for_conversion"
MMPROJ="unsloth/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf"
LORA_DIR="lora_model/qwen3_vl_8b_emoart_130k_v1"
LORA_GGUF="lora_model/qwen3_vl_8b_emoart_130k_v1_adapter.gguf"

# 1. Convert LoRA to GGUF using the F16 base for metadata
if [ ! -f "$LORA_GGUF" ]; then
    echo "Converting LoRA adapter to GGUF..."
    python3 $LLAMA_PATH/convert_lora_to_gguf.py $LORA_DIR \
        --base $BASE_MODEL_DIR \
        --outfile $LORA_GGUF
fi

# 2. Run Inference (Using your existing GGUF base model for speed)
# Note: Removed --color to fix the 'invalid argument' error
$LLAMA_PATH/llama-mtmd-cli \
    -m "unsloth/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf" \
    --mmproj "$MMPROJ" \
    --lora "$LORA_GGUF" \
    --n-gpu-layers 99 \
    --flash-attn on \
    --jinja \
    --temp 1.5 \
    --min-p 0.1 \
    --ctx-size 8192 \
    -p "Describe the artistic style and emotional content of this image. Your answer should use JSON format."