# Psycho LLM

Multi-modal LLM psychological analysis of sandbox drawings and artwork. Uses **Qwen3-VL** (8B / 32B) fine-tuned with **LoRA** on emotion-annotated art datasets.

## Project Structure

```
psycho_llm/
├── src/
│   ├── config.py              # YAML-driven configuration system
│   ├── train.py               # LoRA fine-tuning pipeline
│   ├── inference.py           # CLI inference tool
│   ├── app.py                 # Gradio web UI
│   ├── download.py            # Model downloader
│   ├── constants/paths.py     # Project path constants
│   ├── data_loaders/
│   │   ├── base_dataset.py    # Abstract dataset base class
│   │   └── emoart.py          # EmoArt dataset loader
│   └── models/
│       └── model_utils.py     # Model loading, LoRA, and generation utilities
├── recipes/                   # YAML training recipes
│   ├── Qwen3-VL-8B.yaml
│   └── Qwen3-VL-32B.yaml
├── dataset/                   # Dataset files (gitignored)
├── lora_model/                # Saved LoRA adapters (gitignored)
├── outputs/                   # Training checkpoints (gitignored)
├── train.sh                   # SLURM training script
├── inference.sh               # GGUF/llama.cpp inference script
└── requirements.txt
```

## Setup

```bash
conda create -n psyc python=3.11
conda activate psyc
pip install -r requirements.txt
```

## Usage

### 1. Training (LoRA Fine-Tuning)

```bash
# Using a YAML recipe
python src/train.py --config recipes/Qwen3-VL-8B.yaml --run_name my_exp_v1

# With CLI overrides
python src/train.py --config recipes/Qwen3-VL-8B.yaml --run_name my_exp_v2 --num_epochs 3 --learning_rate 1e-4

# Via SLURM
sbatch train.sh
```

The trained LoRA adapter weights are saved to `lora_model/{run_name}/`. The base model is never modified.

### 2. CLI Inference

```bash
python src/inference.py \
    --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 \
    --image path/to/sandbox_drawing.png \
    --prompt "Analyze the psychological themes in this sandbox image."
```

### 3. Gradio Web UI

```bash
python src/app.py --model_path lora_model/qwen3_vl_8b_emoart_5k_v1

# With a shareable public link
python src/app.py --model_path lora_model/qwen3_vl_8b_emoart_5k_v1 --share
```

## Configuration

Training configs are defined as YAML files in `recipes/`. See `recipes/Qwen3-VL-8B.yaml` for all available options. Any config value can be overridden via CLI:

```bash
python src/train.py --config recipes/Qwen3-VL-8B.yaml --lora_r 32 --batch_size 4
```
