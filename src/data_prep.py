"""
Dataset preparation script for sandbox image fine-tuning.

Reads a CSV of image filenames + scores and produces a JSONL file in the
Qwen3-VL multimodal chat format expected by SFTTrainer.

Usage:
    # Using defaults (dataset/sandbox-001/):
    python src/data_prep.py

    # Custom paths:
    python src/data_prep.py \
        --csv_path   dataset/sandbox-001/annotations.csv \
        --image_dir  dataset/sandbox-001/images \
        --output     dataset/sandbox-001/train_dataset.jsonl

CSV schema:
    image_filename,chaos_tidy_score,monotony_variety_score
    my_image.png,3,4
"""

import argparse
import csv
import json
import os
import sys


# ── Prompt template (generalized for diverse sandbox scenes) ───────────
PROMPT = (
    "Analyze this isometric sandbox image. "
    "Rate it on two dimensions from 1 to 5. "
    "Dimension 1: Chaos to Tidy (1=chaotic/fragmented, 5=tidy/smooth). "
    "Dimension 2: Monotony to Variety (1=monotonous/simple, 5=variety/complicated). "
    "Output strict JSON with keys 'chaos_tidy_score' and 'monotony_variety_score'."
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Qwen3-VL fine-tuning JSONL from sandbox images + CSV scores.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/sandbox-001/annotations.csv",
        help="Path to the annotations CSV file.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="dataset/sandbox-001/images",
        help="Directory containing the raw sandbox images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/sandbox-001/train_dataset.jsonl",
        help="Path where the output JSONL will be written.",
    )
    return parser.parse_args()


def validate_score(value: str, column_name: str, row_num: int) -> int | None:
    """Validate that a score string is an integer in [1, 5].

    Returns the integer on success, or None (with a warning) on failure.
    """
    try:
        score = int(value)
    except (ValueError, TypeError):
        print(f"  ⚠  Row {row_num}: '{column_name}' is not an integer: {value!r}")
        return None

    if not 1 <= score <= 5:
        print(f"  ⚠  Row {row_num}: '{column_name}' out of range [1-5]: {score}")
        return None

    return score


def build_sample(image_abs_path: str, chaos_tidy: int, monotony_variety: int) -> dict:
    """Build a single JSONL sample in the Qwen3-VL multimodal chat format.

    The format mirrors the existing EmoArt data loader:
      messages = [
          {role: user,      content: [{type: text, ...}, {type: image, ...}]},
          {role: assistant,  content: [{type: text, ...}]},
      ]
    """
    response_json = json.dumps(
        {
            "chaos_tidy_score": chaos_tidy,
            "monotony_variety_score": monotony_variety,
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image", "image": image_abs_path},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_json}],
            },
        ],
    }


def main():
    args = parse_args()

    csv_path = args.csv_path
    image_dir = os.path.abspath(args.image_dir)
    output_path = args.output

    # ── Validate inputs ────────────────────────────────────────────────
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Psycho LLM — Sandbox Dataset Preparation")
    print(f"  CSV:       {csv_path}")
    print(f"  Images:    {image_dir}")
    print(f"  Output:    {output_path}")
    print("=" * 60)

    # ── Read CSV and build JSONL ───────────────────────────────────────
    written = 0
    skipped = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as csv_file, \
         open(output_path, "w", encoding="utf-8") as out_file:

        reader = csv.DictReader(csv_file)

        # Verify expected columns exist
        required_cols = {"image_filename", "chaos_tidy_score", "monotony_variety_score"}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            missing = required_cols - set(reader.fieldnames or [])
            print(f"ERROR: CSV is missing required columns: {missing}")
            sys.exit(1)

        for row_num, row in enumerate(reader, start=2):  # row 1 = header
            filename = row["image_filename"].strip()

            # Check image exists
            image_path = os.path.join(image_dir, filename)
            if not os.path.isfile(image_path):
                print(f"  ⚠  Row {row_num}: Image not found: {image_path}")
                skipped += 1
                continue

            # Validate scores
            chaos_tidy = validate_score(row["chaos_tidy_score"], "chaos_tidy_score", row_num)
            monotony_variety = validate_score(
                row["monotony_variety_score"], "monotony_variety_score", row_num,
            )

            if chaos_tidy is None or monotony_variety is None:
                skipped += 1
                continue

            # Build and write sample
            sample = build_sample(
                image_abs_path=image_path,
                chaos_tidy=chaos_tidy,
                monotony_variety=monotony_variety,
            )
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    # ── Summary ────────────────────────────────────────────────────────
    total = written + skipped
    print(f"\nDone! {written}/{total} samples written to {output_path}")
    if skipped:
        print(f"  ({skipped} rows skipped due to missing images or invalid scores)")


if __name__ == "__main__":
    main()
