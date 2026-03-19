"""
Dataset preparation script for image fine-tuning datasets.

Reads a CSV of image filenames + annotations and produces a JSONL file in the
Qwen3-VL multimodal chat format expected by SFTTrainer.

Usage:
    # Using defaults (dataset/sandbox-001/):
    python src/data_prep.py

    # Custom paths:
    python src/data_prep.py \
        --csv_path   dataset/grid-001/annotations.csv \
        --image_dir  dataset/grid-001/images \
        --output     dataset/grid-001/train_dataset.jsonl \
        --prompt_name grid_test

Supported CSV schemas:
    image_filename,chaos_tidy_score,monotony_variety_score[,reasoning|explanation]
    my_image.png,3,2,...

    image_filename,coordinates
    my_image.png,"(2, 4)"
"""

import argparse
import csv
import json
import os
import sys

from prompts import PROMPTS


# Removed inline PROMPT string mappings since they are now extracted to src/prompts.py

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
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="expert_reasoning",
        choices=list(PROMPTS.keys()),
        help="Name of the prompt template to use from src/prompts.py.",
    )
    return parser.parse_args()


def validate_score(
    value: str,
    column_name: str,
    row_num: int,
    min_score: int,
    max_score: int,
) -> int | None:
    """Validate that a score string is an integer in [min_score, max_score].

    Returns the integer on success, or None (with a warning) on failure.
    """
    try:
        score = int(value)
    except (ValueError, TypeError):
        print(f"  ⚠  Row {row_num}: '{column_name}' is not an integer: {value!r}")
        return None

    if not min_score <= score <= max_score:
        print(
            f"  ⚠  Row {row_num}: '{column_name}' out of range "
            f"[{min_score}-{max_score}]: {score}"
        )
        return None

    return score


def validate_coordinates(value: str, row_num: int) -> list[int] | None:
    """Validate coordinate strings in the form '(x,y)' or '(x, y)'."""
    if value is None:
        print(f"  ⚠  Row {row_num}: 'coordinates' is missing")
        return None

    text = value.strip()
    if not (text.startswith("(") and text.endswith(")")):
        print(f"  ⚠  Row {row_num}: invalid coordinates format: {value!r}")
        return None

    parts = [part.strip() for part in text[1:-1].split(",")]
    if len(parts) != 2:
        print(f"  ⚠  Row {row_num}: invalid coordinates format: {value!r}")
        return None

    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        print(f"  ⚠  Row {row_num}: coordinates must be integers: {value!r}")
        return None

    return [x, y]


def detect_annotation_schema(fieldnames: list[str] | None) -> str | None:
    """Detect which annotation schema the CSV uses."""
    if not fieldnames:
        return None

    fields = set(fieldnames)
    if {"image_filename", "chaos_tidy_score", "monotony_variety_score"}.issubset(fields):
        return "scores"
    if {"image_filename", "coordinates"}.issubset(fields):
        return "coordinates"
    return None


def build_coordinate_reasoning(coordinates: list[int]) -> str:
    """Generate a short reasoning trace for grid-coordinate supervision."""
    x, y = coordinates
    return (
        f"Count from [0, 0] along the right-edge diagonal until x = {x}, "
        f"then count along the left-edge diagonal until y = {y}."
    )


def build_sample(
    image_abs_path: str,
    response_payload: dict,
    prompt_string: str,
    reasoning_content: str | None = None,
) -> dict:
    """Build a single JSONL sample in the Qwen3-VL multimodal chat format.

    The format mirrors the existing EmoArt data loader:
      messages = [
          {role: user,      content: [{type: text, ...}, {type: image, ...}]},
          {role: assistant,  content: [{type: text, ...}]},
      ]
    """
    response_json = json.dumps(
        response_payload,
        ensure_ascii=False,
    )

    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": response_json}],
    }
    if reasoning_content:
        assistant_message["reasoning_content"] = reasoning_content

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_string},
                    {"type": "image", "image": image_abs_path},
                ],
            },
            assistant_message,
        ],
    }


def score_range_for_prompt(prompt_name: str) -> tuple[int, int]:
    if prompt_name == "expert_reasoning":
        return 1, 3
    return 1, 5


def main():
    args = parse_args()

    csv_path = args.csv_path
    image_dir = os.path.abspath(args.image_dir)
    output_path = args.output
    prompt_string = PROMPTS[args.prompt_name]
    min_score, max_score = score_range_for_prompt(args.prompt_name)

    # ── Validate inputs ────────────────────────────────────────────────
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Psycho LLM — Dataset Preparation")
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

        schema = detect_annotation_schema(reader.fieldnames)
        if schema is None:
            print(
                "ERROR: CSV schema not recognized. Expected either "
                "{image_filename, chaos_tidy_score, monotony_variety_score} "
                "or {image_filename, coordinates}."
            )
            sys.exit(1)

        print(f"  Schema:    {schema}")
        if schema == "scores":
            print(f"  Score range: [{min_score}-{max_score}]")

        # Support both "reasoning" and "explanation" as optional columns
        reasoning_col = None
        if "reasoning" in reader.fieldnames:
            reasoning_col = "reasoning"
        elif "explanation" in reader.fieldnames:
            reasoning_col = "explanation"

        for row_num, row in enumerate(reader, start=2):  # row 1 = header
            filename = row["image_filename"].strip()

            # Check image exists
            image_path = os.path.join(image_dir, filename)
            if not os.path.isfile(image_path):
                print(f"  ⚠  Row {row_num}: Image not found: {image_path}")
                skipped += 1
                continue

            if schema == "scores":
                chaos_tidy = validate_score(
                    row["chaos_tidy_score"],
                    "chaos_tidy_score",
                    row_num,
                    min_score,
                    max_score,
                )
                monotony_variety = validate_score(
                    row["monotony_variety_score"],
                    "monotony_variety_score",
                    row_num,
                    min_score,
                    max_score,
                )

                if chaos_tidy is None or monotony_variety is None:
                    skipped += 1
                    continue

                response_payload = {
                    "chaos_tidy_score": chaos_tidy,
                    "monotony_variety_score": monotony_variety,
                }
                reasoning = row.get(reasoning_col, "").strip() if reasoning_col else None
                if reasoning:
                    response_payload["reasoning"] = reasoning
            else:
                coordinates = validate_coordinates(row["coordinates"], row_num)
                if coordinates is None:
                    skipped += 1
                    continue

                response_payload = {"coordinates": coordinates}
                reasoning = row.get(reasoning_col, "").strip() if reasoning_col else ""
                if not reasoning:
                    reasoning = build_coordinate_reasoning(coordinates)

            # Build and write sample
            sample = build_sample(
                image_abs_path=image_path,
                response_payload=response_payload,
                prompt_string=prompt_string,
                reasoning_content=reasoning if schema == "coordinates" else None,
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
