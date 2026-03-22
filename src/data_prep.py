"""
Dataset preparation script for multimodal fine-tuning datasets.

Supports two annotation sources and writes a JSONL file in the multimodal chat
format expected by SFTTrainer.

Usage:
    # CSV annotations (existing sandbox/grid-001 flow):
    python src/data_prep.py

    python src/data_prep.py \
        --source_type csv \
        --csv_path dataset/grid-001/annotations.csv \
        --image_dir dataset/grid-001/images \
        --output dataset/grid-001/train_dataset.jsonl \
        --prompt_name grid_test

    # Sidecar JSON annotations stored next to images (grid-002 flow):
    python src/data_prep.py \
        --source_type sidecar_json \
        --dataset_dir dataset/grid-002 \
        --output dataset/grid-002/train_dataset.jsonl

Supported CSV schemas:
    image_filename,chaos_tidy_score,monotony_variety_score[,reasoning|explanation]
    my_image.png,3,2,...

    image_filename,coordinates
    my_image.png,"(2, 4)"

Supported sidecar JSON schema:
    {
      "system_prompt": "...",
      "user_prompt": "...",
      "index": [0, 0],
      "boundary": [278, 221, ...]
    }
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from prompts import PROMPTS


SIDECAR_PROMPT_KEYS = {"system_prompt", "user_prompt"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a multimodal fine-tuning JSONL from CSV or sidecar JSON annotations.",
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="auto",
        choices=["auto", "csv", "sidecar_json"],
        help=(
            "Annotation source type. 'auto' uses sidecar_json when --dataset_dir is set, "
            "otherwise falls back to csv."
        ),
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
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing image files and sidecar JSON annotations.",
    )
    parser.add_argument(
        "--annotation_ext",
        type=str,
        default=".json",
        help="Sidecar annotation file extension for sidecar_json mode.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".png",
        help="Image file extension for sidecar_json mode.",
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
    parser.add_argument(
        "--sidecar_exclude_keys",
        nargs="*",
        default=["boundary"],
        help=(
            "Keys to omit from the assistant response payload when reading sidecar JSON files. "
            "Defaults to excluding 'boundary' because it is already present in the user prompt."
        ),
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
    user_prompt: str,
    system_prompt: str | None = None,
    reasoning_content: str | None = None,
) -> dict:
    """Build a single JSONL sample in the Qwen3-VL multimodal chat format.

    The format mirrors the existing EmoArt data loader:
      messages = [
          {role: system,    content: [{type: text, ...}]},
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

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image_abs_path},
            ],
        }
    )
    messages.append(assistant_message)

    return {"messages": messages}


def score_range_for_prompt(prompt_name: str) -> tuple[int, int]:
    if prompt_name == "expert_reasoning":
        return 1, 3
    return 1, 5


def resolve_source_type(args) -> str:
    """Resolve which annotation source should be used."""
    if args.source_type != "auto":
        return args.source_type
    if args.dataset_dir:
        return "sidecar_json"
    return "csv"


def resolve_reasoning_column(fieldnames: list[str] | None) -> str | None:
    """Support both reasoning column names used in older CSV datasets."""
    if not fieldnames:
        return None
    if "reasoning" in fieldnames:
        return "reasoning"
    if "explanation" in fieldnames:
        return "explanation"
    return None


def normalize_extension(ext: str) -> str:
    """Ensure file extensions consistently begin with a dot."""
    return ext if ext.startswith(".") else f".{ext}"


def extract_sidecar_response_payload(
    annotation_data: dict,
    exclude_keys: set[str],
) -> dict:
    """Extract assistant response fields from a sidecar annotation file."""
    return {
        key: value
        for key, value in annotation_data.items()
        if key not in SIDECAR_PROMPT_KEYS and key not in exclude_keys
    }


def validate_index_payload(value, sample_name: str) -> bool:
    """Validate the common grid-index payload used by the sidecar dataset."""
    if value is None:
        return True
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) for v in value):
        print(
            f"  ⚠  Sample {sample_name}: 'index' must be a list of two integers, "
            f"got {value!r}"
        )
        return False
    return True


def prepare_csv_dataset(args) -> tuple[int, int]:
    """Build JSONL samples from a CSV annotation file."""
    csv_path = args.csv_path
    image_dir = os.path.abspath(args.image_dir)
    output_path = args.output
    prompt_string = PROMPTS[args.prompt_name]
    min_score, max_score = score_range_for_prompt(args.prompt_name)

    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Psycho LLM — Dataset Preparation")
    print("  Source:    csv")
    print(f"  CSV:       {csv_path}")
    print(f"  Images:    {image_dir}")
    print(f"  Output:    {output_path}")
    print("=" * 60)

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

        reasoning_col = resolve_reasoning_column(reader.fieldnames)

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
                image_abs_path=os.path.abspath(image_path),
                response_payload=response_payload,
                user_prompt=prompt_string,
                reasoning_content=reasoning if schema == "coordinates" else None,
            )
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


def prepare_sidecar_json_dataset(args) -> tuple[int, int]:
    """Build JSONL samples from per-image sidecar JSON annotations."""
    dataset_dir = Path(args.dataset_dir or args.image_dir).resolve()
    output_path = args.output
    annotation_ext = normalize_extension(args.annotation_ext)
    image_ext = normalize_extension(args.image_ext)
    exclude_keys = set(args.sidecar_exclude_keys or [])

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    annotation_paths = sorted(dataset_dir.glob(f"*{annotation_ext}"))
    if not annotation_paths:
        print(
            f"ERROR: No sidecar annotation files with extension {annotation_ext!r} "
            f"found in {dataset_dir}"
        )
        sys.exit(1)

    print("=" * 60)
    print("  Psycho LLM — Dataset Preparation")
    print("  Source:    sidecar_json")
    print(f"  Dataset:   {dataset_dir}")
    print(f"  Output:    {output_path}")
    print(f"  Sidecars:  {len(annotation_paths)} files")
    print("=" * 60)

    written = 0
    skipped = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for annotation_path in annotation_paths:
            sample_name = annotation_path.stem
            image_path = annotation_path.with_suffix(image_ext)

            if not image_path.is_file():
                print(f"  ⚠  Sample {sample_name}: image not found: {image_path}")
                skipped += 1
                continue

            try:
                with open(annotation_path, "r", encoding="utf-8") as annotation_file:
                    annotation_data = json.load(annotation_file)
            except json.JSONDecodeError as exc:
                print(f"  ⚠  Sample {sample_name}: invalid JSON ({exc})")
                skipped += 1
                continue

            if not isinstance(annotation_data, dict):
                print(f"  ⚠  Sample {sample_name}: sidecar JSON must be an object")
                skipped += 1
                continue

            system_prompt = annotation_data.get("system_prompt")
            user_prompt = annotation_data.get("user_prompt")
            response_payload = extract_sidecar_response_payload(annotation_data, exclude_keys)

            if not isinstance(system_prompt, str) or not system_prompt.strip():
                print(f"  ⚠  Sample {sample_name}: missing or empty 'system_prompt'")
                skipped += 1
                continue
            if not isinstance(user_prompt, str) or not user_prompt.strip():
                print(f"  ⚠  Sample {sample_name}: missing or empty 'user_prompt'")
                skipped += 1
                continue
            if not response_payload:
                print(f"  ⚠  Sample {sample_name}: no assistant response fields found")
                skipped += 1
                continue
            if not validate_index_payload(response_payload.get("index"), sample_name):
                skipped += 1
                continue

            sample = build_sample(
                image_abs_path=str(image_path.resolve()),
                response_payload=response_payload,
                user_prompt=user_prompt.strip(),
                system_prompt=system_prompt.strip(),
            )
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


def print_summary(output_path: str, written: int, skipped: int):
    """Print a shared success summary after dataset generation."""
    total = written + skipped
    print(f"\nDone! {written}/{total} samples written to {output_path}")
    if skipped:
        print("  Skipped samples due to missing files or invalid annotations.")


def main():
    args = parse_args()
    source_type = resolve_source_type(args)

    if source_type == "csv":
        written, skipped = prepare_csv_dataset(args)
    else:
        written, skipped = prepare_sidecar_json_dataset(args)

    print_summary(args.output, written, skipped)


if __name__ == "__main__":
    main()
