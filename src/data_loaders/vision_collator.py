"""
Custom data collator for multimodal vision-language training with the HF backend.

Processes each sample individually using the Qwen3-VL processor, avoiding
batched image indexing issues. Images are loaded from paths embedded in the
chat messages rather than passed separately.

NOTE: HF Datasets (Arrow) pads all dict keys in a list column, so a text-only
part like {"type": "text", "text": "..."} gets stored as
{"type": "text", "text": "...", "image": None}. The Qwen3-VL chat template
generates an extra <|image_pad|> for the None image key. We strip these.
"""

import torch
from PIL import Image as PILImage


class VisionDataCollator:
    """Collator for multimodal (image + text) supervised fine-tuning.

    Processes each sample's messages individually through the processor's
    chat template, then pads and stacks the results into a batch.

    Args:
        processor: The model's AutoProcessor (e.g. Qwen3-VL processor).
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def _clean_messages(self, messages):
        """Clean up messages from HF Dataset serialization artifacts.

        HF Datasets (Arrow) forces all dicts in a list column to have the
        same keys. So {"type": "text", "text": "..."} becomes
        {"type": "text", "text": "...", "image": None}. The Qwen3-VL chat
        template treats this None as an actual image reference and generates
        a spurious <|image_pad|> token. We strip None-valued keys.
        """
        images = []
        clean_messages = []
        for msg in messages:
            content = msg.get("content")
            clean_msg = {
                key: value
                for key, value in msg.items()
                if key != "content" and value is not None
            }
            if isinstance(content, list):
                clean_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        clean_parts.append(part)
                        continue

                    # Strip None-valued keys (HF Arrow padding artifact)
                    clean_part = {k: v for k, v in part.items() if v is not None}

                    if clean_part.get("type") == "image":
                        img_ref = clean_part.get("image")
                        if isinstance(img_ref, str):
                            images.append(PILImage.open(img_ref).convert("RGB"))
                        elif hasattr(img_ref, "convert"):
                            images.append(img_ref.convert("RGB"))
                        # Replace with a clean marker (no path)
                        clean_parts.append({"type": "image"})
                    else:
                        clean_parts.append(clean_part)
                clean_msg["content"] = clean_parts
                clean_messages.append(clean_msg)
            else:
                clean_msg["content"] = content
                clean_messages.append(clean_msg)
        return clean_messages, images

    def _process_single(self, example):
        """Process a single example: load images, apply chat template, tokenize."""
        clean_messages, images = self._clean_messages(example["messages"])

        # Apply chat template
        text = self.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=False,
        )

        # Process: tokenize text + encode images
        processed = self.processor(
            text=[text],
            images=images if images else None,
            padding=False,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return processed

    def __call__(self, examples):
        # Process each example individually to avoid batched image indexing bugs
        batch_items = [self._process_single(ex) for ex in examples]

        # Find max sequence length in this batch for padding
        max_len = max(item["input_ids"].shape[1] for item in batch_items)
        pad_id = self.processor.tokenizer.pad_token_id or 0

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        has_vision = any("pixel_values" in item for item in batch_items)

        for item in batch_items:
            seq_len = item["input_ids"].shape[1]
            pad_len = max_len - seq_len

            # Pad input_ids (right-padding)
            input_ids = item["input_ids"].squeeze(0)
            if pad_len > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_id, dtype=input_ids.dtype),
                ])
            all_input_ids.append(input_ids)

            # Pad attention_mask
            attn_mask = item["attention_mask"].squeeze(0)
            if pad_len > 0:
                attn_mask = torch.cat([
                    attn_mask,
                    torch.zeros(pad_len, dtype=attn_mask.dtype),
                ])
            all_attention_mask.append(attn_mask)

            # Labels = input_ids with padding masked to -100
            labels = input_ids.clone()
            if pad_len > 0:
                labels[-pad_len:] = -100
            all_labels.append(labels)

            # Collect vision tensors
            if has_vision and "pixel_values" in item:
                all_pixel_values.append(item["pixel_values"].squeeze(0))
                if "image_grid_thw" in item:
                    # Don't squeeze — keep [1, 3] so torch.cat gives [N, 3]
                    all_image_grid_thw.append(item["image_grid_thw"])

        batch = {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "labels": torch.stack(all_labels),
        }

        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return batch
