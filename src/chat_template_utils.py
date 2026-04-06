"""Helpers for formatting multimodal chat prompts across processor variants."""

from __future__ import annotations

from typing import Any


def _get_chat_template(owner) -> str | dict | None:
    return getattr(owner, "chat_template", None)


def _stringify_message_content(role: str, content: Any, tokenizer) -> str:
    image_token = getattr(tokenizer, "image_token", "<|image|>")
    audio_token = getattr(tokenizer, "audio_token", "<|audio|>")
    video_token = getattr(tokenizer, "video_token", "<|video|>")

    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return "" if content is None else str(content).strip()

    rendered_parts = []
    for item in content:
        if not isinstance(item, dict):
            rendered_parts.append(str(item).strip())
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "").strip()
            if text:
                rendered_parts.append(text)
        elif item_type == "image":
            rendered_parts.append(image_token)
        elif item_type == "audio":
            rendered_parts.append(audio_token)
        elif item_type == "video":
            rendered_parts.append(video_token)

    if not rendered_parts:
        return ""

    pieces = []
    for part in rendered_parts:
        if part == image_token:
            pieces.append(f"\n\n{image_token}\n\n")
        elif part == video_token:
            pieces.append(f"\n\n{video_token}\n\n")
        else:
            if pieces and not pieces[-1].endswith(("\n\n", "\n")):
                pieces.append("\n")
            pieces.append(part)
    return "".join(pieces).strip()


def _is_gemma4_tokenizer(tokenizer) -> bool:
    return all(
        getattr(tokenizer, attr, None) is not None
        for attr in ("sot_token", "eot_token", "image_token")
    )


def _render_gemma4_chat(messages, tokenizer, add_generation_prompt: bool, enable_thinking: bool = False) -> str:
    bos_token = getattr(tokenizer, "bos_token", "") or ""
    sot_token = tokenizer.sot_token
    eot_token = tokenizer.eot_token
    soc_token = getattr(tokenizer, "soc_token", None)
    eoc_token = getattr(tokenizer, "eoc_token", None)
    think_token = getattr(tokenizer, "think_token", None)

    chunks = [bos_token]
    loop_messages = list(messages)

    first_role = loop_messages[0]["role"] if loop_messages else None
    needs_system_turn = bool(enable_thinking or first_role in {"system", "developer"})
    if needs_system_turn:
        chunks.append(f"{sot_token}system\n")
        if enable_thinking and think_token:
            chunks.append(think_token)
        if first_role in {"system", "developer"}:
            chunks.append(_stringify_message_content("system", loop_messages[0].get("content"), tokenizer))
            loop_messages = loop_messages[1:]
        chunks.append(f"{eot_token}\n")

    for message in loop_messages:
        role = "model" if message.get("role") == "assistant" else message.get("role", "user")
        chunks.append(f"{sot_token}{role}\n")
        chunks.append(_stringify_message_content(role, message.get("content"), tokenizer))
        chunks.append(f"{eot_token}\n")

    if add_generation_prompt:
        chunks.append(f"{sot_token}model\n")
        if not enable_thinking and soc_token and eoc_token:
            chunks.append(f"{soc_token}thought\n{eoc_token}")

    return "".join(chunks)


def apply_chat_template_with_fallback(processor, messages, **kwargs):
    """Format messages for multimodal models, with a manual Gemma 4 fallback."""
    chat_template = _get_chat_template(processor)
    if chat_template is not None:
        return processor.apply_chat_template(messages, chat_template=chat_template, **kwargs)

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = _get_chat_template(tokenizer)
    if tokenizer_template is not None:
        return processor.apply_chat_template(messages, chat_template=tokenizer_template, **kwargs)

    if tokenizer is not None and _is_gemma4_tokenizer(tokenizer):
        add_generation_prompt = bool(kwargs.get("add_generation_prompt", False))
        enable_thinking = bool(kwargs.get("enable_thinking", False))
        return _render_gemma4_chat(
            messages,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    raise ValueError(
        "Cannot use apply_chat_template because no chat template is available and this processor "
        "does not match the built-in Gemma 4 fallback formatter."
    )
