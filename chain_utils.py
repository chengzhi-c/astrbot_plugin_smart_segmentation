"""Pure helpers for AstrBot message chains / results."""

from __future__ import annotations

from typing import Any

from astrbot.api.event import MessageChain, MessageEventResult, ResultContentType
from astrbot.api.message_components import At, Plain

from .segmentation import is_action_only_text, strip_thinking_content
from .settings import SegmentationSettings


def is_streaming_result(result: Any) -> bool:
    return bool(
        result is not None
        and getattr(result, "result_content_type", None) == ResultContentType.STREAMING_RESULT
        and getattr(result, "async_stream", None) is not None
    )


def _is_plain_chain(message_chain: MessageChain) -> bool:
    chain = getattr(message_chain, "chain", None)
    return (
        isinstance(chain, list)
        and any(isinstance(component, Plain) for component in chain)
        and all(isinstance(component, (At, Plain)) for component in chain)
    )


def extract_plain_text_chain(message_chain: MessageChain) -> str:
    if not _is_plain_chain(message_chain):
        return ""
    texts = [component.text for component in message_chain.chain if isinstance(component, Plain)]
    return strip_thinking_content("".join(texts))


def replace_plain_text_chain(message_chain: MessageChain, text: str) -> bool:
    if not _is_plain_chain(message_chain):
        return False

    # 只保留首个 Plain 并写入首段：第 2+ 个 Plain 的文本已包含在 extract 时
    # join 的全文里（分段后由补发段送达），此处丢弃不丢内容。非 Plain 组件
    # 保持原有相对顺序。
    replaced = False
    next_chain = []
    for component in message_chain.chain:
        if isinstance(component, Plain):
            if not replaced:
                next_chain.append(Plain(text))
                replaced = True
            continue
        next_chain.append(component)

    message_chain.chain = next_chain
    return True


def is_model_text_result(result: MessageEventResult) -> bool:
    # is_model_result 自 astrbot 4.20 才有；缺失时回退 is_llm_result（各版本均有），
    # 否则旧版本上任何插件的纯文本结果都会被误当成模型回复拿去分段。
    checker = getattr(result, "is_model_result", None) or getattr(
        result,
        "is_llm_result",
        None,
    )
    if callable(checker):
        try:
            if not checker():
                return False
        except Exception:
            return False
    return _is_plain_chain(result)


def should_segment_text(text: str, settings: SegmentationSettings) -> bool:
    if not text:
        return False
    if len(text) < settings.min_length:
        return False
    return not is_action_only_text(text)
