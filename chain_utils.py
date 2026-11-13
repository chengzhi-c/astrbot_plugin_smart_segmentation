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
    """只接受「若干 At 前缀 + 末尾 Plain（可多条）」。

    At 落在任何 Plain 之后时不分段：replace_plain_text_chain 把正文写进第一条
    Plain、其余 Plain 交给补发，At 就落到分段后第一条消息的末尾、后面没有正文——
    读 at.name 的协议端渲染成裸 @。宁可不分段，也不搬动链里的组件。
    """
    chain = getattr(message_chain, "chain", None)
    if not isinstance(chain, list) or not chain:
        return False
    if not isinstance(chain[-1], Plain):
        return False
    seen_plain = False
    for component in chain:
        if isinstance(component, Plain):
            seen_plain = True
        elif not isinstance(component, At) or seen_plain:
            return False
    # chain[-1] 已保证至少有一个 Plain
    return True


def extract_plain_text_chain(message_chain: MessageChain) -> str:
    if not _is_plain_chain(message_chain):
        return ""
    texts = [component.text for component in message_chain.chain if isinstance(component, Plain)]
    return strip_thinking_content("".join(texts))


def replace_plain_text_chain(message_chain: MessageChain, text: str) -> bool:
    """首段写进第一条 Plain；其余 Plain 丢弃（它们的文本已由补发段送达）。"""
    if not _is_plain_chain(message_chain):
        return False
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
