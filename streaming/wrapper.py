"""Realtime stream wrapper helpers (pure boundary + async generator)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult
from astrbot.api.message_components import Plain

from ..segmentation import STRONG_BOUNDARY_MARKS, WEAK_BOUNDARY_MARKS
from ..settings import SegmentationSettings

STREAMING_WRAPPED_KEY = "smart_segmentation_streaming_wrapped"
STREAMING_SEGMENT_CHAIN_TYPE = "smart_segmentation_stream_segment"


def find_stream_boundary(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
) -> int | None:
    newline_match = re.search(r"\n+", text)
    if newline_match is not None:
        if newline_match.start() > 0:
            return newline_match.start()
        return newline_match.end()

    min_chars = max(1, min_chars)
    for index, char in enumerate(text):
        if char in STRONG_BOUNDARY_MARKS and index + 1 >= min_chars:
            return index + 1

    max_chars = max(min_chars, max_chars)
    if len(text) >= max_chars:
        window = text[:max_chars]
        split_at = max(window.rfind(mark) for mark in WEAK_BOUNDARY_MARKS)
        if split_at >= min_chars:
            return split_at + 1
        return max_chars

    return None


def pop_stream_segments(
    buffer: str,
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[list[str], str]:
    text = str(buffer or "").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []

    while text:
        boundary = find_stream_boundary(
            text,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if boundary is None:
            break
        segment = text[:boundary].strip()
        text = text[boundary:].lstrip()
        if segment:
            segments.append(segment)

    return segments, text


def make_stream_segment_chain(text: str) -> MessageChain:
    return MessageChain([Plain(text)], type=STREAMING_SEGMENT_CHAIN_TYPE)


def _make_flush_chain(buffer: str) -> MessageChain | None:
    """辅助：从 buffer 构造一条消息链，空 buffer 返回 None。"""
    text = str(buffer or "").strip()
    return make_stream_segment_chain(text) if text else None


async def stream_realtime_segments(
    stream: AsyncGenerator[MessageChain, None],
    settings: SegmentationSettings,
) -> AsyncGenerator[MessageChain, None]:
    buffer = ""
    try:
        async for chain in stream:
            if not isinstance(chain, MessageChain):
                continue

            if chain.type == "break":
                if (flush := _make_flush_chain(buffer)):
                    yield flush
                buffer = ""
                yield chain
                continue

            # Preserve original component order: Plain merges into the pending
            # buffer (flushing when a boundary is reached), non-Plain flushes
            # the buffer first and passes through.
            for component in chain.chain:
                if isinstance(component, Plain):
                    buffer += component.text
                    segments, buffer = pop_stream_segments(
                        buffer,
                        min_chars=settings.streaming_min_chars,
                        max_chars=settings.streaming_max_chars,
                    )
                    for segment in segments:
                        yield make_stream_segment_chain(segment)
                else:
                    if (flush := _make_flush_chain(buffer)):
                        yield flush
                        buffer = ""
                    yield MessageChain([component])

        if (flush := _make_flush_chain(buffer)):
            yield flush
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("智能分段流式兼容处理异常: %s", exc, exc_info=True)
        if (flush := _make_flush_chain(buffer)):
            yield flush


def decorate_streaming_result(
    event: AstrMessageEvent,
    result: MessageEventResult,
    settings: SegmentationSettings,
    *,
    is_guarded: bool,
) -> None:
    if not settings.streaming_compat_enabled:
        return
    if is_guarded:
        return
    if event.get_extra(STREAMING_WRAPPED_KEY, False):
        return

    stream = getattr(result, "async_stream", None)
    if stream is None:
        return

    result.async_stream = stream_realtime_segments(stream, settings)
    event.set_extra(STREAMING_WRAPPED_KEY, True)
    logger.info("智能分段已接管流式输出，将按实时断句分段发送")
