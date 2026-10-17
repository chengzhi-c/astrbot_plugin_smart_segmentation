"""Realtime stream wrapper helpers (pure boundary + async generator)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult
from astrbot.api.message_components import Plain

from ..segmentation import STRONG_BOUNDARY_MARKS, WEAK_BOUNDARY_MARKS
from ..settings import SegmentationSettings

STREAMING_WRAPPED_KEY = "smart_segmentation_streaming_wrapped"
STREAMING_SEGMENT_CHAIN_TYPE = "smart_segmentation_stream_segment"


# 硬切是唯一可能落在 emoji 字形簇中间的边界（标点/换行边界的前一字符不可能是
# 簇内连接符）。这些续字符都依附于前一个字符，边界落在它们前面会把一个 emoji
# 切成两半；国旗则是成对区域指示符。
_CLUSTER_CONTINUATION = frozenset("\u200d\ufe0f\u20e3")  # ZWJ / 变体选择符 / 组合键帽
_SKIN_TONE_FIRST, _SKIN_TONE_LAST = "\U0001F3FB", "\U0001F3FF"


def _is_regional_indicator(char: str) -> bool:
    return "\U0001F1E6" <= char <= "\U0001F1FF"


def _shift_boundary_off_grapheme(text: str, boundary: int) -> int:
    """把边界回退到 emoji 字形簇起点之前，避免一个 emoji 被切成两半。

    只回退到索引 1：boundary=0 会产出空段，让 pop_stream_segments 死循环。
    """
    while boundary > 1:
        previous = text[boundary - 1]
        following = text[boundary] if boundary < len(text) else ""
        if (
            following in _CLUSTER_CONTINUATION
            or previous == "\u200d"
            or _SKIN_TONE_FIRST <= following <= _SKIN_TONE_LAST
            or (
                _is_regional_indicator(previous)
                and _is_regional_indicator(following)
            )
        ):
            boundary -= 1
        else:
            break
    return boundary


def find_stream_boundary(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
) -> int | None:
    min_chars = max(1, min_chars)
    max_chars = max(min_chars, max_chars)

    # 只在 [min_chars-1, max_chars) 窗口内定位边界：合格边界的起点必落在窗口内，
    # 不必每次都对剩余全文扫描（大 chunk 下那是 O(N^2/min_chars)，阻塞事件循环）。
    window_end = min(max_chars, len(text))

    # 换行受 min/max 约束：不足 min_chars 会碎，超过 max_chars 等太久。第一个
    # 换行不满足时继续看后面的换行——否则"短首行 + 按行列举"这类常见回复形状
    # 会完全失去实时分段，全文憋到流结束才发出。
    # 边界取换行 run 的末尾（含换行符本身）：平台把各段 Plain 无分隔符拼接，
    # 丢边界换行会把 markdown 列表/分步说明黏成一行。
    search_from = min_chars - 1
    while (newline_at := text.find("\n", search_from, window_end)) != -1:
        # 命中位置必在某个换行 run 内（run 可能从窗口前开始）；run 末尾超过
        # max_chars 时，后续 run 的末尾只会更大，即不存在合格换行边界。
        run_end = newline_at + 1
        run_limit = min(len(text), max_chars + 1)
        while run_end < run_limit and text[run_end] == "\n":
            run_end += 1
        if run_end > max_chars:
            break
        # min_chars 按内容字符计：长换行 run 不能把原始长度垫过闸门，否则会
        # 产出只有两三个内容字符的碎消息。纯空白段仍然放行。
        head = text[:run_end].strip()
        if not head or len(head) >= min_chars:
            return run_end
        search_from = run_end

    for index in range(min_chars - 1, window_end):
        # 强句读符同样受 max_chars 约束：只查下界会让超长句子整段穿过，
        # 且分段结果随 provider 的 chunk 大小漂移（同一文本切法不同）。
        # min_chars 按内容字符计：段首空白不能把长度垫过闸门。
        if text[index] in STRONG_BOUNDARY_MARKS and len(
            text[: index + 1].strip()
        ) >= min_chars:
            return index + 1

    if len(text) >= max_chars:
        window = text[:max_chars]
        split_at = max(window.rfind(mark) for mark in WEAK_BOUNDARY_MARKS)
        # 内容长度达标时 split_at 必然 >= min_chars-1，只需判内容
        if len(text[: split_at + 1].strip()) >= min_chars:
            return split_at + 1
        return _shift_boundary_off_grapheme(text, max_chars)

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
        # 只去空格/制表符：换行符是内容（平台无分隔符拼接各段），不能剥掉
        segment = text[:boundary].strip(" \t")
        # 只剥空格/制表符：换行符是内容（平台无分隔符拼接各段），硬切位置上的
        # 换行不能丢，否则 markdown 列表被黏成一行、结果还随 chunk 大小漂移。
        text = text[boundary:].lstrip(" \t")
        if segment.strip():
            segments.append(segment)

    return segments, text


def make_stream_segment_chain(text: str) -> MessageChain:
    return MessageChain([Plain(text)], type=STREAMING_SEGMENT_CHAIN_TYPE)


def _make_flush_chain(buffer: str) -> MessageChain | None:
    """辅助：从 buffer 构造一条消息链，空 buffer 返回 None。

    只剥空格/制表符：换行符是内容（平台无分隔符拼接各段），不能剥掉。
    """
    text = str(buffer or "").strip(" \t")
    return make_stream_segment_chain(text) if text.strip() else None


async def stream_realtime_segments(
    stream: AsyncGenerator[MessageChain, None],
    settings: SegmentationSettings,
) -> AsyncGenerator[MessageChain, None]:
    buffer = ""
    try:
        async for chain in stream:
            if not isinstance(chain, MessageChain):
                continue

            # 带 type 的链（break / audio_chunk / reasoning / tool_call 等）承载业务
            # 语义（如 webchat 的音频播放、对话分段），不能拆也不能丢：先把待发
            # 文本 flush 出去，再原样透传。
            if chain.type is not None:
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
        # 先把已缓冲内容 flush 给用户，再上抛：未打补丁的核心语义就是让源流异常
        # 冒泡到平台，吞掉它会让用户拿到静默截断的回复还以为发送成功。
        logger.error("智能分段流式兼容处理异常: %s", exc, exc_info=True)
        if (flush := _make_flush_chain(buffer)):
            yield flush
        raise


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
