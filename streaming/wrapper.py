"""Realtime stream wrapper helpers (pure boundary + async generator)."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult
from astrbot.api.message_components import At, Plain, Reply

from ..bounds import PLUGIN_NAME
from ..errors import brief_exception
from ..segmentation import (
    _FENCE_RE,
    STRONG_BOUNDARY_MARKS,
    WEAK_BOUNDARY_MARKS,
    _ThinkingFilter,
    clean_segment_tail,
    has_visible_text,
    mask_uncuttable_spans,
)
from ..settings import SegmentationSettings

STREAMING_WRAPPED_KEY = "smart_segmentation_streaming_wrapped"
STREAMING_SEGMENT_CHAIN_TYPE = "smart_segmentation_stream_segment"

# At / Reply 自身没有正文：单独成条就是一条只有 @ 或只有引用、没有内容的消息。
_DECORATION_COMPONENTS = (At, Reply)


# 硬切是唯一可能落在 emoji 字形簇中间的边界（标点/换行边界的前一字符不可能是
# 簇内连接符）。这些续字符都依附于前一个字符，边界落在它们前面会把一个 emoji
# 切成两半；国旗则是成对区域指示符。
_CLUSTER_CONTINUATION = frozenset("\u200d\ufe0f\u20e3")  # ZWJ / 变体选择符 / 组合键帽
_SKIN_TONE_FIRST, _SKIN_TONE_LAST = "\U0001f3fb", "\U0001f3ff"


def _is_regional_indicator(char: str) -> bool:
    return "\U0001f1e6" <= char <= "\U0001f1ff"


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
            or (_is_regional_indicator(previous) and _is_regional_indicator(following))
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
    # 多看一个围栏标记，避免 max_chars 恰落在闭围栏里时误判；不扫描全文。
    scan_end = min(len(text), max_chars + len("```"))
    head = text[:scan_end]
    # 等长遮罩保留原文索引；未闭合单元也遮到窗口末尾。规则与本地分段同源，
    # 否则同一段文本在两条路径上会切出不同边界。
    spans = mask_uncuttable_spans(head, open_tails=True)
    # 行首的 ~ 可能还在组成围栏，收齐前不能当作句尾波浪号发送。
    pending_fence = re.search(r"(?:^|\n) {0,3}(~{1,2})$", head)
    if pending_fence:
        spans.append(pending_fence.span(1))
    masked = list(head)
    for start, end in spans:
        masked[start:end] = '"' * (end - start)
    window = "".join(masked[:window_end])

    # 换行受 min/max 约束：不足 min_chars 会碎，超过 max_chars 等太久。第一个
    # 换行不满足时继续看后面的换行——否则"短首行 + 按行列举"这类常见回复形状
    # 会完全失去实时分段，全文憋到流结束才发出。
    # 边界取换行 run 的末尾（含换行符本身）：平台把各段 Plain 无分隔符拼接，
    # 丢边界换行会把 markdown 列表/分步说明黏成一行。
    search_from = min_chars - 1
    while (newline_at := window.find("\n", search_from, window_end)) != -1:
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
        line_head = text[:run_end].strip()
        if not line_head or len(line_head) >= min_chars:
            return run_end
        search_from = run_end

    for index in range(min_chars - 1, window_end):
        # 强句读符同样受 max_chars 约束：只查下界会让超长句子整段穿过，
        # 且分段结果随 provider 的 chunk 大小漂移（同一文本切法不同）。
        # min_chars 按内容字符计：段首空白不能把长度垫过闸门。
        if (
            window[index] in STRONG_BOUNDARY_MARKS
            and len(text[: index + 1].strip()) >= min_chars
        ):
            return index + 1

    if len(text) >= max_chars:
        split_at = max(window.rfind(mark) for mark in WEAK_BOUNDARY_MARKS)
        # 内容长度达标时 split_at 必然 >= min_chars-1，只需判内容
        if len(text[: split_at + 1].strip()) >= min_chars:
            return split_at + 1
        for start, end in spans:
            if start < max_chars <= end and len(text[:start].strip()) >= min_chars:
                return start
        if pending_fence and len(text) == len(head):
            return None
        # shortcut: 单元本身超长或其前缀不足 min 时仍在 max 硬切；宁可降级渲染，
        # 也不为等待 URL 结束/闭围栏而无限缓冲，且边界始终为正，保证 pop 前进。
        shifted = _shift_boundary_off_grapheme(text, max_chars)
        # 回退会一路退到 1（连续国旗 emoji 这种病态输入）：退到不足 min_chars 时
        # 宁可硬切在 max_chars，也不要连发一串单字消息。
        return shifted if shifted >= min_chars else max_chars

    return None


def pop_stream_segments(
    buffer: str,
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[list[str], str]:
    # 块边界可能把 "\r\n" 截成两块：末尾孤立的 "\r" 留到下一次调用再归一化，
    # 就地转成 "\n" 会让 "abc\r" + "\ndef" 多出一个空行，切法随 chunk 大小漂移。
    # flush 路径没有"下一次"，由 _make_flush_chain 归一化。
    raw = buffer
    dangling_cr = raw.endswith("\r")
    if dangling_cr:
        raw = raw[:-1]
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []

    while text:
        boundary = find_stream_boundary(
            text,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if boundary is None:
            break
        # 平台会无分隔符拼接各段，边界空白也是单词分隔符和缩进。
        segment, text = text[:boundary], text[boundary:]
        if has_visible_text(segment):
            segments.append(segment)

    return segments, text + ("\r" if dangling_cr else "")


def make_stream_segment_chain(
    text: str,
    prefix: Sequence[Any] = (),
) -> MessageChain:
    return MessageChain([*prefix, Plain(text)], type=STREAMING_SEGMENT_CHAIN_TYPE)


def _make_flush_chain(
    buffer: str,
    prefix: Sequence[Any] = (),
) -> MessageChain | None:
    """辅助：从 buffer 构造一条消息链，空 buffer 返回 None。

    只剥尾部空格/制表符：换行符与行首缩进都是内容（平台无分隔符拼接各段）。
    """
    # 只去尾部空格：剩下的这段可能从行首开始，行首缩进是内容（代码块、嵌套列表）
    text = buffer.replace("\r\n", "\n").replace("\r", "\n").rstrip(" \t")
    return make_stream_segment_chain(text, prefix) if has_visible_text(text) else None


async def stream_realtime_segments(
    stream: AsyncGenerator[MessageChain, None],
    settings: SegmentationSettings,
) -> AsyncGenerator[MessageChain, None]:
    buffer = ""
    tail_context = ""
    decorations: list[Any] = []
    thinking_filter = _ThinkingFilter()

    def take_decorations() -> list[Any]:
        carried = list(decorations)
        decorations.clear()
        return carried

    def clean_tail(text: str, *, final: bool = False) -> str:
        nonlocal tail_context
        combined = tail_context + text
        spans = mask_uncuttable_spans(combined, open_tails=True)
        protected_prefix = max((end - len(tail_context) for _, end in spans), default=0)
        # 长代码或 URL 被强制截断后，下一段仍属于同一受保护单元。
        tail_context = (
            combined[spans[-1][0] :] if spans and spans[-1][1] == len(combined) else ""
        )
        # 已发完的代码行不影响后续围栏识别，只保留开围栏和当前行。
        if "\n" in tail_context and _FENCE_RE.match(tail_context):
            opening, _, rest = tail_context.partition("\n")
            tail_context = opening + "\n" + rest.rsplit("\n", 1)[-1]
        # 下一块尚未到达时，数字后的逗号可能是千位分隔符。
        if not final and len(text) >= 2 and text[-2].isdigit() and text[-1] in ".,":
            return text
        # 长度硬切处的点号可能是小数点，只清理自然断句和最终收尾。
        if final or (
            text and text[-1] in STRONG_BOUNDARY_MARKS + WEAK_BOUNDARY_MARKS + "\n"
        ):
            return clean_segment_tail(
                text, settings.tail_clean_marks, protected_prefix=protected_prefix
            )
        return text

    def flush() -> MessageChain | None:
        nonlocal buffer, tail_context
        pending, buffer = buffer, ""
        if settings.tail_clean_enabled:
            pending = clean_tail(pending, final=True)
        tail_context = ""
        if not has_visible_text(pending):
            return None
        return _make_flush_chain(pending, take_decorations())

    def add_text(text: str) -> list[str]:
        nonlocal buffer
        if not text:
            return []
        buffer += text
        segments, buffer = pop_stream_segments(
            buffer,
            min_chars=settings.streaming_min_chars,
            max_chars=settings.streaming_max_chars,
        )
        if settings.tail_clean_enabled:
            # 块尾符号需要下一块或 EOF 才能区分数字延续与真正收尾。
            if (
                segments
                and not buffer
                and segments[-1][-1] in settings.tail_clean_marks
                and (
                    segments[-1][-1] in ".,:："
                    or segments[-1][-1]
                    not in STRONG_BOUNDARY_MARKS + WEAK_BOUNDARY_MARKS
                )
            ):
                buffer = segments.pop()
            segments = [clean_tail(segment) for segment in segments]
            segments = [segment for segment in segments if has_visible_text(segment)]
        return segments

    try:
        async for chain in stream:
            # 带 type 的链（break / audio_chunk / reasoning / tool_call 等）承载业务
            # 语义（如 webchat 的音频播放、对话分段），不能拆也不能丢：先把待发
            # 文本 flush 出去，再原样透传。
            if chain.type is not None:
                if pending := thinking_filter.flush_pending():
                    for segment in add_text(pending):
                        yield make_stream_segment_chain(segment, take_decorations())
                if (flushed := flush()) is not None:
                    yield flushed
                yield chain
                continue

            # Preserve original component order: Plain merges into the pending
            # buffer (flushing when a boundary is reached), non-Plain flushes
            # the buffer first and passes through.
            for component in chain.chain:
                if isinstance(component, Plain):
                    for segment in add_text(thinking_filter.feed(component.text)):
                        yield make_stream_segment_chain(segment, take_decorations())
                elif isinstance(component, _DECORATION_COMPONENTS):
                    if pending := thinking_filter.flush_pending():
                        for segment in add_text(pending):
                            yield make_stream_segment_chain(segment, take_decorations())
                    # 只跟随它之后的正文：先把之前的正文 flush 出去，再挂到下一段。
                    if (flushed := flush()) is not None:
                        yield flushed
                    decorations.append(component)
                else:
                    if pending := thinking_filter.flush_pending():
                        for segment in add_text(pending):
                            yield make_stream_segment_chain(segment, take_decorations())
                    if (flushed := flush()) is not None:
                        yield flushed
                    yield MessageChain([component])

        for segment in add_text(thinking_filter.finish()):
            yield make_stream_segment_chain(segment, take_decorations())
        if (flushed := flush()) is not None:
            yield flushed
        if decorations:
            # 整条流都没有正文：装饰组件没有可依附的消息，丢弃好过发一条空 @。
            logger.debug(
                "智能分段流式兼容丢弃无正文可依附的装饰组件: %s",
                [type(component).__name__ for component in decorations],
            )
    except Exception as exc:
        # 先把已缓冲内容 flush 给用户，再上抛：未打补丁的核心语义就是让源流异常
        # 冒泡到平台，吞掉它会让用户拿到静默截断的回复还以为发送成功。
        # exc_info 不开：堆栈会写进产品日志，摘要足够定位，与 patches/follow_up 同口径。
        logger.error("智能分段流式兼容处理异常: %s", brief_exception(exc))
        if (flushed := flush()) is not None:
            yield flushed
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
    plugins = event.plugins_name
    if plugins is not None and plugins != ["*"] and PLUGIN_NAME not in plugins:
        return
    # 平台不支持流式消息时，宿主会把每条链重新拼成一个 buffer 再按标点切分
    # （aiocqhttp 默认如此）：本插件实时分段的边界到不了用户，只多一层缓冲延迟。
    if not event.platform_meta.support_streaming_message:
        return
    if is_guarded:
        return
    if event.get_extra(STREAMING_WRAPPED_KEY, False):
        return

    stream = result.async_stream
    if stream is None:
        return

    result.async_stream = stream_realtime_segments(stream, settings)
    event.set_extra(STREAMING_WRAPPED_KEY, True)
    logger.info("智能分段已接管流式输出，将按实时断句分段发送")
