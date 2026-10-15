from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from astrbot.api.event import MessageChain
from astrbot.api.message_components import At, Plain

from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
from astrbot_plugin_smart_segmentation.streaming.wrapper import (
    STREAMING_SEGMENT_CHAIN_TYPE,
    decorate_streaming_result,
    stream_realtime_segments,
)


def _collected(items: list[MessageChain]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chain in items:
        for comp in chain.chain:
            if isinstance(comp, Plain):
                out.append(("plain", comp.text))
            elif isinstance(comp, At):
                out.append(("at", str(getattr(comp, "qq", ""))))
    return out


@pytest.mark.asyncio
async def test_stream_splits_and_keeps_component_order() -> None:
    async def gen():
        yield MessageChain([Plain("第一句。"), At(qq="1")])
        yield MessageChain([Plain("第二句。")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    assert _collected(collected) == [
        ("plain", "第一句。"),
        ("at", "1"),
        ("plain", "第二句。"),
    ]


@pytest.mark.asyncio
async def test_stream_break_flushes() -> None:
    async def gen():
        yield MessageChain([Plain("你好呀，接下来")])  # 无边界，留在 buffer
        yield MessageChain([Plain("还没完")], type="break")
        yield MessageChain([Plain("尾巴。")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    types = [c.type for c in collected]
    assert "break" in types
    # break 到达时 buffer 非空：待发段必须在 break 之前 flush；break 链自身
    # 携带的内容按原样透传
    texts = [
        comp.text
        for chain in collected
        for comp in chain.chain
        if isinstance(comp, Plain)
    ]
    assert texts == ["你好呀，接下来", "还没完", "尾巴。"]


@pytest.mark.asyncio
async def test_stream_exception_flushes_then_reraises() -> None:
    """源流异常必须先 flush 已缓冲内容，再上抛。

    吞掉异常会让平台以为流正常结束，用户拿到静默截断的回复；未打补丁的核心
    语义就是让异常冒泡。
    """

    async def gen():
        yield MessageChain([Plain("前")])
        raise RuntimeError("boom")

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected: list[MessageChain] = []
    with pytest.raises(RuntimeError):
        async for chain in stream_realtime_segments(gen(), settings):
            collected.append(chain)
    assert _collected(collected) == [("plain", "前")]


@pytest.mark.asyncio
async def test_stream_accumulates_multiple_plain_chunks() -> None:
    """多个 Plain chunk 必须累加进同一 buffer（逐 token 流式输出的常态）。"""

    async def gen():
        yield MessageChain([Plain("你好啊，")])
        yield MessageChain([Plain("接下来继续说。")])
        yield MessageChain([Plain("尾巴")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    texts = [c.text for chain in collected for c in chain.chain if isinstance(c, Plain)]
    assert texts == ["你好啊，接下来继续说。", "尾巴"]


@pytest.mark.asyncio
async def test_stream_flushes_buffer_before_non_plain_component() -> None:
    """非 Plain 组件到达时若 buffer 非空，待发段必须排在它之前。"""

    async def gen():
        yield MessageChain([Plain("还没到边界的半句话")])
        yield MessageChain([At(qq="1")])
        yield MessageChain([Plain("接着说的话。第二句。")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    assert _collected(collected) == [
        ("plain", "还没到边界的半句话"),
        ("at", "1"),
        ("plain", "接着说的话。"),
        ("plain", "第二句。"),
    ]


@pytest.mark.asyncio
async def test_stream_realtime_splits_sentences() -> None:
    """强句读符边界的端到端守卫：拆掉强标点循环时唯一烧红的用例。"""

    async def gen():
        yield MessageChain([Plain("你好啊。接下来")])
        yield MessageChain([Plain("继续说。尾巴")])

    settings = SegmentationSettings(streaming_min_chars=3, streaming_max_chars=90)
    out: list[str] = []
    async for chain in stream_realtime_segments(gen(), settings):
        texts = [c.text for c in chain.chain if hasattr(c, "text")]
        out.extend(texts)
    assert out[0] == "你好啊。"
    assert any("继续说。" in item for item in out)


@pytest.mark.asyncio
async def test_stream_flushes_buffer_before_typed_chain() -> None:
    """平台专用链（audio_chunk 等）到达时若 buffer 非空，同样先 flush。"""

    async def gen():
        yield MessageChain([Plain("还没到边界的半句话")])
        yield MessageChain([Plain("音频数据")], type="audio_chunk")

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    assert _collected(collected) == [
        ("plain", "还没到边界的半句话"),
        ("plain", "音频数据"),
    ]
    assert collected[1].type == "audio_chunk"


class _FakeEvent:
    def __init__(self) -> None:
        self.extra: dict = {}

    def get_extra(self, key: str, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value


def test_decorate_wraps_once() -> None:
    async def gen():
        if False:
            yield MessageChain()

    result = SimpleNamespace(async_stream=gen())
    original_stream = result.async_stream
    event = _FakeEvent()
    settings = SegmentationSettings(streaming_compat_enabled=True)

    decorate_streaming_result(event, result, settings, is_guarded=False)  # type: ignore[arg-type]
    assert result.async_stream is not original_stream
    first_wrapped = result.async_stream

    decorate_streaming_result(event, result, settings, is_guarded=False)  # type: ignore[arg-type]
    assert result.async_stream is first_wrapped


def test_decorate_skips_when_disabled() -> None:
    async def gen():
        if False:
            yield MessageChain()

    result = SimpleNamespace(async_stream=gen())
    original_stream = result.async_stream
    event = _FakeEvent()
    settings = SegmentationSettings(streaming_compat_enabled=False)
    decorate_streaming_result(event, result, settings, is_guarded=False)  # type: ignore[arg-type]
    assert result.async_stream is original_stream


def test_decorate_skips_when_guarded() -> None:
    async def gen():
        if False:
            yield MessageChain()

    result = SimpleNamespace(async_stream=gen())
    original_stream = result.async_stream
    event = _FakeEvent()
    settings = SegmentationSettings(streaming_compat_enabled=True)
    decorate_streaming_result(event, result, settings, is_guarded=True)  # type: ignore[arg-type]
    assert result.async_stream is original_stream


@pytest.mark.asyncio
async def test_stream_passes_through_platform_typed_chains() -> None:
    """audio_chunk / reasoning 等平台专用链必须原样透传。

    type 承载业务语义（如 webchat 的音频播放、推理内容展示），拆分会把 base64
    音频切成文本垃圾并丢掉类型。
    """
    async def gen():
        yield MessageChain([Plain("。")], type="audio_chunk")
        yield MessageChain([Plain("让我想想。")], type="reasoning")
        yield MessageChain([Plain("你好。世界。")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    assert [c.type for c in collected] == [
        "audio_chunk",
        "reasoning",
        STREAMING_SEGMENT_CHAIN_TYPE,
        STREAMING_SEGMENT_CHAIN_TYPE,
    ]
    assert collected[0].chain[0].text == "。"
    assert collected[1].chain[0].text == "让我想想。"
    assert collected[2].chain[0].text == "你好。"
    assert collected[3].chain[0].text == "世界。"
