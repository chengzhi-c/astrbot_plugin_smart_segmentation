from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from astrbot.api.event import MessageChain
from astrbot.api.message_components import At, Plain

from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
from astrbot_plugin_smart_segmentation.streaming.wrapper import (
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
        yield MessageChain([Plain("你好呀。")])
        yield MessageChain([Plain("还没完")], type="break")
        yield MessageChain([Plain("尾巴")])

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    types = [c.type for c in collected]
    assert "break" in types
    assert _collected(collected)[0] == ("plain", "你好呀。")


@pytest.mark.asyncio
async def test_stream_exception_still_flushes() -> None:
    async def gen():
        yield MessageChain([Plain("前")])
        raise RuntimeError("boom")

    settings = SegmentationSettings(streaming_min_chars=2, streaming_max_chars=90)
    collected = [chain async for chain in stream_realtime_segments(gen(), settings)]
    assert _collected(collected) == [("plain", "前")]


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
