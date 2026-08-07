from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from astrbot_plugin_smart_segmentation.segmenter import OutboundSegmenter
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
from astrbot_plugin_smart_segmentation.streaming.wrapper import stream_realtime_segments
from astrbot_plugin_smart_segmentation.errors import brief_exception


class FakeContext:
    def __init__(self, *, provider_id: str = "p1", completion: str = '["你好", "世界"]') -> None:
        self._provider_id = provider_id
        self._completion = completion
        self.generate_calls: list[dict] = []
        self.delay = 0.0
        self.raise_cancelled = False

    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return self._provider_id

    async def llm_generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.raise_cancelled:
            raise asyncio.CancelledError()
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(completion_text=self._completion)


@pytest.mark.asyncio
async def test_segment_llm_success() -> None:
    ctx = FakeContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "你好。世界。继续说一点", settings)  # type: ignore[arg-type]
    assert segs == ["你好", "世界"]
    assert ctx.generate_calls
    assert "prompt" in ctx.generate_calls[0]


@pytest.mark.asyncio
async def test_segment_fallback_on_bad_json() -> None:
    ctx = FakeContext(completion="not-json")
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_segment_skip_short() -> None:
    ctx = FakeContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=50)
    segs = await segmenter.segment(event, "短", settings)  # type: ignore[arg-type]
    assert segs == []
    assert not ctx.generate_calls


@pytest.mark.asyncio
async def test_segment_timeout_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = FakeContext(completion='["x","y"]')
    ctx.delay = 0.05
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=0.001)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_segment_no_provider_falls_back() -> None:
    ctx = FakeContext(provider_id="")
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, provider_id="", max_segments=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


def test_brief_exception_truncates() -> None:
    exc = RuntimeError("x" * 500)
    text = brief_exception(exc)
    assert text.startswith("RuntimeError:")
    assert len(text) < 220


@pytest.mark.asyncio
async def test_segment_cancelled_when_stopping_raises() -> None:
    ctx = FakeContext()
    ctx.raise_cancelled = True
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    segmenter.mark_stopping()
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    with pytest.raises(asyncio.CancelledError):
        await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_segment_llm_receives_provider_id() -> None:
    ctx = FakeContext(provider_id="p9")
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    await segmenter.segment(event, "你好。世界。继续说一点", settings)  # type: ignore[arg-type]
    assert ctx.generate_calls[0].get("chat_provider_id") == "p9"


class PromptOnlyContext:
    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, prompt: str):
        return SimpleNamespace(completion_text='["a", "b"]')


@pytest.mark.asyncio
async def test_llm_generate_fallback_to_prompt_only() -> None:
    segmenter = OutboundSegmenter(PromptOnlyContext())  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["a", "b"]


class TypeErrorProviderContext(FakeContext):
    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        raise TypeError("signature mismatch")


@pytest.mark.asyncio
async def test_provider_id_typeerror_fallback() -> None:
    ctx = TypeErrorProviderContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_stream_realtime_splits_sentences() -> None:
    async def gen():
        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Plain

        yield MessageChain([Plain("你好啊。接下来")])
        yield MessageChain([Plain("继续说。尾巴")])

    settings = SegmentationSettings(streaming_min_chars=3, streaming_max_chars=90)
    out: list[str] = []
    async for chain in stream_realtime_segments(gen(), settings):
        texts = [c.text for c in chain.chain if hasattr(c, "text")]
        out.extend(texts)
    assert out[0] == "你好啊。"
    assert any("继续说。" in item for item in out)
