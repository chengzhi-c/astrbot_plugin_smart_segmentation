from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from astrbot_plugin_smart_segmentation.segmenter import OutboundSegmenter
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
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


class RaisingGenerateContext:
    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, **kwargs):
        raise RuntimeError("provider exploded")


@pytest.mark.asyncio
async def test_segment_llm_exception_falls_back() -> None:
    """LLM 调用抛异常必须走本地兜底：冒泡出 hook 会让每条消息都炸。"""
    segmenter = OutboundSegmenter(RaisingGenerateContext())  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_segment_empty_completion_skips() -> None:
    """模型返回空内容时不分段（而不是拿原文去本地切）。"""
    ctx = FakeContext(completion="")
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == []


@pytest.mark.asyncio
async def test_segment_prefers_configured_provider_id() -> None:
    """配置了 provider_id 时优先用它，且不再查询当前会话 provider。"""
    ctx = FakeContext(provider_id="session-provider")
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(
        min_length=2, max_segments=5, timeout_seconds=5, provider_id="configured"
    )
    await segmenter.segment(event, "你好。世界。继续说一点", settings)  # type: ignore[arg-type]
    assert ctx.generate_calls[0].get("chat_provider_id") == "configured"


def test_brief_exception_truncates() -> None:
    exc = RuntimeError("x" * 500)
    text = brief_exception(exc)
    assert text.startswith("RuntimeError:")
    assert len(text) < 220


def test_brief_exception_formats_dict_result() -> None:
    """平台异常的 exc.result dict 走 wording 字段并被截断。"""
    exc = RuntimeError("boom")
    exc.result = {  # type: ignore[attr-defined]
        "retcode": 1200,
        "wording": "发送失败" + "y" * 500,
    }
    text = brief_exception(exc)
    assert text.startswith("RuntimeError(retcode=1200")
    assert "发送失败" in text
    assert len(text) < 220


class TypeErrorGenerateContext:
    """llm_generate 声明接受 chat_provider_id，调用时却抛 TypeError（旧签名）。"""

    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, **kwargs):
        if "chat_provider_id" in kwargs:
            raise TypeError("unexpected keyword argument 'chat_provider_id'")
        return SimpleNamespace(completion_text='["a", "b"]')


@pytest.mark.asyncio
async def test_llm_generate_retries_prompt_only_on_type_error() -> None:
    """带参调用被旧签名拒绝时必须退化为 prompt-only 重试一次。"""
    segmenter = OutboundSegmenter(TypeErrorGenerateContext())  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    # 本地兜底会得到 ["第一句", "第二句"]；走到重试才证明降级路径生效
    assert segs == ["a", "b"]


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
