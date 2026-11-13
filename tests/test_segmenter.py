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

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
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
    segs = await segmenter.segment(event, "你好。世界。", settings)  # type: ignore[arg-type]
    assert segs == ["你好", "世界"]
    # 只传 prompt 与 provider：temperature/max_tokens 传过去也会被 provider 丢弃
    assert set(ctx.generate_calls[0]) == {"prompt", "chat_provider_id"}


@pytest.mark.asyncio
async def test_segment_rejects_model_output_that_changed_text() -> None:
    """模型漏字/改写时不能拿它的结果替换原链：用户会收到一条被改残的回复。"""
    ctx = FakeContext()

    async def lossy_generate(**kwargs):
        return SimpleNamespace(
            completion_text='["今天下午去看了那场展览", "人比想象中少很多"]',
        )

    ctx.llm_generate = lossy_generate  # type: ignore[method-assign]
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    text = "今天下午去看了那场展览。人比想象中少很多。不过展品没什么意思。"
    segs = await segmenter.segment(
        SimpleNamespace(unified_msg_origin="sess:1"),  # type: ignore[arg-type]
        text,
        SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5),
    )
    # 退回本地切分：漏掉的那句必须还在
    assert "".join(segs).replace("。", "") == text.replace("。", "")


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
    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
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


class TypeErrorProviderContext(FakeContext):
    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        raise TypeError("signature mismatch")


@pytest.mark.asyncio
async def test_provider_cancelled_error_falls_back_to_local() -> None:
    """provider 自己抛 CancelledError 不是调用方取消：不能让 hook 挂掉，走本地兜底。"""
    ctx = FakeContext()
    ctx.raise_cancelled = True
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_provider_id_typeerror_fallback() -> None:
    ctx = TypeErrorProviderContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]


def test_brief_exception_never_raises() -> None:
    """摘要函数在 except 块里调用：它自己抛异常会让上抛路径变成二次异常。"""

    class Bad(Exception):
        def __str__(self) -> str:
            raise RuntimeError("str() exploded")

    assert brief_exception(Bad()) == "Bad"
