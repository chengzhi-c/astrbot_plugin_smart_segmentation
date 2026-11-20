from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from astrbot_plugin_smart_segmentation.segmenter import OutboundSegmenter
from astrbot_plugin_smart_segmentation.settings import (
    SegmentationSettings,
    load_settings,
)


class FakeContext:
    def __init__(
        self, *, provider_id: str = "p1", completion: str | None = '["你好", "世界"]'
    ) -> None:
        self._provider_id = provider_id
        self._completion = completion
        self.generate_calls: list[dict] = []
        self.provider_lookups = 0
        self.delay = 0.0
        self.raise_cancelled = False

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        self.provider_lookups += 1
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
async def test_segment_preserves_format_characters_inside_visible_text() -> None:
    ctx = FakeContext(completion='["你\u200b好", "👩\u200d💻"]')
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    segments = await segmenter.segment(
        SimpleNamespace(unified_msg_origin="sess:1"),  # type: ignore[arg-type]
        "你\u200b好👩\u200d💻",
        SegmentationSettings(min_length=2),
    )
    assert segments == ["你\u200b好", "👩\u200d💻"]


@pytest.mark.asyncio
async def test_segment_rejects_model_output_that_changed_text(caplog) -> None:
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
    assert "分段结果与原文不一致" in caplog.text


@pytest.mark.asyncio
async def test_segment_accepts_comma_kept_at_boundary() -> None:
    """模型把逗号留在段尾是合法切分：内容没变，不能退回本地兜底。"""
    ctx = FakeContext()

    async def comma_generate(**kwargs):
        return SimpleNamespace(
            completion_text='["已经在回了，别急，", "你这不是困了是心累了吧"]',
        )

    ctx.llm_generate = comma_generate  # type: ignore[method-assign]
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    text = "已经在回了，别急，你这不是困了是心累了吧"
    segs = await segmenter.segment(
        SimpleNamespace(unified_msg_origin="sess:1"),  # type: ignore[arg-type]
        text,
        SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5),
    )
    assert segs == ["已经在回了，别急，", "你这不是困了是心累了吧"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_model_can_omit_boundary_comma_only_when_tail_clean_enabled(
    enabled: bool, caplog
) -> None:
    parts = ["材料已备齐，装配也已完成", "接下来开始测试"]
    text = "材料已备齐，装配也已完成，接下来开始测试"
    context = FakeContext(completion=json.dumps(parts))
    result = await OutboundSegmenter(context).segment(
        SimpleNamespace(unified_msg_origin="sess:1"),
        text,
        SegmentationSettings(min_length=2, tail_clean_enabled=enabled),
    )
    assert result == (parts if enabled else [])
    assert ("分段结果与原文不一致" in caplog.text) is not enabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marks,accepted", [(["？"], True), (["，"], False), ([], False)]
)
async def test_custom_tail_marks_control_model_omission(marks, accepted, caplog):
    context = FakeContext(completion='["材料齐了", "开始装配，"]')
    await OutboundSegmenter(context).segment(
        SimpleNamespace(unified_msg_origin="sess:1"),
        "材料齐了？开始装配，",
        load_settings(
            {"min_length": 2, "tail_clean_enabled": True, "tail_clean_marks": marks}
        ),
    )
    assert ("分段结果与原文不一致" not in caplog.text) is accepted
    if accepted:
        assert "「？」" in context.generate_calls[0]["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,parts",
    [
        ("你好，朋友。再见", ["你好朋友", "再见"]),
        ("数值3.14。再见", ["数值3.", "14。再见"]),
        ("看https://example.com/a;b。再见", ["看https://example.com/a", "b。再见"]),
    ],
    ids=["internal-comma", "decimal-boundary", "url-boundary"],
)
async def test_tail_clean_still_rejects_model_content_changes(source, parts, caplog):
    context = FakeContext(completion=json.dumps(parts))
    await OutboundSegmenter(context).segment(
        SimpleNamespace(unified_msg_origin="sess:1"),
        source,
        SegmentationSettings(min_length=2, tail_clean_enabled=True),
    )
    assert "分段结果与原文不一致" in caplog.text


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
async def test_segment_timeout_falls_back() -> None:
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
        raise RuntimeError("https://user:secret@example.com/api?token=abcdef")


@pytest.mark.asyncio
async def test_segment_llm_exception_falls_back(caplog) -> None:
    """LLM 调用抛异常必须走本地兜底：冒泡出 hook 会让每条消息都炸。"""
    segmenter = OutboundSegmenter(RaisingGenerateContext())  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]
    assert "RuntimeError" in caplog.text
    assert "secret" not in caplog.text
    assert "abcdef" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["", None])
async def test_segment_empty_completion_skips(completion) -> None:
    """模型返回空内容时不分段（而不是拿原文去本地切）。"""
    ctx = FakeContext(completion=completion)
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
    assert ctx.provider_lookups == 0


@pytest.mark.asyncio
async def test_segment_skips_model_when_stopping() -> None:
    ctx = FakeContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    segmenter.mark_stopping()
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    assert await segmenter.segment(event, "第一句。第二句。", settings) == []  # type: ignore[arg-type]
    assert ctx.generate_calls == []


class UnavailableProviderContext(FakeContext):
    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        raise ValueError("No available chat provider")


@pytest.mark.asyncio
async def test_provider_lookup_failure_uses_local_fallback() -> None:
    ctx = UnavailableProviderContext()
    segmenter = OutboundSegmenter(ctx)  # type: ignore[arg-type]
    event = SimpleNamespace(unified_msg_origin="sess:1")
    settings = SegmentationSettings(min_length=2, max_segments=5, timeout_seconds=5)
    segs = await segmenter.segment(event, "第一句。第二句。", settings)  # type: ignore[arg-type]
    assert segs == ["第一句", "第二句"]
    assert ctx.generate_calls == []


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
