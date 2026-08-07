from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain

from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin


class FakeResult:
    def __init__(self, text: str) -> None:
        self.chain = [Plain(text)]
        self.is_model_result = lambda: True


class FakeEvent:
    def __init__(self, result: FakeResult) -> None:
        self._result = result
        self.extra: dict = {}
        self.unified_msg_origin = "sess:1"

    def get_result(self):
        return self._result

    def get_extra(self, key: str, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value


class FakeContext:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.generate_calls: list[dict] = []

    def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(completion_text='["你好", "世界"]')

    async def send_message(self, session, chain):
        text = "".join(getattr(c, "text", "") for c in chain.chain)
        self.sent.append((session, text))
        return True


@pytest.mark.asyncio
async def test_plugin_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    async def immediate_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.asyncio.sleep",
        immediate_sleep,
    )

    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(
        ctx,  # type: ignore[arg-type]
        {
            "enabled": True,
            "min_length": 2,
            "max_segments": 5,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    # first segment replaced in place, follow-up registered
    assert [c.text for c in result.chain] == ["你好"]
    assert plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]

    # re-decorate same result is a no-op
    calls_before = len(ctx.generate_calls)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert len(ctx.generate_calls) == calls_before

    await plugin.after_message_sent(event)  # type: ignore[arg-type]
    tasks = [t for t in list(plugin._follow_ups._tasks) if not t.done()]  # type: ignore[attr-defined]
    if tasks:
        await asyncio.gather(*tasks)
    assert ctx.sent == [("sess:1", "世界")]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_disabled_passthrough() -> None:
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": False})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。"]
    assert ctx.generate_calls == []
    await plugin.terminate()


class FakeStreamingResult:
    result_content_type = "STREAMING_RESULT"

    def __init__(self, stream) -> None:
        self.async_stream = stream
        self.is_model_result = lambda: False


async def _dummy_stream():
    if False:
        yield MessageChain()


@pytest.mark.asyncio
async def test_plugin_streaming_branch_lazily_creates_manager() -> None:
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "streaming_compat_enabled": True,
            "streaming_min_chars": 2,
            "streaming_max_chars": 90,
        },
    )
    try:
        result = FakeStreamingResult(_dummy_stream())
        event = FakeEvent(result)  # type: ignore[arg-type]
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        assert plugin._streaming is not None
        assert result.async_stream is not _dummy_stream
        assert ctx.generate_calls == []
    finally:
        await plugin.terminate()
