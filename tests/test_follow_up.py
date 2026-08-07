from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from astrbot_plugin_smart_segmentation.follow_up import (
    FollowUpDispatcher,
    is_platform_send_failure,
)
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings


class FakeContext:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail_once = False

    async def send_message(self, session, chain):
        if self.fail_once:
            self.fail_once = False
            exc = RuntimeError("fail")
            exc.result = {"status": "failed", "retcode": 1, "message": "boom"}  # type: ignore[attr-defined]
            raise exc
        text = "".join(getattr(c, "text", "") for c in chain.chain)
        self.sent.append((session, text))
        return True


def test_is_platform_send_failure_dict_result() -> None:
    exc = RuntimeError("x")
    exc.result = {"status": "failed", "retcode": 1200}  # type: ignore[attr-defined]
    assert is_platform_send_failure(exc) is True


def test_is_platform_send_failure_aiocqhttp_action_failed() -> None:
    exc = type(
        "ActionFailed",
        (),
        {"__module__": "astrbot.core.platform.sources.aiocqhttp.something"},
    )()
    assert is_platform_send_failure(exc) is True


def test_is_platform_send_failure_network_errors() -> None:
    assert is_platform_send_failure(TimeoutError("t")) is True
    assert is_platform_send_failure(ConnectionError("c")) is True
    assert is_platform_send_failure(OSError("o")) is True


def test_is_platform_send_failure_retcode_nonzero() -> None:
    exc = RuntimeError("x")
    exc.result = {"retcode": 1}  # type: ignore[attr-defined]
    assert is_platform_send_failure(exc) is True


def test_is_platform_send_failure_plain_exception_false() -> None:
    assert is_platform_send_failure(RuntimeError("x")) is False
    assert is_platform_send_failure(ValueError("v")) is False


@pytest.mark.asyncio
async def test_follow_up_sends_all_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(
        session="s1",
        segments=["二", "三"],
        settings=settings,
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    task = asyncio.create_task(dispatcher._run(pending))
    await task
    assert ctx.sent == [("s1", "二"), ("s1", "三")]


@pytest.mark.asyncio
async def test_follow_up_stops_on_platform_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    ctx = FakeContext()
    ctx.fail_once = True
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(
        session="s1",
        segments=["二", "三"],
        settings=settings,
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_session_guard_blocks_reentry() -> None:
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    assert dispatcher.is_session_guarded("s1") is False
    with dispatcher._guard_session("s1"):
        assert dispatcher.is_session_guarded("s1") is True
    assert dispatcher.is_session_guarded("s1") is False


def test_pending_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    now = 1000.0
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.time.monotonic",
        lambda: now,
    )
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings()
    pending_id = dispatcher.register(
        session="s1",
        segments=["二"],
        settings=settings,
    )
    assert dispatcher.pop(pending_id) is not None
    pending_id = dispatcher.register(
        session="s1",
        segments=["二"],
        settings=settings,
    )
    now += 61.0
    assert dispatcher.pop(pending_id) is None


def test_delay_clamped_when_max_below_base() -> None:
    from astrbot_plugin_smart_segmentation.segmentation import calculate_send_delay

    delay = calculate_send_delay("abc", 5.0, 0.0, 1.0)
    assert delay <= 1.0
