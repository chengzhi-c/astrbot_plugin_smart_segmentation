from __future__ import annotations

import asyncio

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
    """status="failed" 但无 retcode 时也必须判为失败（不能只看 retcode）。"""
    exc = RuntimeError("x")
    exc.result = {"status": "failed"}  # type: ignore[attr-defined]
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


def test_is_platform_send_failure_retcode_zero_string() -> None:
    """字符串 "0" 也是成功码：误判为失败会中止全部剩余补发、丢失消息。"""
    exc = RuntimeError("x")
    exc.result = {"retcode": "0"}  # type: ignore[attr-defined]
    assert is_platform_send_failure(exc) is False


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
    # 首段就失败：剩余段必须合并为一条补发，不能静默丢弃（内容丢失）
    assert ctx.sent == [("s1", "二\n三")]


@pytest.mark.asyncio
async def test_follow_up_stops_when_send_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """send_message 返回 False（找不到平台）必须停止逐条补发，但内容不能丢。"""
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )

    class FalseContext(FakeContext):
        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            self.sent.append((session, text))
            return False

    ctx = FalseContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    # 第一段返回 False（平台没接住）：它与第二段合并为一条补发，内容不丢
    assert ctx.sent == [("s1", "二"), ("s1", "二\n三")]


@pytest.mark.asyncio
async def test_follow_up_stops_when_disabled_mid_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """补发执行中插件被禁用，剩余段不得继续发送（用户不会在关闭后还收到消息）。"""
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    enabled = True

    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: enabled)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    pending = dispatcher.pop(pending_id)
    assert pending is not None

    original_send = ctx.send_message

    async def send_then_disable(session, chain):
        sent = await original_send(session, chain)
        nonlocal enabled
        enabled = False  # 第一段发出后插件被禁用
        return sent  # 必须返回 True：否则中断来自 `not sent` 而非 is_enabled

    ctx.send_message = send_then_disable  # type: ignore[method-assign]
    await dispatcher._run(pending)
    assert ctx.sent == [("s1", "二")]


def test_session_guard_blocks_reentry() -> None:
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    assert dispatcher.is_session_guarded("s1") is False
    with dispatcher._guard_session("s1"):
        assert dispatcher.is_session_guarded("s1") is True
    assert dispatcher.is_session_guarded("s1") is False


@pytest.mark.asyncio
async def test_follow_up_holds_session_guard_while_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """补发执行期间必须持有会话守卫：否则同会话新回复会与本轮补发交错，
    主处理器也可能被补发重新触发。"""
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    guarded_during_send: list[bool] = []

    class GuardCheckingContext(FakeContext):
        async def send_message(self, session, chain):
            guarded_during_send.append(self._dispatcher.is_session_guarded(session))
            return await super().send_message(session, chain)

    ctx = GuardCheckingContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    ctx._dispatcher = dispatcher  # type: ignore[attr-defined]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    assert guarded_during_send == [True, True]
    assert dispatcher.is_session_guarded("s1") is False


def test_pending_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """未认领的过期 pending 会被清理；但被显式认领的那条必须交付。

    注入 TTL 常量而不是 patch 全局 time.monotonic：follow_up.time 就是 stdlib
    time 模块，patch 它会污染同进程内所有代码的时间语义。
    """
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod

    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings()
    pending_id = dispatcher.register(
        session="s1",
        segments=["二"],
        settings=settings,
    )
    assert dispatcher.pop(pending_id) is not None  # 未过期：照常取到

    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)
    stale_id = dispatcher.register(session="s1", segments=["二"], settings=settings)
    # 显式认领的过期条目仍要交付：首段已替换进原链，丢弃即内容丢失
    assert dispatcher.pop(stale_id) is not None

    stale_id = dispatcher.register(session="s1", segments=["二"], settings=settings)
    dispatcher.register(session="s1", segments=["三"], settings=settings)  # 触发清理
    assert stale_id not in dispatcher._pending


@pytest.mark.asyncio
async def test_merged_remainder_respects_stopping_and_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合并补发兜底同样要遵守停止/禁用：否则插件关闭后用户仍会收到消息。"""
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: False)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending = dispatcher.pop(
        dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    )
    assert pending is not None
    await dispatcher._send_merged_remainder(pending, 0)
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_same_session_bursts_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同会话两轮补发必须串行：第三方 agent runner 没有会话锁，两个 burst
    并发起飞会把消息顺序打乱。"""
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )
    order: list[str] = []

    class SlowContext(FakeContext):
        async def send_message(self, session, chain):
            order.append("".join(getattr(c, "text", "") for c in chain.chain))
            await asyncio.sleep(0)  # 让出事件循环，暴露交错
            return True

    ctx = SlowContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    first = dispatcher.pop(
        dispatcher.register(session="s1", segments=["a1", "a2"], settings=settings)
    )
    second = dispatcher.pop(
        dispatcher.register(session="s1", segments=["b1", "b2"], settings=settings)
    )
    assert first is not None and second is not None
    dispatcher.schedule(first)
    dispatcher.schedule(second)
    await asyncio.gather(*list(dispatcher._tasks))
    assert order == ["a1", "a2", "b1", "b2"]


@pytest.mark.asyncio
async def test_schedule_after_shutdown_creates_no_task() -> None:
    """shutdown 之后 schedule 不得再造出取消不掉的任务。"""
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    await dispatcher.shutdown()
    pending_id = dispatcher.register(
        session="s1",
        segments=["二"],
        settings=SegmentationSettings(),
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    dispatcher.schedule(pending)
    assert dispatcher._tasks == set()
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_unexpected_exception_also_merges_remainder() -> None:
    """非平台类异常同样不能丢内容：适配器自身的 bug 不该让用户少收消息。"""

    class BuggyContext(FakeContext):
        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            if text == "一":
                raise RuntimeError("adapter bug")  # 无 .result、非 OSError
            self.sent.append((session, text))
            return True

    ctx = BuggyContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)
    pending_id = dispatcher.register(
        session="s1",
        segments=["一", "二", "三"],
        settings=settings,
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    assert ctx.sent == [("s1", "一\n二\n三")]


def test_result_applied_marker_uses_object_identity() -> None:
    """去重必须按对象身份判定，不能按 id()。

    背景：result 释放后地址可能被同一 event 上的新 result 复用（agent 多步
    循环每步都新建 MessageEventResult），按 id() 判定会把新回复误判为已处理
    而静默跳过分段。
    """

    class FakeEvent:
        def __init__(self) -> None:
            self.extra: dict = {}

        def get_extra(self, key, default=None):
            return self.extra.get(key, default)

        def set_extra(self, key, value) -> None:
            self.extra[key] = value

    event = FakeEvent()
    dispatcher = FollowUpDispatcher(FakeContext(), is_enabled=lambda: True)  # type: ignore[arg-type]
    result = object()
    dispatcher.mark_pending(event, "p1", result)
    assert dispatcher.is_result_applied(event, result) is True
    assert dispatcher.is_result_applied(event, object()) is False
    assert dispatcher.is_result_applied(event, None) is False

    # clear_extras 之后任何参数都不能炸（ extras 里只剩空串）
    dispatcher.clear_extras(event)
    assert dispatcher.is_result_applied(event, None) is False
    assert dispatcher.is_result_applied(event, result) is False
