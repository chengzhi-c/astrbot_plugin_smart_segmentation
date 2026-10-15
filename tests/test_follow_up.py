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
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_follow_up_stops_when_send_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """send_message 返回 False（找不到平台）必须停止剩余补发。"""
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
    # 只尝试第一段就中止（不去重发第二段）
    assert ctx.sent == [("s1", "二")]


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

    class DisablingContext(FakeContext):
        async def send_message(self, session, chain):
            await super().send_message(session, chain)
            return True

    ctx = DisablingContext()
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


@pytest.mark.asyncio
async def test_session_guard_blocks_reentry() -> None:
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


def test_result_applied_marker_survives_address_reuse() -> None:
    """去重必须按对象身份判定，不能按 id()。

    result 释放后地址可能被同一 event 上的新 result 复用（agent 多步循环每步
    都新建 MessageEventResult），旧实现存 id() 会把新回复误判为已处理而静默
    跳过分段。
    """
    import gc

    class FakeEvent:
        def __init__(self) -> None:
            self.extra: dict = {}

        def get_extra(self, key, default=None):
            return self.extra.get(key, default)

        def set_extra(self, key, value) -> None:
            self.extra[key] = value

    class WeakBox:
        __slots__ = ("x", "__weakref__")  # 支持弱引用：生产路径

        def __init__(self) -> None:
            self.x = None

    class HardBox:
        __slots__ = ("x",)  # 不支持弱引用：强引用回退路径

        def __init__(self) -> None:
            self.x = None

    # 弱引用路径：存活时按身份判真，释放后任何对象都判假（含地址复用的对象）
    event = FakeEvent()
    dispatcher = FollowUpDispatcher(FakeContext(), is_enabled=lambda: True)  # type: ignore[arg-type]
    box: WeakBox = WeakBox()
    dispatcher.mark_pending(event, "p1", box)
    assert dispatcher.is_result_applied(event, box) is True
    assert dispatcher.is_result_applied(event, WeakBox()) is False
    assert dispatcher.is_result_applied(event, None) is False
    del box
    gc.collect()
    assert dispatcher.is_result_applied(event, WeakBox()) is False

    # clear_extras 之后任何参数都不能炸（ extras 里只剩空串）
    dispatcher.clear_extras(event)
    assert dispatcher.is_result_applied(event, None) is False
    assert dispatcher.is_result_applied(event, WeakBox()) is False

    # 强引用回退路径 + 历史缺陷红灯：旧实现把 id(result) 当整数存进 extras
    # （不持引用），box 释放后同形状新对象复用地址即被误判为已处理。多轮验证；
    # 对当前实现该断言不会触发（标记持有对象，地址不可能被复用）——它守的是
    # "退回到旧实现"这一回归，不是当前行为。
    dispatcher.clear_extras(event)
    for _ in range(20):
        box = HardBox()
        dispatcher.mark_pending(event, "p3", box)
        address = id(box)
        del box
        for _ in range(200):
            reused = HardBox()
            if id(reused) == address:
                assert dispatcher.is_result_applied(event, reused) is False
                break
