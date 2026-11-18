from __future__ import annotations

import asyncio

import pytest
from astrbot_plugin_smart_segmentation.follow_up import FollowUpDispatcher
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

# delay_max=0 把抖动与字数权重一起夹成 0，补发立即发生。
FAST_SETTINGS = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)


class FakeContext:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail_once = False

    async def send_message(self, session, chain):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("fail")
        text = "".join(getattr(c, "text", "") for c in chain.chain)
        self.sent.append((session, text))
        return True


@pytest.mark.asyncio
async def test_follow_up_sends_all_segments() -> None:
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
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
async def test_follow_up_stops_on_platform_failure() -> None:
    ctx = FakeContext()
    ctx.fail_once = True
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
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
async def test_follow_up_stops_when_send_returns_false() -> None:
    """send_message 返回 False 表示找不到可用平台：停止补发，不再合并重试。"""

    class FalseContext(FakeContext):
        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            self.sent.append((session, text))
            return False

    ctx = FalseContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending_id = dispatcher.register(
        session="s1", segments=["二", "三"], settings=settings
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    # 合并重试同样找不到平台，只会把「平台不在」掩盖成「发送失败」，直接停。
    assert ctx.sent == [("s1", "二")]


@pytest.mark.asyncio
async def test_follow_up_stops_when_disabled_mid_burst() -> None:
    """补发执行中插件被禁用，剩余段不得继续发送（用户不会在关闭后还收到消息）。"""
    enabled = True

    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: enabled)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending_id = dispatcher.register(
        session="s1", segments=["二", "三"], settings=settings
    )
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
async def test_follow_up_holds_session_guard_while_sending() -> None:
    """补发执行期间必须持有会话守卫：否则同会话新回复会与本轮补发交错，
    主处理器也可能被补发重新触发。"""
    guarded_during_send: list[bool] = []

    class GuardCheckingContext(FakeContext):
        async def send_message(self, session, chain):
            guarded_during_send.append(self._dispatcher.is_session_guarded(session))
            return await super().send_message(session, chain)

    ctx = GuardCheckingContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    ctx._dispatcher = dispatcher  # type: ignore[attr-defined]
    settings = FAST_SETTINGS
    pending_id = dispatcher.register(
        session="s1", segments=["二", "三"], settings=settings
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    assert guarded_during_send == [True, True]
    assert dispatcher.is_session_guarded("s1") is False


@pytest.mark.asyncio
async def test_platform_cancelled_send_falls_back_to_merged() -> None:
    """平台适配器被取消时 CancelledError 会穿过 send_message。

    那不是调用方取消：剩余段必须走合并兜底，且不能连坐同会话后面排队的任务。
    """

    class CancelledContext(FakeContext):
        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            self.sent.append((session, text))
            if text == "一":
                child = asyncio.ensure_future(asyncio.sleep(30))
                child.cancel()
                await child
            return True

    ctx = CancelledContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["首段", "一", "二"],
            settings=settings,
        ),
    )
    assert pending is not None
    dispatcher.schedule(pending)

    for task in list(dispatcher._tasks):
        await task
    assert ctx.sent == [("s1", "一"), ("s1", "一\n二")]


@pytest.mark.asyncio
async def test_second_platform_cancellation_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:

    class AlwaysCancelledContext(FakeContext):
        attempts = 0

        async def send_message(self, session, chain):
            self.attempts += 1
            raise asyncio.CancelledError

    ctx = AlwaysCancelledContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["head", "sent", "not-sent"],
            settings=settings,
        ),
    )
    assert pending is not None
    await dispatcher._run(pending)
    assert ctx.attempts == 2
    assert "合并补发" in caplog.text
    assert "合并补发再次被平台取消" in caplog.text


@pytest.mark.asyncio
async def test_caller_cancelled_burst_does_not_send() -> None:
    """调用方取消（terminate）就是不该再发：不能把取消当成异常去合并兜底。"""

    class SlowContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            self.sent.append((session, text))
            self.started.set()
            await asyncio.Event().wait()
            return True

    ctx = SlowContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["head", "sent", "not-sent"],
            settings=settings,
        ),
    )
    assert pending is not None
    dispatcher.schedule(pending)
    await asyncio.wait_for(ctx.started.wait(), timeout=1)

    for task in list(dispatcher._tasks):
        task.cancel()
    for task in list(dispatcher._tasks):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert ctx.sent == [("s1", "sent")]


@pytest.mark.asyncio
async def test_wait_previous_swallows_platform_cancellation() -> None:
    """上一轮任务是被取消的(不是我们自己被取消):本轮照常发自己的段,不能连坐。"""

    started = asyncio.Event()
    release = asyncio.Event()
    finished = False

    async def hang() -> None:
        nonlocal finished
        started.set()
        try:
            await release.wait()
        finally:
            finished = True

    previous = asyncio.ensure_future(hang())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not previous.done()
    previous.cancel()
    with pytest.raises(asyncio.CancelledError):
        await previous
    assert previous.cancelled()

    await FollowUpDispatcher._wait_previous(previous)  # 不应上抛
    assert finished  # 真等过它:直接 return 的实现会在这里挂掉

    class SlowContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def send_message(self, session, chain):
            self.started.set()
            await asyncio.Event().wait()
            return True

    ctx = SlowContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(session="s1", segments=["一", "二"], settings=settings),
    )
    assert pending is not None
    task = asyncio.ensure_future(dispatcher._run_orphan(pending))
    await asyncio.wait_for(ctx.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_orphan_delivery_waits_for_inflight_burst() -> None:
    """兜底补发必须排在在飞 burst 之后：插进去会把上一轮的消息顺序打乱。"""

    class SlowContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def send_message(self, session, chain):
            text = "".join(getattr(c, "text", "") for c in chain.chain)
            self.sent.append((session, text))
            if text == "A1":
                self.started.set()
                await self.release.wait()
            return True

    ctx = SlowContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    first = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["首段", "A1", "A2"],
            settings=settings,
        ),
    )
    assert first is not None
    dispatcher.schedule(first)
    await asyncio.wait_for(ctx.started.wait(), timeout=1)
    stale = dispatcher.register(session="s1", segments=["B1"], settings=settings)
    dispatcher.flush_stale(stale)
    ctx.release.set()

    for task in list(dispatcher._tasks):
        await task
    assert [text for _session, text in ctx.sent] == ["A1", "A2", "B1"]


@pytest.mark.asyncio
async def test_expired_pending_is_delivered_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """交接钩子没触发时必须兜底交付：首段已替换进原链，剩余段只存在 pending 里，
    静默清理等于用户收到被插件自己截断的回复。"""
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod

    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)

    dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    dispatcher.reap_expired()
    for task in list(dispatcher._tasks):
        await task
    assert ctx.sent == [("s1", "二\n三")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stopping", [False, True])
async def test_merged_remainder_respects_stopping_and_disabled(
    stopping: bool,
) -> None:
    """合并补发兜底同样要遵守停止/禁用：否则插件关闭后用户仍会收到消息。"""
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: stopping)  # type: ignore[arg-type]
    if stopping:
        dispatcher.mark_stopping()
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    )
    assert pending is not None
    await dispatcher._send_merged_remainder(pending, 0)
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_merged_remainder_is_split_at_message_limit() -> None:
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["a" * 1000, "b" * 1000],
            settings=settings,
        )
    )
    assert pending is not None

    await dispatcher._send_merged_remainder(pending, 0)

    sent = [text for _, text in ctx.sent]
    assert len(sent) > 1
    assert max(map(len, sent)) <= 1500
    assert "".join(sent) == "a" * 1000 + "\n" + "b" * 1000


@pytest.mark.asyncio
async def test_same_session_bursts_are_serialized() -> None:
    """同会话两轮补发必须串行：第三方 agent runner 没有会话锁，两个 burst
    并发起飞会把消息顺序打乱。"""
    order: list[str] = []

    class SlowContext(FakeContext):
        async def send_message(self, session, chain):
            order.append("".join(getattr(c, "text", "") for c in chain.chain))
            await asyncio.sleep(0)  # 让出事件循环，暴露交错
            return True

    ctx = SlowContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    first = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["首段1", "a1", "a2"],
            settings=settings,
        )
    )
    second = dispatcher.pop(
        dispatcher.register(
            session="s1",
            segments=["首段2", "b1", "b2"],
            settings=settings,
        )
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
