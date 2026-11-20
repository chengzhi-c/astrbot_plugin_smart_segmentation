from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from astrbot.api.star import Context
from astrbot_plugin_smart_segmentation.follow_up import FollowUpDispatcher
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

# delay_max=0 把抖动与字数权重一起夹成 0，补发立即发生。
FAST_SETTINGS = SegmentationSettings(delay_base=0, delay_per_char=0, delay_max=0)


class FakeContext:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, session, chain):
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
async def test_follow_up_does_not_repeat_unknown_segment() -> None:
    class SubmittedThenFailedContext(FakeContext):
        send_message = Context.send_message

        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True
            self.platform_manager = SimpleNamespace(platform_insts=[self])

        def meta(self):
            return SimpleNamespace(id="test", name="test")

        async def send_by_session(self, session, chain):
            await super().send_message(str(session), chain)
            if self.fail_once:
                self.fail_once = False
                raise TimeoutError("adapter submitted before response timed out")

        def get_config(self, **_kwargs):
            return {}

    ctx = SubmittedThenFailedContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    pending_id = dispatcher.register(
        session="test:GroupMessage:42",
        segments=["二", "三", "四"],
        settings=settings,
    )
    pending = dispatcher.pop(pending_id)
    assert pending is not None
    await dispatcher._run(pending)
    assert ctx.sent == [
        ("test:GroupMessage:42", "二"),
        ("test:GroupMessage:42", "三\n四"),
    ]


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

    当前段可能已提交，后续段合并补发，且不能连坐同会话后面排队的任务。
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
            segments=["首段", "一", "二", "三"],
            settings=settings,
        ),
    )
    assert pending is not None
    dispatcher.schedule(pending)

    for task in list(dispatcher._tasks):
        await task
    assert ctx.sent == [("s1", "一"), ("s1", "二\n三")]


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
async def test_orphan_caller_cancellation_stops_send() -> None:
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
async def test_cancelling_queued_burst_does_not_cancel_previous_send() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowFirstContext(FakeContext):
        async def send_message(self, session, chain):
            if chain.get_plain_text() == "first":
                started.set()
                await release.wait()
            return await super().send_message(session, chain)

    ctx = SlowFirstContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)
    tasks = []
    try:
        for segment in ("first", "second", "third"):
            pending = dispatcher.pop(
                dispatcher.register(
                    session="s1", segments=[segment], settings=FAST_SETTINGS
                )
            )
            dispatcher.schedule(pending, start_index=0)
            tasks.append(next(task for task in dispatcher._tasks if task not in tasks))
            if segment == "first":
                await asyncio.wait_for(started.wait(), timeout=1)
        queued_task = tasks[1]
        await asyncio.sleep(0)
        queued_task.cancel()
        await asyncio.gather(queued_task, return_exceptions=True)
        await asyncio.sleep(0)
        assert ctx.sent == []
        release.set()
        await asyncio.gather(*dispatcher._tasks, return_exceptions=True)
        assert ctx.sent == [("s1", "first"), ("s1", "third")]
    finally:
        release.set()
        await dispatcher.shutdown()


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
    stale = dispatcher.register(session="s1", segments=["B0", "B1"], settings=settings)
    dispatcher._pending[stale].first_send_delivered = True
    dispatcher.flush_stale(stale)
    ctx.release.set()

    for task in list(dispatcher._tasks):
        await task
    assert [text for _session, text in ctx.sent] == ["A1", "A2", "B1"]


@pytest.mark.asyncio
async def test_expired_pending_without_first_send_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首段未确认发送时过期 pending 只回收，不得自行复活发送整条回复。"""
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod

    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: True)  # type: ignore[arg-type]
    settings = FAST_SETTINGS
    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)

    dispatcher.register(session="s1", segments=["二", "三"], settings=settings)
    dispatcher.reap_expired()
    for task in list(dispatcher._tasks):
        await task
    assert ctx.sent == []


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
@pytest.mark.parametrize(
    "interrupt", [None, "disable", "stop", "false", "unknown", "platform_cancel"]
)
async def test_merged_remainder_is_split_at_message_limit(
    interrupt: str | None,
) -> None:
    enabled = True
    ctx = FakeContext()
    dispatcher = FollowUpDispatcher(ctx, is_enabled=lambda: enabled)  # type: ignore[arg-type]
    original_send = ctx.send_message

    async def send_then_interrupt(session, chain):
        nonlocal enabled
        sent = await original_send(session, chain)
        if interrupt == "disable":
            enabled = False
        elif interrupt == "stop":
            dispatcher.mark_stopping()
        elif interrupt == "false":
            return False
        elif len(ctx.sent) == 1:
            if interrupt == "unknown":
                raise TimeoutError("adapter submitted before response timed out")
            if interrupt == "platform_cancel":
                raise asyncio.CancelledError
        return sent

    ctx.send_message = send_then_interrupt  # type: ignore[method-assign]
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
    if interrupt in {"disable", "stop", "false"}:
        assert sent == ["a" * 1000 + "\n" + "b" * 499]
        return
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


@pytest.mark.asyncio
async def test_shutdown_is_bounded_when_task_ignores_cancellation(monkeypatch) -> None:
    """平台适配器吞掉取消时 terminate 不能无限等，热重载必须能继续。"""

    started = asyncio.Event()

    async def stubborn() -> None:
        try:
            started.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(10)

    dispatcher = FollowUpDispatcher(FakeContext(), is_enabled=lambda: True)  # type: ignore[arg-type]
    dispatcher._track(stubborn, "s1")
    task = next(iter(dispatcher._tasks))
    await asyncio.wait_for(started.wait(), timeout=1)
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.SHUTDOWN_TIMEOUT_SECONDS",
        0.05,
    )
    started = asyncio.get_running_loop().time()
    try:
        await dispatcher.shutdown()
        assert asyncio.get_running_loop().time() - started < 1.0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
