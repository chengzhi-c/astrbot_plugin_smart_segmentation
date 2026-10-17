from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
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

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
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
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
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
async def test_plugin_skips_segmentation_when_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """terminate 落在分段调用期间时不得替换结果链。

    否则链已被换成首段、补发又被 after_message_sent 丢弃，用户只看到首段。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)

    async def fake_segment(event, text, settings):
        plugin._stopping = True  # 模拟 terminate 在分段调用期间发生
        return ["你好", "世界", "继续说一点"]

    monkeypatch.setattr(plugin._segmenter, "segment", fake_segment)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert plugin._follow_ups.pending_id_of(event) == ""

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


@pytest.mark.asyncio
async def test_sync_loop_installs_patch_without_message_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流式路径无 hook，补丁装卸必须能不依赖消息事件完成（D-1）。"""
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    from astrbot_plugin_smart_segmentation import main as main_mod

    original = ResultDecorateStage.process
    monkeypatch.setattr(main_mod, "STREAMING_SYNC_INTERVAL_SECONDS", 0.01)

    config = {"enabled": True, "streaming_compat_enabled": False}
    plugin = main_mod.SmartSegmentationPlugin(FakeContext(), config)  # type: ignore[arg-type]
    try:
        assert plugin._streaming is None

        config["streaming_compat_enabled"] = True
        await asyncio.sleep(0.05)

        assert plugin._streaming is not None
        assert plugin._streaming._desired_enabled is True
        # 只做决策不够：补丁必须真的装上，否则流式兼容不生效
        assert plugin._streaming._installed is True
        assert ResultDecorateStage.process is not original
    finally:
        await plugin.terminate()
        assert ResultDecorateStage.process is original


@pytest.mark.asyncio
async def test_after_message_sent_drops_pending_when_stopping() -> None:
    """停止中必须丢弃 pending 并清簿记：否则补发会在插件关闭后飞出。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]

    plugin._stopping = True
    await plugin.after_message_sent(event)  # type: ignore[arg-type]
    # 让出事件循环：若这里走了 schedule，任务会真的发出去
    await asyncio.sleep(0)
    assert plugin._follow_ups._pending == {}  # type: ignore[attr-defined]
    assert plugin._follow_ups._tasks == set()  # type: ignore[attr-defined]
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    assert ctx.sent == []

    await plugin.terminate()


@pytest.mark.asyncio
async def test_terminate_unpatches_even_if_shutdown_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutdown 抛异常时补丁仍必须卸掉：否则它永久留在 ResultDecorateStage 上，
    插件实例也随闭包一起无法回收。"""
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    from astrbot_plugin_smart_segmentation.streaming.patches import _ORIGINAL_ATTR

    original = ResultDecorateStage.process
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {"enabled": True, "streaming_compat_enabled": True},
    )
    assert plugin._streaming is not None
    assert plugin._streaming._installed is True

    async def broken_shutdown() -> None:
        raise RuntimeError("shutdown exploded")

    monkeypatch.setattr(plugin._follow_ups, "shutdown", broken_shutdown)
    with pytest.raises(RuntimeError):
        await plugin.terminate()

    assert ResultDecorateStage.process is original
    assert not hasattr(ResultDecorateStage, _ORIGINAL_ATTR)


@pytest.mark.asyncio
async def test_guarded_decorate_clears_stale_bookkeeping() -> None:
    """会话守卫分支必须清掉旧簿记：否则上一轮残留的 pending 会被本轮
    after_message_sent 当成新登记的发出去（重复补发），或在 _pending 里滞留。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    stale_id = plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]
    assert stale_id
    assert stale_id in plugin._follow_ups._pending  # type: ignore[attr-defined]

    # 模拟：上一轮登记了 pending，但 RespondStage 被跳过；本轮同 event 被守卫挡住
    plugin._follow_ups._guards["sess:1"] = 1  # type: ignore[attr-defined]
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    assert stale_id not in plugin._follow_ups._pending  # type: ignore[attr-defined]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_sync_streaming_does_not_install_after_terminate() -> None:
    """terminate 之后在途 hook 不得装上补丁：同步任务已取消，没人再卸它。

    关键场景是加载时流式兼容没开（`_streaming is None`）——补丁管理器自己的
    `_stopping` 保护不到它，只有 main 的停止检查能拦住。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process
    config = {"enabled": True, "streaming_compat_enabled": False}
    plugin = SmartSegmentationPlugin(FakeContext(), config)  # type: ignore[arg-type]
    assert plugin._streaming is None
    await plugin.terminate()

    config["streaming_compat_enabled"] = True
    plugin._sync_streaming(plugin._get_settings())  # 在途 hook 的迟到调用
    assert plugin._streaming is None
    assert ResultDecorateStage.process is original


@pytest.mark.asyncio
async def test_sync_loop_stops_on_terminate() -> None:
    """terminate 必须回收周期同步任务，不留悬挂协程。"""
    from astrbot_plugin_smart_segmentation import main as main_mod

    plugin = main_mod.SmartSegmentationPlugin(FakeContext(), {"enabled": True})  # type: ignore[arg-type]
    task = plugin._sync_task
    assert task is not None
    await plugin.terminate()
    assert task.done()
