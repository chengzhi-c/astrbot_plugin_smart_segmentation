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
        # 宿主实现：把链里 Plain 的文本拼起来（多段用空格 join）
        self.get_plain_text = lambda: " ".join(
            comp.text for comp in self.chain if isinstance(comp, Plain)
        )


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
    # 桩输出默认与下文文本一致；分段模型路径若失效（校验拒绝/调用失败）会退回本地
    # 切分，而这些文本没有本地边界，因此断言能区分「模型分段生效」与「全程兜底」。
    def __init__(self, completion: str = '["今天天气不错", "我们出去玩吧"]') -> None:
        self.sent: list[tuple[str, str]] = []
        self.generate_calls: list[dict] = []
        self.completion = completion

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(completion_text=self.completion)

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
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    # first segment replaced in place, follow-up registered
    assert [c.text for c in result.chain] == ["今天天气不错"]
    assert plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]

    # re-decorate same result is a no-op
    calls_before = len(ctx.generate_calls)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert len(ctx.generate_calls) == calls_before

    await plugin.after_message_sent(event)  # type: ignore[arg-type]
    tasks = [t for t in list(plugin._follow_ups._tasks) if not t.done()]  # type: ignore[attr-defined]
    if tasks:
        await asyncio.gather(*tasks)
    assert ctx.sent == [("sess:1", "我们出去玩吧")]

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
async def test_plugin_skips_text_already_sent_by_send_message_to_user() -> None:
    """宿主 RespondStage 按这份 extra 抑制重复回复。

    换链后 result.get_plain_text() 只剩首段，宿主的比对永远不命中，同一条回复会被
    发两遍（agent 用 send_message_to_user 直发全文 + 最终回复又是全文）。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    event.set_extra(
        "_send_message_to_user_current_session_plain_texts",
        ["你好。世界。继续说一点"],
    )

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert ctx.generate_calls == []
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_normalizes_thinking_tags_when_only_body_was_sent() -> None:
    """已发的是剥离后的正文、结果链还带 think：宿主按原文比对抑制不了，
    至少不能把 think 原文发出去。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    body = "今天天气不错我们出去玩吧"
    result = FakeResult("<think>内部推理</think>" + body)
    event = FakeEvent(result)
    event.set_extra("_send_message_to_user_current_session_plain_texts", [body])

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert ctx.generate_calls == []
    assert [c.text for c in result.chain] == [body]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_but_still_strips_thinking_tags() -> None:
    """让位给宿主内置分段时也要摘掉 think 原文：宿主会把链直接切分发给用户。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    body = "今天天气不错我们出去玩吧"
    result = FakeResult("<think>内部推理</think>" + body)
    event = FakeEvent(result)
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == [body]
    assert ctx.generate_calls == []
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_without_touching_multi_plain_chain() -> None:
    """让位时只有带 think 的链才归一：多 Plain 且无 think 的链不能并成一段。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("今天天气不错")
    result.chain = [Plain("今天天气不错"), Plain("我们出去玩吧")]
    event = FakeEvent(result)
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["今天天气不错", "我们出去玩吧"]
    assert ctx.generate_calls == []
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_to_host_builtin_segmented_reply() -> None:
    """宿主内置分段会在装饰钩子之后再切一次首段：同时开启时必须让位。

    否则段数膨胀、插件边界被销毁，且补发被宿主的逐条间隔拖后。
    """

    class SegmentedContext(FakeContext):
        def get_config(self, umo=None):
            return {"platform_settings": {"segmented_reply": {"enable": True}}}

    ctx = SegmentedContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    text = "你好。世界。继续说一点"
    result = FakeResult(text)
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == [text]
    assert ctx.generate_calls == []
    assert plugin._warned_builtin_segmentation is True  # type: ignore[attr-defined]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_builtin_segmentation_warning_logged_once(monkeypatch) -> None:
    """让位 warning 只记一次：每轮都刷会在长会话里把日志刷满。"""
    from astrbot_plugin_smart_segmentation import main as main_mod

    warnings: list[str] = []
    monkeypatch.setattr(main_mod.logger, "warning", lambda *args, **_k: warnings.append(str(args)))

    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]
    for text in ("今天天气不错我们出去玩吧", "再来一条同样被让位的回复"):
        await plugin.on_decorating_result(FakeEvent(FakeResult(text)))  # type: ignore[arg-type]
    assert len(warnings) == 1
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_keeps_segmenting_where_host_skips_builtin_segmentation() -> None:
    """宿主在 qq_official_webhook 等平台跳过内置分段：这里不该跟着让位。"""

    class SegmentedContext(FakeContext):
        def get_config(self, umo=None):
            return {"platform_settings": {"segmented_reply": {"enable": True}}}

    ctx = SegmentedContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    event.get_platform_name = lambda: "qq_official_webhook"  # type: ignore[attr-defined]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["今天天气不错"]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_skips_model_call_when_stopping() -> None:
    """停止中不要再打一次分段模型：既浪费额度，也会拖住卸载。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    plugin._stopping = True
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert ctx.generate_calls == []
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
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
async def test_sync_loop_reaps_expired_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """交接钩子一直不触发时，只有周期同步能兜底交付：循环里必须 reap。"""
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod
    from astrbot_plugin_smart_segmentation import main as main_mod

    monkeypatch.setattr(main_mod, "STREAMING_SYNC_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)

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
    settings = plugin._get_settings()  # type: ignore[attr-defined]
    assert settings is not None
    plugin._follow_ups.register(  # type: ignore[attr-defined]
        session="sess:1",
        segments=["二", "三"],
        settings=settings,
    )

    await asyncio.sleep(0.06)
    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ctx.sent == [("sess:1", "二\n三")]

    await plugin.terminate()


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
    result = FakeResult("今天天气不错我们出去玩吧")
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
    result = FakeResult("今天天气不错我们出去玩吧")
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

    # 清簿记不能等于丢内容：剩余段要合并补发出去
    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ctx.sent == [("sess:1", "我们出去玩吧")]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_stale_pending_from_previous_round_is_delivered() -> None:
    """同一事件上一轮登记了 pending 但没触发 after_message_sent（宿主判空/第三方插件
    不调）：这一轮要先把它交付掉，而不是被本轮 after_message_sent 认领。"""
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
    first = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(first)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    stale_id = plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]
    assert stale_id

    # 宿主再次进入装饰钩子（同一 event，新 result），而上一轮的 after 钩子没触发
    event._result = FakeResult("新的回复。第二段内容")  # type: ignore[attr-defined]
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]

    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ("sess:1", "我们出去玩吧") in ctx.sent
    assert plugin._follow_ups.pending_id_of(event) != stale_id  # type: ignore[attr-defined]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_skips_round_after_streaming_finished() -> None:
    """流式回复发完后宿主对后续结果一律不发送。

    第三方插件把结果类型改回 LLM_RESULT 时，这里若照常分段并登记补发，剩余段会在
    兜底超时后被当成孤儿重发一遍（用户看到重复残段）。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    event.set_extra("_streaming_finished", True)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert ctx.generate_calls == []
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
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
