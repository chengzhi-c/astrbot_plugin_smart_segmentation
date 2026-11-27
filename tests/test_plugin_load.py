"""Real AstrBot environment smoke tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def host_plugin():
    from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin

    return SmartSegmentationPlugin


def test_hooks_registered_in_astrbot_registry() -> None:
    """两个钩子必须注册进 AstrBot 的真实 handler registry。

    按事件类型 + 模块路径后缀匹配，而不是硬编码 handler_full_name：完整名字
    的格式（`f"{module}_{name}"`）和模块路径的导入方式都属于 AstrBot 内部约
    定，写死会让测试随导入方式变化而脆弱。
    """
    from astrbot.core.star.star_handler import EventType, star_handlers_registry
    from astrbot_plugin_smart_segmentation import main as main_mod

    module_suffix = main_mod.__name__

    def _registered(event_type: EventType) -> set[str]:
        return {
            handler.handler_name
            for handler in star_handlers_registry.get_handlers_by_event_type(
                event_type,
            )
            if handler.handler_module_path.endswith(module_suffix)
        }

    assert "on_decorating_result" in _registered(EventType.OnDecoratingResultEvent)
    assert "after_message_sent" in _registered(EventType.OnAfterMessageSentEvent)


@pytest.mark.asyncio
async def test_terminate_clean(host_plugin) -> None:
    class FakeContext:
        pass

    plugin = host_plugin(FakeContext(), {"enabled": True})

    # 在飞任务与未认领的 pending 都应在关停时回收。
    pending_id = plugin._follow_ups.register(
        session="sess:1",
        segments=["二", "三"],
        settings=plugin._get_settings(),
    )
    assert pending_id
    plugin._follow_ups.schedule(plugin._follow_ups.pop(pending_id))
    plugin._follow_ups.register(
        session="sess:1",
        segments=["四"],
        settings=plugin._get_settings(),
    )
    await plugin.terminate()
    assert plugin._stopping is True
    # terminate 必须通知各子系统，否则在飞的分段调用/补发任务不会被拦下
    assert plugin._segmenter._stopping is True
    assert plugin._follow_ups._stopping is True
    assert plugin._follow_ups._pending == {}
    assert plugin._follow_ups._tasks == set()


@pytest.mark.asyncio
async def test_streaming_patch_idempotent_sync(host_plugin) -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    class FakeContext:
        pass

    original = ResultDecorateStage.process
    plugin = host_plugin(
        FakeContext(),
        {"enabled": True, "streaming_compat_enabled": True},
    )
    try:
        first = ResultDecorateStage.process
        plugin._sync_streaming(plugin._get_settings())
        assert ResultDecorateStage.process is first
    finally:
        plugin._streaming.sync(None)  # type: ignore[union-attr]
        assert ResultDecorateStage.process is original
        await plugin.terminate()


@pytest.mark.asyncio
async def test_host_config_probe_failure_warns_once(host_plugin, caplog) -> None:
    """宿主配置读取失败只告警一次；重复告警会随每条回复刷屏。"""

    class RaisingContext:
        def get_config(self, _umo):
            raise RuntimeError("boom")

    plugin = host_plugin(RaisingContext(), {"enabled": True})
    try:

        class Event:
            unified_msg_origin = "sess:1"

            @staticmethod
            def get_platform_name():
                return "test"

        assert plugin._host_segmented_reply_enabled(Event()) is False
        assert plugin._host_segmented_reply_enabled(Event()) is False
        assert caplog.text.count("读取宿主分段回复配置失败") == 1
    finally:
        await plugin.terminate()
