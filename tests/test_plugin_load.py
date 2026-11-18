"""Real AstrBot environment smoke tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_astrbot_version_upper_bound_locked() -> None:
    metadata = (Path(__file__).resolve().parents[1] / "metadata.yaml").read_text(
        encoding="utf-8"
    )
    constraint = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in metadata.splitlines()
        if line.startswith("astrbot_version:")
    )
    assert "<5" in constraint, f"流式兼容只验证到 AstrBot 4.x：{constraint}"


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


def test_plugin_is_real_star_and_instantiates(host_plugin) -> None:
    from astrbot.api.star import Star

    class FakeContext:
        pass

    plugin = host_plugin(FakeContext(), {"enabled": True})
    assert isinstance(plugin, Star)
    assert callable(plugin.on_decorating_result)
    assert callable(plugin.after_message_sent)


def test_terminate_clean(host_plugin) -> None:
    class FakeContext:
        pass

    plugin = host_plugin(FakeContext(), {"enabled": True})

    async def scenario() -> None:
        # 先制造真实的在飞状态，否则下面的清理断言是空 vs 空
        pending_id = plugin._follow_ups.register(
            session="sess:1",
            segments=["二", "三"],
            settings=plugin._get_settings(),
        )
        assert pending_id
        plugin._follow_ups.schedule(plugin._follow_ups.pop(pending_id))
        # 再放一条从未被认领的 pending：只有 shutdown 的清理能移除它
        plugin._follow_ups.register(
            session="sess:1",
            segments=["四"],
            settings=plugin._get_settings(),
        )
        await plugin.terminate()

    asyncio.run(scenario())
    assert plugin._stopping is True
    # terminate 必须通知各子系统，否则在飞的分段调用/补发任务不会被拦下
    assert plugin._segmenter._stopping is True
    assert plugin._follow_ups._stopping is True
    assert plugin._follow_ups._pending == {}
    assert plugin._follow_ups._tasks == set()


def test_streaming_patch_idempotent_sync(host_plugin) -> None:
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
        asyncio.run(plugin.terminate())
