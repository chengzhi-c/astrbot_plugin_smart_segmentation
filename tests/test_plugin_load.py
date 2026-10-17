"""Real AstrBot environment smoke tests (skipped when astrbot is absent).

These verify the plugin actually registers hooks into AstrBot's handler
registry and instantiates against the real Star base.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("astrbot")

from astrbot.api.star import Star  # noqa: E402
from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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


def test_plugin_is_real_star_and_instantiates() -> None:
    class FakeContext:
        pass

    plugin = SmartSegmentationPlugin(FakeContext(), {"enabled": True})  # type: ignore[arg-type]
    assert isinstance(plugin, Star)
    assert callable(plugin.on_decorating_result)
    assert callable(plugin.after_message_sent)


def test_terminate_clean() -> None:
    class FakeContext:
        pass

    plugin = SmartSegmentationPlugin(FakeContext(), {"enabled": True})  # type: ignore[arg-type]

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


def test_disabled_plugin_no_streaming_manager() -> None:
    class FakeContext:
        pass

    plugin = SmartSegmentationPlugin(FakeContext(), {"enabled": False})  # type: ignore[arg-type]
    assert plugin._streaming is None
    asyncio.run(plugin.terminate())


def test_streaming_patch_real_install_and_restore() -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    class FakeContext:
        pass

    original = ResultDecorateStage.process
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        FakeContext(),
        {"enabled": True, "streaming_compat_enabled": True},
    )
    try:
        assert plugin._streaming is not None
        assert plugin._streaming._installed is True
        assert ResultDecorateStage.process is not original
        assert (
            getattr(
                ResultDecorateStage.process,
                "_smart_segmentation_patch_owner",
                None,
            )
            == plugin._instance_token
        )
    finally:
        plugin._streaming.sync(None)  # type: ignore[union-attr]
        assert ResultDecorateStage.process is original
        asyncio.run(plugin.terminate())


def test_streaming_patch_idempotent_sync() -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    class FakeContext:
        pass

    original = ResultDecorateStage.process
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
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
