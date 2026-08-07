"""Real AstrBot environment smoke tests (skipped when astrbot is absent).

These verify the plugin actually registers hooks into AstrBot's handler
registry and instantiates against the real Star base.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("astrbot")

from astrbot.api.star import Star  # noqa: E402
from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_astrbot_version_upper_bound_locked() -> None:
    """A4 删除了 on_decorating_result 的流式分支，前提是 DecorateStage 对
    STREAMING_RESULT 提前返回（astrbot 4.16-4.x 实测）。

    若上界抬到 5 及以上，必须先重新审视 A4：
    见 test_streaming_integration.py::test_decorate_stage_early_returns_for_streaming
    """
    meta = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    constraint = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in meta.splitlines()
        if line.startswith("astrbot_version:")
    )
    assert "<5" in constraint, (
        f"astrbot_version 上界变更为 {constraint}，A4 的删除依据需重新验证"
    )


def _registry():
    from astrbot.core.star.star_handler import star_handlers_registry

    return star_handlers_registry


def test_hooks_registered_in_astrbot_registry() -> None:
    registry = _registry()
    pending = registry.get_handler_by_full_name(
        "astrbot_plugin_smart_segmentation.main_on_decorating_result",
    )
    after = registry.get_handler_by_full_name(
        "astrbot_plugin_smart_segmentation.main_after_message_sent",
    )
    assert pending is not None
    assert after is not None


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
    asyncio.run(plugin.terminate())
    assert plugin._stopping is True


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
