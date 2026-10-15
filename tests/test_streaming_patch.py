"""Unit tests for the streaming patch recovery state machine.

Real install/restore against the actual ResultDecorateStage is covered by
test_plugin_load.py and test_streaming_integration.py.
"""

from __future__ import annotations

from astrbot_plugin_smart_segmentation.streaming.patches import (
    StreamingPatchManager,
    _PATCH_OWNER_ATTR,
)


def test_prepare_install_restores_foreign_then_installs() -> None:
    class Stage:
        pass

    async def original_process(stage_self, event):
        if False:
            yield None

    def old_patch(stage_self, event):
        if False:
            yield None

    setattr(old_patch, _PATCH_OWNER_ATTR, "old-token")
    Stage.process = old_patch
    Stage._smart_segmentation_original_process = original_process  # type: ignore[attr-defined]

    def get_settings():
        from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

        return SegmentationSettings(streaming_compat_enabled=True)

    manager = StreamingPatchManager(
        owner_token="mine",
        get_settings=get_settings,
        is_session_guarded=lambda _s: False,
    )
    manager._holder = Stage
    assert manager._prepare_install() is True
    assert Stage.process is original_process
    assert manager._original is original_process


def test_prepare_install_skips_when_foreign_without_original() -> None:
    class Stage:
        pass

    def old_patch(stage_self, event):
        if False:
            yield None

    setattr(old_patch, _PATCH_OWNER_ATTR, "old-token")
    Stage.process = old_patch
    Stage._smart_segmentation_original_process = None  # type: ignore[attr-defined]

    def get_settings():
        from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

        return SegmentationSettings(streaming_compat_enabled=True)

    manager = StreamingPatchManager(
        owner_token="mine",
        get_settings=get_settings,
        is_session_guarded=lambda _s: False,
    )
    manager._holder = Stage
    assert manager._prepare_install() is False
    assert manager._installed is False
