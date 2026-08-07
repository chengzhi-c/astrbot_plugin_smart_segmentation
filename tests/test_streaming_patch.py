from __future__ import annotations

from astrbot_plugin_smart_segmentation.streaming.patches import (
    StreamingPatchManager,
    _PATCH_OWNER_ATTR,
)


class _Target:
    async def process(self, event):
        if False:
            yield None


def test_patch_install_and_restore_owner() -> None:
    calls: list[str] = []

    class Holder:
        async def process(self, event):
            calls.append("original")
            if False:
                yield None

    Holder._smart_segmentation_original_process = None  # type: ignore[attr-defined]

    class Stage:
        process = Holder.process

    def get_settings():
        from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

        return SegmentationSettings(streaming_compat_enabled=True)

    manager = StreamingPatchManager(
        owner_token="owner-token",
        get_settings=get_settings,
        is_session_guarded=lambda _s: False,
    )

    handle = manager._handle
    handle.holder = Stage
    assert manager._prepare_install(
        handle,
        attr="process",
        original_attr="_smart_segmentation_original_process",
    )
    assert handle.original is not None

    async def patched(stage_self, event):
        async for item in handle.original(stage_self, event):  # type: ignore[misc]
            yield item

    setattr(patched, _PATCH_OWNER_ATTR, "owner-token")
    Stage.process = patched
    handle.installed = True

    assert getattr(Stage.process, _PATCH_OWNER_ATTR) == "owner-token"
    manager.restore_all()
    assert handle.installed is False
    assert Stage.process is handle.original


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
    handle = manager._handle
    handle.holder = Stage
    assert manager._prepare_install(
        handle,
        attr="process",
        original_attr="_smart_segmentation_original_process",
    ) is True
    assert Stage.process is original_process
    assert handle.original is original_process


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
    handle = manager._handle
    handle.holder = Stage
    assert manager._prepare_install(
        handle,
        attr="process",
        original_attr="_smart_segmentation_original_process",
    ) is False
    assert handle.installed is False
