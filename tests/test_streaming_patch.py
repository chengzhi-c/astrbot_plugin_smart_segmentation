"""Unit tests for the streaming patch recovery state machine.

Real install/restore against the actual ResultDecorateStage is covered by
test_plugin_load.py and test_streaming_integration.py.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from astrbot_plugin_smart_segmentation.streaming.patches import (
    StreamingPatchManager,
    _ORIGINAL_ATTR,
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


def test_restore_clears_original_attr_and_allows_reinstall() -> None:
    """恢复后必须清掉 _ORIGINAL_ATTR：留着它，之后任何替换 process 的代码都会
    被当成"外来修改"，后续合法安装全部被静默跳过。"""
    class Stage:
        pass

    async def original_process(stage_self, event):
        if False:
            yield None

    Stage.process = original_process

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
    manager._apply_patch()
    assert manager._installed is True
    assert Stage.process is not original_process

    manager.sync(None)
    assert Stage.process is original_process
    assert not hasattr(Stage, _ORIGINAL_ATTR)

    # 第三方随后替换了 process：再开启流式兼容必须能正常装回去
    # （直接走 _prepare_install/_apply_patch：sync→_install 会自行解析真实 holder）
    async def foreign_process(stage_self, event):
        if False:
            yield None

    Stage.process = foreign_process
    assert manager._prepare_install() is True
    manager._apply_patch()
    assert manager._installed is True
    assert getattr(Stage.process, _PATCH_OWNER_ATTR, None) == "mine"
    assert manager._original is foreign_process

    manager.sync(None)
    assert Stage.process is foreign_process
    assert not hasattr(Stage, _ORIGINAL_ATTR)


def test_sync_reinstalls_when_ownership_lost() -> None:
    """补丁被别的实例接管/还原后，sync 必须重新装上而不是永远早退。

    否则 `_installed` 仍为 True，周期同步永远早退，流式兼容静默失效。
    """
    class Stage:
        pass

    def unowned(stage_self, event):
        if False:
            yield None

    def get_settings():
        from astrbot_plugin_smart_segmentation.settings import SegmentationSettings

        return SegmentationSettings(streaming_compat_enabled=True)

    calls: list[str] = []
    manager = StreamingPatchManager(
        owner_token="mine",
        get_settings=get_settings,
        is_session_guarded=lambda _s: False,
    )
    manager._holder = Stage
    manager._install = lambda: calls.append("install")  # type: ignore[method-assign]
    manager._restore = lambda: calls.append("restore")  # type: ignore[method-assign]

    # 自以为装着，但当前 process 不属于本实例 → 必须重新安装
    manager._desired_enabled = True
    manager._installed = True
    Stage.process = unowned
    manager.sync(get_settings())
    assert calls == ["install"]
    assert manager._installed is False  # 交给 _install 重建

    # 确实属于本实例 → 幂等早退，不重复安装
    calls.clear()
    manager._installed = True

    def owned(stage_self, event):
        if False:
            yield None

    setattr(owned, _PATCH_OWNER_ATTR, "mine")
    Stage.process = owned
    manager.sync(get_settings())
    assert calls == []


def test_decoration_failure_does_not_block_the_message(monkeypatch) -> None:
    """装饰失败不能让整条消息发不出去：补丁不在 handler 注册表里，异常会穿过
    整个管线，让 RespondStage 根本不执行。"""
    from astrbot_plugin_smart_segmentation.streaming import patches as patches_mod

    class Stage:
        pass

    async def original_process(stage_self, event):
        yield "sentinel"

    Stage.process = original_process

    def boom(*args, **kwargs):
        raise RuntimeError("decorate exploded")

    # 强制走进流式分支，并让装饰本身爆炸
    monkeypatch.setattr(patches_mod, "is_streaming_result", lambda _result: True)
    monkeypatch.setattr(patches_mod, "decorate_streaming_result", boom)

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
    manager._apply_patch()
    assert manager._installed is True

    class Event:
        unified_msg_origin = "sess:1"

        def get_result(self):
            return SimpleNamespace(result_content_type="streaming_result")

    async def scenario():
        return [item async for item in Stage.process(Stage(), Event())]

    assert asyncio.run(scenario()) == ["sentinel"]  # 原 stage 产出照常透传

    manager.sync(None)
    assert Stage.process is original_process
