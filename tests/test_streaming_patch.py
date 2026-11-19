from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
from astrbot_plugin_smart_segmentation.streaming.patches import (
    _PATCH_OWNER_ATTR,
    StreamingPatchManager,
)


def _settings() -> SegmentationSettings:
    return SegmentationSettings(streaming_compat_enabled=True)


def _manager(token: str = "mine") -> StreamingPatchManager:
    return StreamingPatchManager(
        owner_token=token,
        get_settings=_settings,
        is_session_guarded=lambda _session: False,
    )


def test_foreign_patch_is_left_untouched(caplog) -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process

    async def foreign(stage_self, event):
        if False:
            yield None

    setattr(foreign, _PATCH_OWNER_ATTR, "other")
    ResultDecorateStage.process = foreign
    manager = _manager()
    try:
        manager.sync(_settings())
        manager.sync(_settings())
        assert ResultDecorateStage.process is foreign
        assert manager._installed is False
        assert caplog.text.count("已由其他插件接管") == 1

        ResultDecorateStage.process = original
        manager.sync(_settings())
        assert manager._installed is True
        assert ResultDecorateStage.process is not original
        manager.sync(None)
        assert ResultDecorateStage.process is original
    finally:
        ResultDecorateStage.process = original


def test_takeover_then_third_party_restores_our_patch_still_uninstalls_cleanly() -> (
    None
):
    """第三方接管后又把我们这份 patched 挂回来时，disable 仍必须还原到宿主原函数。

    接管分支若清空 `_original`，`_install` 再次命中自己的 owner 时只剩
    `_installed=True`，`_restore` 就没有可还原的目标，宿主的 process 会永久挂着
    本插件的闭包（插件实例泄漏）。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process

    async def foreign(stage_self, event):
        if False:
            yield None

    setattr(foreign, _PATCH_OWNER_ATTR, "other")
    manager = _manager()
    try:
        manager.sync(_settings())
        ours = ResultDecorateStage.process
        assert ours is not original

        # 第三方接管：保存我们的 patched 再替换。sync 走进接管分支，
        # 旧实现会在这里把 _original 清成 None。
        ResultDecorateStage.process = foreign
        manager.sync(_settings())
        assert manager._installed is False

        # 第三方卸载，把保存的那份 patched 原样还原回来。没有中间 sync：
        # 紧接着 disable/terminate 就必须还原到最初的宿主原函数。
        ResultDecorateStage.process = ours
        manager.sync(None)
        assert ResultDecorateStage.process is original

        # 另一条路径：`_install` 命中 owner == 自己但 `_original` 为空（如本实例
        # disable 后 patched 又被挂回）时，必须从 patched 自身记录的宿主原函数恢复。
        recreated = _manager()
        ResultDecorateStage.process = ours
        recreated.sync(_settings())
        assert recreated._installed is True
        recreated.sync(None)
        assert ResultDecorateStage.process is original
    finally:
        ResultDecorateStage.process = original


def test_disabled_under_foreign_takeover_leaves_the_slot_to_the_third_party() -> None:
    """第三方占槽期间本实例被禁用，不得把对方的补丁改回自己的。

    旧实例把宿主原函数记在类属性上，第三方接管后该记录还在：若 disable 直接按
    `_original` 还原，就会把第三方正在用的补丁踢掉。这里钉住「禁用只清自己的
    状态，槽位归当时的所有者」。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process

    async def foreign(stage_self, event):
        if False:
            yield None

    setattr(foreign, _PATCH_OWNER_ATTR, "other")
    manager = _manager()
    try:
        manager.sync(_settings())
        ResultDecorateStage.process = foreign
        manager.sync(_settings())
        manager.sync(None)
        assert ResultDecorateStage.process is foreign
    finally:
        ResultDecorateStage.process = original


def test_next_instance_recovers_from_a_stale_patch_left_on_the_host() -> None:
    """旧实例未卸载就消失时，新实例要能识别并接管残留补丁。

    宿主热重载会直接构造新实例，旧实例的 terminate 不保证被调用；类属性里记着
    真宿主原函数，新实例据此先还原再装自己，否则 process 会永远停在旧闭包上。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process
    stale = _manager("stale")
    fresh = _manager("fresh")
    try:
        stale.sync(_settings())
        assert ResultDecorateStage.process is not original
        # 旧实例不 terminate，直接构造新实例并同步。
        fresh.sync(_settings())
        assert fresh._installed is True
        fresh.sync(None)
        assert ResultDecorateStage.process is original
        # 旧实例随后再同步两轮都不得把已退休的补丁装回去。
        stale.sync(_settings())
        stale.sync(_settings())
        assert ResultDecorateStage.process is original
    finally:
        ResultDecorateStage.process = original


def test_stale_install_record_does_not_touch_another_plugins_patch() -> None:
    """本插件装过一次后第三方再接管，类上的安装记录不得让我们的新实例越权。

    判定残留只看对方函数带不带本插件标记：没有标记的补丁属于别人，即便类上还
    留着我们上次安装的宿主原函数。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process

    async def foreign(stage_self, event):
        if False:
            yield None

    setattr(foreign, _PATCH_OWNER_ATTR, "other")
    first = _manager("first")
    second = _manager("second")
    try:
        first.sync(_settings())
        ResultDecorateStage.process = foreign
        second.sync(_settings())
        assert ResultDecorateStage.process is foreign, (
            "第三方补丁被本插件的残留记录顶掉"
        )
        assert second._installed is False
    finally:
        ResultDecorateStage.process = original


@pytest.mark.asyncio
async def test_dead_patch_restored_by_third_party_detaches_on_first_call() -> None:
    """已禁用实例的闭包被第三方卸载时挂回，调用一次就应还原为宿主原函数。

    宿主热重载不保证调用 terminate，闭包可能长期留在类上；靠调用时自愈把槽位
    还给宿主，避免已卸载的插件继续占用管线。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process

    async def fake_process(stage_self, event):
        yield "sentinel"

    manager = _manager("dead")
    try:
        ResultDecorateStage.process = fake_process
        manager.sync(_settings())
        dead = ResultDecorateStage.process
        manager.sync(None)
        assert ResultDecorateStage.process is fake_process
        ResultDecorateStage.process = dead  # 第三方把这份闭包挂回来
        event = SimpleNamespace(
            get_result=lambda: SimpleNamespace(),
            unified_msg_origin="fake:group:1",
        )
        assert [item async for item in dead(object(), event)] == ["sentinel"]
        assert ResultDecorateStage.process is fake_process
    finally:
        ResultDecorateStage.process = original


def test_restore_failure_keeps_original_for_retry(caplog) -> None:
    """挂回失败时保留状态，下一次 sync 还能重试，不能把补丁永久留在宿主上。"""
    from astrbot_plugin_smart_segmentation.streaming import patches as patches_mod

    async def original(stage_self, event):
        yield "sentinel"

    async def ours(stage_self, event):
        yield "sentinel"

    setattr(ours, patches_mod._PATCH_OWNER_ATTR, "mine")
    setattr(ours, patches_mod._PATCH_ORIGINAL_ATTR, original)
    setattr(ours, patches_mod._PATCH_INSTANCE_ATTR, 1)

    class HolderMeta(type):
        fail = False

        def __setattr__(cls, name, value):
            if name == patches_mod._TARGET_ATTR and cls.fail:
                raise RuntimeError("boom")
            super().__setattr__(name, value)

    class Holder(metaclass=HolderMeta):
        process = ours

    setattr(Holder, patches_mod._HOST_ORIGINAL_ATTR, original)
    manager = _manager("mine")
    manager._holder = Holder
    manager._installed = True
    manager._original = original

    Holder.fail = True
    manager._restore()
    manager._restore()
    assert Holder.process is ours
    assert manager._installed is True
    assert manager._original is original
    assert caplog.text.count("将在下次同步重试") == 1

    Holder.fail = False
    manager._restore()
    assert Holder.process is original
    assert manager._installed is False
    assert manager._original is None
    assert getattr(Holder, patches_mod._HOST_ORIGINAL_ATTR, None) is None


def test_decoration_failure_does_not_block_the_message(monkeypatch) -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
    from astrbot_plugin_smart_segmentation.streaming import patches as patches_mod

    original = ResultDecorateStage.process

    async def original_process(stage_self, event):
        yield "sentinel"

    ResultDecorateStage.process = original_process

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(patches_mod, "is_streaming_result", lambda _result: True)
    monkeypatch.setattr(patches_mod, "decorate_streaming_result", fail)
    manager = _manager()
    try:
        manager.sync(_settings())

        class Event:
            unified_msg_origin = "sess:1"

            def get_result(self):
                return SimpleNamespace(result_content_type="streaming_result")

        async def scenario():
            return [
                item
                async for item in ResultDecorateStage.process(
                    ResultDecorateStage(),
                    Event(),
                )
            ]

        assert asyncio.run(scenario()) == ["sentinel"]
    finally:
        manager.sync(None)
        ResultDecorateStage.process = original


def test_stopping_patch_delegates_without_decorating(monkeypatch) -> None:
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
    from astrbot_plugin_smart_segmentation.streaming import patches as patches_mod

    original = ResultDecorateStage.process
    calls: list[object] = []

    async def original_process(stage_self, event):
        yield "sentinel"

    ResultDecorateStage.process = original_process
    monkeypatch.setattr(patches_mod, "is_streaming_result", lambda _result: True)
    monkeypatch.setattr(
        patches_mod,
        "decorate_streaming_result",
        lambda *args: calls.append(args),
    )
    manager = _manager()
    try:
        manager.sync(_settings())
        patched = ResultDecorateStage.process
        manager.mark_stopping()

        class Event:
            unified_msg_origin = "sess:1"

            def get_result(self):
                return SimpleNamespace(result_content_type="streaming_result")

        async def scenario():
            return [item async for item in patched(ResultDecorateStage(), Event())]

        assert asyncio.run(scenario()) == ["sentinel"]
        assert calls == []
    finally:
        manager.sync(None)
        ResultDecorateStage.process = original
