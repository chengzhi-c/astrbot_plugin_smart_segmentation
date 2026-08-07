"""Integration guard: patch must wrap real STREAMING_RESULT via real Stage.

This is the machine-checked form of the probe that justified rejecting O-1
and applying A4. If AstrBot removes the DecorateStage early-return for
STREAMING_RESULT, or changes the patch target shape, this test goes red.
"""

from __future__ import annotations

import pytest

pytest.importorskip("astrbot.core.pipeline.result_decorate.stage")

from astrbot.api.event import MessageChain, MessageEventResult, ResultContentType  # noqa: E402
from astrbot.api.message_components import Plain  # noqa: E402
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage  # noqa: E402

from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin  # noqa: E402

_ORIGINAL_ATTR = "_smart_segmentation_original_process"


class _FakeContext:
    async def send_message(self, session, chain):
        return True


class _FakeEvent:
    def __init__(self, result) -> None:
        self._result = result
        self.unified_msg_origin = "test:session"
        self.extra: dict = {}
        self.plugins_name = None

    def get_result(self):
        return self._result

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value) -> None:
        self.extra[key] = value


async def _dummy_stream():
    yield MessageChain([Plain("第一句。")])
    yield MessageChain([Plain("第二句。")])


def _make_streaming_result():
    result = MessageEventResult()
    result.chain = [Plain("placeholder")]
    result.result_content_type = ResultContentType.STREAMING_RESULT
    result.async_stream = _dummy_stream()
    return result


@pytest.fixture(autouse=True)
def _restore_stage_process():
    original = ResultDecorateStage.process
    saved_attr = getattr(ResultDecorateStage, _ORIGINAL_ATTR, None)
    yield
    ResultDecorateStage.process = original
    if saved_attr is None:
        if hasattr(ResultDecorateStage, _ORIGINAL_ATTR):
            delattr(ResultDecorateStage, _ORIGINAL_ATTR)
    else:
        setattr(ResultDecorateStage, _ORIGINAL_ATTR, saved_attr)


@pytest.mark.asyncio
async def test_decorate_stage_early_returns_for_streaming() -> None:
    """A4 的前提：真实 DecorateStage 对 STREAMING_RESULT 提前返回。

    若此断言失败，说明 AstrBot 改了行为，main.py 里被删除的流式分支需要复原。
    """
    stage = ResultDecorateStage()
    result = _make_streaming_result()
    event = _FakeEvent(result)
    original_stream = result.async_stream

    consumed = [item async for item in stage.process(event)]

    assert consumed == []
    assert result.async_stream is original_stream


@pytest.mark.asyncio
async def test_patch_wraps_real_streaming_result() -> None:
    """O-1 否决的前提：补丁路径是流式包装的唯一生效路径。"""
    plugin = SmartSegmentationPlugin(
        _FakeContext(),  # type: ignore[arg-type]
        {"enabled": True, "streaming_compat_enabled": True},
    )
    try:
        assert plugin._streaming is not None
        assert plugin._streaming._installed is True

        result = _make_streaming_result()
        event = _FakeEvent(result)
        original_stream = result.async_stream

        async for _ in ResultDecorateStage.process(ResultDecorateStage(), event):
            pass

        assert result.async_stream is not original_stream
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_unpatched_leaves_streaming_untouched() -> None:
    """卸载补丁后流式结果原样通过（对照组，A4 删除后的世界）。"""
    plugin = SmartSegmentationPlugin(
        _FakeContext(),  # type: ignore[arg-type]
        {"enabled": True, "streaming_compat_enabled": True},
    )
    plugin._streaming.sync(None)

    try:
        result = _make_streaming_result()
        event = _FakeEvent(result)
        original_stream = result.async_stream

        async for _ in ResultDecorateStage.process(ResultDecorateStage(), event):
            pass

        assert result.async_stream is original_stream
    finally:
        await plugin.terminate()
