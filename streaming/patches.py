"""Optional monkey-patch: wrap streaming inside ResultDecorateStage only.

No platform-specific send_streaming patches (e.g. aiocqhttp): a single patch
target needs no handle abstraction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..chain_utils import is_streaming_result
from ..errors import brief_exception
from ..settings import SegmentationSettings
from .wrapper import decorate_streaming_result

_PATCH_OWNER_ATTR = "_smart_segmentation_patch_owner"
_PATCH_ORIGINAL_ATTR = "_smart_segmentation_patch_original"
_PATCH_INSTANCE_ATTR = "_smart_segmentation_patch_instance"
_PATCH_SEQUENCE_ATTR = "_smart_segmentation_patch_sequence"
_HOST_ORIGINAL_ATTR = "_smart_segmentation_host_original"
_TARGET_ATTR = "process"


class StreamingPatchManager:
    """Install/restore ResultDecorateStage.process patch when enabled.

    同一进程里可能同时存在多个实例（宿主热重载不会保证调用 terminate）。
    安装序号记在宿主类上，按进程单调递增：旧实例遇到序号更大的补丁就退休，
    避免两边来回抢槽位；类上记的真宿主原函数让任何实例都能把槽位还原干净。
    计数存在类上而非模块级：热重载会删掉插件模块，模块级计数器会随之归零。
    """

    def __init__(
        self,
        *,
        owner_token: str,
        get_settings: Callable[[], SegmentationSettings | None],
        is_session_guarded: Callable[[str], bool],
    ) -> None:
        self._owner_token = owner_token
        self._instance_id: int | None = None
        self._get_settings = get_settings
        self._is_session_guarded = is_session_guarded
        self._stopping = False
        self._desired_enabled = False
        self._holder: Any = None
        self._original: Callable[..., Any] | None = None
        self._installed = False
        self._blocked_on: Any = None
        self._retired = False

    def mark_stopping(self) -> None:
        self._stopping = True

    def sync(self, settings: SegmentationSettings | None) -> None:
        if self._stopping:
            settings = None
        want = bool(settings is not None and settings.streaming_compat_enabled)
        if want != self._desired_enabled:
            self._blocked_on = None
        if want == self._desired_enabled and want == self._installed:
            if want:
                if self._owns_current_patch():
                    return
                # 槽位被换掉：可能只是第三方接管，也可能更新的实例已装过。
                if self._superseded():
                    self._retired = True
                self._installed = False
                return
            # 未启用且未安装：本插件的 patched 可能被第三方卸载时挂回来，仍需还原。
            if self._owns_current_patch() and self._recoverable_original() is not None:
                self._restore()
            return
        self._desired_enabled = want
        if want:
            if not self._retired:
                self._install()
        else:
            self._restore()

    def _current(self) -> Any:
        return getattr(self._holder, _TARGET_ATTR, None)

    def _owns_current_patch(self) -> bool:
        return getattr(self._current(), _PATCH_OWNER_ATTR, None) == self._owner_token

    def _recoverable_original(self) -> Callable[..., Any] | None:
        current = self._current()
        return (
            self._original
            or getattr(self._holder, _HOST_ORIGINAL_ATTR, None)
            or getattr(current, _PATCH_ORIGINAL_ATTR, None)
        )

    def _next_sequence(self) -> int:
        """取下一个安装序号。序号记在宿主类上，跨模块重载保持单调。"""
        holder = self._holder
        previous = int(getattr(holder, _PATCH_SEQUENCE_ATTR, 0) or 0)
        if self._instance_id is not None and self._instance_id > previous:
            return self._instance_id
        sequence = previous + 1
        setattr(holder, _PATCH_SEQUENCE_ATTR, sequence)
        self._instance_id = sequence
        return sequence

    def _superseded(self) -> bool:
        """本实例是否已被更新的实例取代。

        两条判据并用：槽位上挂着序号更大的补丁，或宿主类上记录的安装序号已经
        超过本实例——后者覆盖「新实例装过又已禁用」的情况，槽位此时是宿主原函数。
        """
        current = self._current()
        other = getattr(current, _PATCH_INSTANCE_ATTR, None)
        if (
            other is not None
            and self._instance_id is not None
            and other > self._instance_id
        ):
            return True
        latest = int(getattr(self._holder, _PATCH_SEQUENCE_ATTR, 0) or 0)
        return self._instance_id is not None and latest > self._instance_id

    def _install(self) -> None:
        try:
            from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

            self._holder = ResultDecorateStage
            current = self._current()
            owner = getattr(current, _PATCH_OWNER_ATTR, None)
            if owner == self._owner_token:
                if self._original is None:
                    self._original = self._recoverable_original()
                self._installed = self._original is not None
                self._blocked_on = None
                return
            is_ours = owner is not None and hasattr(current, _PATCH_ORIGINAL_ATTR)
            if is_ours:
                if self._superseded():
                    # 更新的实例已接管：本实例退休，避免两边来回抢槽位。
                    self._retired = True
                    self._installed = False
                    self._blocked_on = current
                    return
                host_original = self._recoverable_original()
                if host_original is None:
                    self._installed = False
                    return
                setattr(self._holder, _TARGET_ATTR, host_original)
                logger.debug("检测到旧实例残留的流式补丁,已先恢复")
                current = host_original
                owner = None
            if owner is not None:
                if self._blocked_on is not current:
                    logger.warning("流式兼容未启用:分段环节已由其他插件接管")
                self._blocked_on = current
                self._installed = False
                return
            # 没有 owner 标记的函数只可能是宿主原函数。类上已有记录而当前函数
            # 不是它，说明有别的插件覆盖了槽位（对方没打标记）：不能包装它。
            recorded = getattr(self._holder, _HOST_ORIGINAL_ATTR, None)
            if recorded is not None and current is not recorded:
                if self._blocked_on is not current:
                    logger.warning("流式兼容未启用:分段环节已由其他插件接管")
                self._blocked_on = current
                self._installed = False
                return
            self._original = current
            self._next_sequence()
            self._apply_patch()
        except Exception as exc:
            failure = (type(exc), brief_exception(exc))
            if failure != self._blocked_on:
                self._blocked_on = failure
                logger.warning(
                    "智能分段流式兼容启用失败,本条回复按原样发送: %s",
                    failure[1],
                )

    def _apply_patch(self) -> None:
        original = self._original
        manager = self

        async def patched_process(stage_self, event: AstrMessageEvent):
            # 第三方接管后又把这份闭包挂回来时，本实例可能已被禁用或已停止：
            # 首次被调用就自愈回宿主原函数，避免已卸载的插件长期占着槽位。
            if manager._stopping or not manager._desired_enabled:
                manager._detach_self(patched_process)
                async for item in original(stage_self, event):
                    yield item
                return
            try:
                result = event.get_result()
                if not manager._stopping and is_streaming_result(result):
                    settings = manager._get_settings()
                    if settings is not None and settings.streaming_compat_enabled:
                        decorate_streaming_result(
                            event,
                            result,
                            settings,
                            is_guarded=manager._is_session_guarded(
                                event.unified_msg_origin,
                            ),
                        )
            except Exception as exc:
                # 装饰失败不能让整条消息发不出去（补丁不在 handler 注册表里，
                # 异常会穿过整个管线）：按未打补丁的语义继续。
                logger.warning(
                    "智能分段流式拆段失败,本条回复按原样发送: %s",
                    brief_exception(exc),
                )
            async for item in original(stage_self, event):
                yield item

        setattr(patched_process, _PATCH_OWNER_ATTR, self._owner_token)
        setattr(patched_process, _PATCH_ORIGINAL_ATTR, original)
        setattr(patched_process, _PATCH_INSTANCE_ATTR, self._instance_id)
        setattr(self._holder, _HOST_ORIGINAL_ATTR, original)
        setattr(self._holder, _TARGET_ATTR, patched_process)
        self._installed = True
        self._blocked_on = None
        logger.info("智能分段流式兼容补丁已启用")

    def _detach_self(self, patched: Any) -> None:
        """把被挂回的闭包换回宿主原函数（闭包被调用时发现自身已失效）。"""
        holder = self._holder
        if holder is None or getattr(holder, _TARGET_ATTR, None) is not patched:
            return
        original = getattr(holder, _HOST_ORIGINAL_ATTR, None) or getattr(
            patched, _PATCH_ORIGINAL_ATTR, None
        )
        if original is None:
            return
        setattr(holder, _TARGET_ATTR, original)
        if getattr(holder, _HOST_ORIGINAL_ATTR, None) is original:
            delattr(holder, _HOST_ORIGINAL_ATTR)
        self._installed = False
        logger.debug("已失效的流式补丁被挂回,调用时已还原为宿主原函数")

    def _restore(self) -> None:
        holder = self._holder
        try:
            if holder is not None and self._owns_current_patch():
                target = self._recoverable_original()
                if target is not None:
                    setattr(holder, _TARGET_ATTR, target)
                    if getattr(holder, _HOST_ORIGINAL_ATTR, None) is target:
                        delattr(holder, _HOST_ORIGINAL_ATTR)
            elif self._installed:
                logger.debug("未恢复流式补丁:当前方法不属于本实例")
        except Exception as exc:
            logger.debug("恢复流式兼容补丁失败: %s", brief_exception(exc))
        finally:
            self._installed = False
            self._original = None
