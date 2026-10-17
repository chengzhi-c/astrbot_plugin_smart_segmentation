"""Optional monkey-patch: wrap streaming inside ResultDecorateStage only.

C-light: no platform-specific send_streaming patches (e.g. aiocqhttp).
Single patch target, so no handle abstraction.
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
_ORIGINAL_ATTR = "_smart_segmentation_original_process"
_TARGET_ATTR = "process"
_TARGET_LABEL = "ResultDecorateStage.process"


class StreamingPatchManager:
    """Install/restore ResultDecorateStage.process patch when enabled."""

    def __init__(
        self,
        *,
        owner_token: str,
        get_settings: Callable[[], SegmentationSettings | None],
        is_session_guarded: Callable[[str], bool],
    ) -> None:
        self._owner_token = owner_token
        self._get_settings = get_settings
        self._is_session_guarded = is_session_guarded
        self._stopping = False
        self._desired_enabled = False
        self._holder: Any = None
        self._original: Callable[..., Any] | None = None
        self._installed = False

    def mark_stopping(self) -> None:
        self._stopping = True

    def sync(self, settings: SegmentationSettings | None) -> None:
        if self._stopping:
            settings = None
        want = bool(settings is not None and settings.streaming_compat_enabled)
        if want == self._desired_enabled and want == self._installed:
            if not want or self._owns_current_patch():
                return
            # 补丁已被别的实例接管/还原，但 _installed 仍为 True：不纠正的话
            # 周期同步会永远早退，流式兼容静默失效直到进程重启。
            self._installed = False
        self._desired_enabled = want
        if want:
            self._install()
        else:
            self._restore()

    def _owns_current_patch(self) -> bool:
        holder = self._holder
        if holder is None:
            return False
        current = getattr(holder, _TARGET_ATTR, None)
        return getattr(current, _PATCH_OWNER_ATTR, None) == self._owner_token

    def _install(self) -> None:
        try:
            from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

            self._holder = ResultDecorateStage
            if not self._prepare_install():
                return
            self._apply_patch()
        except Exception as exc:
            logger.warning("智能分段流式兼容补丁启用失败: %s", exc, exc_info=True)

    def _apply_patch(self) -> None:
        original = self._original
        if original is None:
            return
        manager = self

        async def patched_process(stage_self, event: AstrMessageEvent):
            try:
                result = event.get_result()
                if is_streaming_result(result):
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
                    "智能分段流式装饰失败，按未打补丁继续: %s",
                    brief_exception(exc),
                )
            async for item in original(stage_self, event):
                yield item

        setattr(patched_process, _PATCH_OWNER_ATTR, self._owner_token)
        setattr(self._holder, _TARGET_ATTR, patched_process)
        self._installed = True
        logger.info("智能分段流式兼容补丁已启用")

    def _prepare_install(self) -> bool:
        holder = self._holder
        if holder is None:
            return False
        current = getattr(holder, _TARGET_ATTR)
        if getattr(current, _PATCH_OWNER_ATTR, None) == self._owner_token:
            self._installed = True
            return False

        original = getattr(holder, _ORIGINAL_ATTR, None)
        foreign_owner = getattr(current, _PATCH_OWNER_ATTR, None)
        if foreign_owner is not None:
            if original is None:
                logger.warning(
                    "智能分段流式补丁检测到旧补丁但缺少原始 %s，已跳过安装",
                    _TARGET_LABEL,
                )
                return False
            setattr(holder, _TARGET_ATTR, original)
            logger.info("智能分段检测到旧实例残留的 %s，已先恢复", _TARGET_LABEL)
        elif original is not None and current is not original:
            logger.warning(
                "智能分段流式补丁检测到 %s 已被其他代码修改，已跳过安装",
                _TARGET_LABEL,
            )
            return False

        if original is None:
            original = current
            setattr(holder, _ORIGINAL_ATTR, original)
        self._original = original
        return True

    def _restore(self) -> None:
        if not self._installed:
            return
        holder = self._holder
        original = self._original
        try:
            if (
                holder is not None
                and original is not None
                and getattr(getattr(holder, _TARGET_ATTR), _PATCH_OWNER_ATTR, None)
                == self._owner_token
            ):
                setattr(holder, _TARGET_ATTR, original)
                if getattr(holder, _ORIGINAL_ATTR, None) is original:
                    # 必须清掉残留：留着它，之后任何替换 process 的代码都会被
                    # 当成"外来修改"，后续合法安装全部被跳过。
                    delattr(holder, _ORIGINAL_ATTR)
            else:
                logger.debug("智能分段未恢复 %s：当前方法不属于本实例", _TARGET_LABEL)
        except Exception as exc:
            logger.debug("恢复智能分段流式补丁失败 (%s): %s", _TARGET_LABEL, exc)
        finally:
            self._installed = False
