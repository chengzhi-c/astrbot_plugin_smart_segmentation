"""Optional monkey-patch: wrap streaming inside ResultDecorateStage only.

C-light: no platform-specific send_streaming patches (e.g. aiocqhttp).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..chain_utils import is_streaming_result
from ..settings import SegmentationSettings
from .wrapper import decorate_streaming_result

_PATCH_OWNER_ATTR = "_smart_segmentation_patch_owner"
_ORIGINAL_ATTR = "_smart_segmentation_original_process"


@dataclass(slots=True)
class PatchHandle:
    label: str
    holder: Any = None
    original: Callable[..., Any] | None = None
    installed: bool = False


class StreamingPatchManager:
    """Install/restore a single ResultDecorateStage.process patch when enabled."""

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
        self._handle = PatchHandle(label="ResultDecorateStage.process")

    def mark_stopping(self) -> None:
        self._stopping = True

    def sync(self, settings: SegmentationSettings | None) -> None:
        if self._stopping:
            settings = None
        want = bool(settings is not None and settings.streaming_compat_enabled)
        if want == self._desired_enabled and (
            (want and self._handle.installed) or (not want and not self._handle.installed)
        ):
            return
        self._desired_enabled = want
        if want:
            self._install()
            return
        self.restore_all()

    def restore_all(self) -> None:
        self._restore(self._handle)

    def _install(self) -> None:
        try:
            from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

            handle = self._handle
            handle.holder = ResultDecorateStage
            if not self._prepare_install(handle, attr="process", original_attr=_ORIGINAL_ATTR):
                return

            original = handle.original
            assert original is not None
            manager = self

            async def patched_process(stage_self, event: AstrMessageEvent):
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
                async for item in original(stage_self, event):
                    yield item

            setattr(patched_process, _PATCH_OWNER_ATTR, self._owner_token)
            ResultDecorateStage.process = patched_process
            handle.installed = True
            logger.info("智能分段流式兼容补丁已启用")
        except Exception as exc:
            logger.warning("智能分段流式兼容补丁启用失败: %s", exc, exc_info=True)

    def _prepare_install(
        self,
        handle: PatchHandle,
        *,
        attr: str,
        original_attr: str,
    ) -> bool:
        holder = handle.holder
        if holder is None:
            return False
        current = getattr(holder, attr)
        if getattr(current, _PATCH_OWNER_ATTR, None) == self._owner_token:
            handle.installed = True
            return False

        original = getattr(holder, original_attr, None)
        foreign_owner = getattr(current, _PATCH_OWNER_ATTR, None)
        if foreign_owner is not None:
            if original is None:
                logger.warning(
                    "智能分段流式补丁检测到旧补丁但缺少原始 %s，已跳过安装",
                    handle.label,
                )
                return False
            setattr(holder, attr, original)
            current = original
            logger.info("智能分段检测到旧实例残留的 %s，已先恢复", handle.label)
        elif original is not None and current is not original:
            logger.warning(
                "智能分段流式补丁检测到 %s 已被其他代码修改，已跳过安装",
                handle.label,
            )
            return False

        if original is None:
            original = current
            setattr(holder, original_attr, original)
        handle.original = original
        return True

    def _restore(self, handle: PatchHandle) -> None:
        if not handle.installed:
            return
        holder = handle.holder
        original = handle.original
        try:
            if (
                holder is not None
                and original is not None
                and getattr(getattr(holder, "process"), _PATCH_OWNER_ATTR, None)
                == self._owner_token
            ):
                holder.process = original
            else:
                logger.debug("智能分段未恢复 %s：当前方法不属于本实例", handle.label)
        except Exception as exc:
            logger.debug("恢复智能分段流式补丁失败 (%s): %s", handle.label, exc)
        finally:
            handle.installed = False
