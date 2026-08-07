"""AstrBot Smart Segmentation plugin entry (wiring only)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .bounds import STREAMING_SYNC_INTERVAL_SECONDS
from .chain_utils import (
    extract_plain_text_chain,
    is_model_text_result,
    replace_plain_text_chain,
)
from .follow_up import FollowUpDispatcher
from .segmenter import OutboundSegmenter
from .settings import SettingsLoader

if TYPE_CHECKING:
    from .settings import SegmentationSettings
    from .streaming.patches import StreamingPatchManager


class SmartSegmentationPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._stopping = False
        self._instance_token = uuid4().hex
        self._settings = SettingsLoader(self.config)
        self._segmenter = OutboundSegmenter(context)
        self._follow_ups = FollowUpDispatcher(
            context,
            is_enabled=lambda: self._get_settings() is not None,
        )
        self._streaming: StreamingPatchManager | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._sync_streaming(self._get_settings())
        self._start_sync_loop()

    def _start_sync_loop(self) -> None:
        """流式路径上没有可用 hook（DecorateStage/RespondStage 均提前返回），
        补丁的装卸不能只靠消息事件驱动，需要独立的周期同步。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._sync_task = loop.create_task(self._sync_loop())

    async def _sync_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(STREAMING_SYNC_INTERVAL_SECONDS)
                if self._stopping:
                    return
                self._sync_streaming(self._get_settings())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("智能分段流式补丁周期同步失败: %s", exc)

    def _get_settings(self) -> SegmentationSettings | None:
        return self._settings.get()

    def _ensure_streaming(self) -> StreamingPatchManager:
        if self._streaming is None:
            from .streaming.patches import StreamingPatchManager

            self._streaming = StreamingPatchManager(
                owner_token=self._instance_token,
                get_settings=self._get_settings,
                is_session_guarded=self._follow_ups.is_session_guarded,
            )
        return self._streaming

    def _sync_streaming(self, settings: SegmentationSettings | None) -> None:
        enabled = bool(settings is not None and settings.streaming_compat_enabled)
        if not enabled and self._streaming is None:
            return
        self._ensure_streaming().sync(settings)

    @filter.on_decorating_result(priority=-1000)
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        settings = self._get_settings()
        self._sync_streaming(settings)
        if settings is None:
            return
        if self._follow_ups.is_session_guarded(event.unified_msg_origin):
            return

        result = event.get_result()

        if self._follow_ups.is_result_applied(event, result):
            logger.debug("智能分段跳过：当前发送结果已经处理过")
            return

        if result is None or not is_model_text_result(result):
            return

        outbound_text = extract_plain_text_chain(result)
        if not outbound_text:
            return

        segments = await self._segmenter.segment(event, outbound_text, settings)
        if not segments or len(segments) <= 1:
            return

        first_segment = segments[0]
        follow_up_segments = segments[1:]
        if not replace_plain_text_chain(result, first_segment):
            return

        pending_id = self._follow_ups.register(
            session=event.unified_msg_origin,
            segments=follow_up_segments,
            settings=settings,
        )
        self._follow_ups.mark_pending(event, pending_id, result)
        logger.info(
            "智能分段发送前处理完成，共 %s 段，登记 %s 条补发消息",
            len(segments),
            len(follow_up_segments),
        )

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        settings = self._get_settings()
        pending_id = self._follow_ups.pending_id_of(event)
        if settings is None or self._stopping:
            if pending_id:
                self._follow_ups.discard(pending_id)
            self._follow_ups.clear_extras(event)
            return

        if not pending_id:
            self._follow_ups.clear_extras(event)
            return

        pending = self._follow_ups.pop(pending_id)
        self._follow_ups.clear_extras(event)
        if pending is None or not pending.segments:
            return

        self._follow_ups.schedule(pending)

    async def terminate(self) -> None:
        self._stopping = True
        self._segmenter.mark_stopping()
        self._follow_ups.mark_stopping()
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        if self._streaming is not None:
            self._streaming.mark_stopping()
        await self._follow_ups.shutdown()
        if self._streaming is not None:
            self._streaming.sync(None)
