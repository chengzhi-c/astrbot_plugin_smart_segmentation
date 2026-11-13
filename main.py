"""AstrBot Smart Segmentation plugin entry (wiring only)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
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
from .settings import SegmentationSettings, load_settings

if TYPE_CHECKING:
    from .streaming.patches import StreamingPatchManager


# 宿主 send_message_to_user 写进事件 extra 的「已发往当前会话的纯文本」。
# RespondStage 用它抑制重复回复；换链后 get_plain_text() 只剩首段，比对不再命中。
_SENT_TEXTS_EXTRA_KEY = "_send_message_to_user_current_session_plain_texts"

# 宿主 RespondStage 在流式回复结束后置位：之后再出现的结果一律不再发送。
_STREAMING_FINISHED_EXTRA_KEY = "_streaming_finished"

# 宿主自己对这几个平台跳过内置分段回复（result_decorate/stage.py 的平台判断）。
_HOST_SEGMENT_EXEMPT_PLATFORMS = (
    "qq_official_webhook",
    "weixin_official_account",
    "dingtalk",
)


class SmartSegmentationPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._stopping = False
        self._instance_token = uuid4().hex
        self._segmenter = OutboundSegmenter(context)
        self._follow_ups = FollowUpDispatcher(
            context,
            is_enabled=lambda: self._get_settings() is not None,
        )
        self._streaming: StreamingPatchManager | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._warned_builtin_segmentation = False
        self._sync_streaming(self._get_settings())
        self._start_sync_loop()

    def _start_sync_loop(self) -> None:
        """流式路径上没有可用 hook（DecorateStage/RespondStage 均提前返回），
        补丁的装卸不能只靠消息事件驱动，需要独立的周期同步。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("智能分段未启动流式补丁周期同步：当前没有运行中的事件循环")
            return
        self._sync_task = loop.create_task(self._sync_loop())

    async def _sync_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(STREAMING_SYNC_INTERVAL_SECONDS)
                if self._stopping:
                    return
                self._sync_streaming(self._get_settings())
                # 顺带兜底交付过期未认领的补发段：交接钩子没触发时（第三方插件只
                # 在 DELIVERED 后调、宿主判空跳过）没有别的地方能回收它。
                self._follow_ups.reap_expired()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("智能分段流式补丁周期同步失败: %s", exc)

    def _get_settings(self) -> SegmentationSettings | None:
        return load_settings(self.config)

    @staticmethod
    def _host_already_sent_texts(event: AstrMessageEvent) -> list[str]:
        texts = event.get_extra(_SENT_TEXTS_EXTRA_KEY, None)
        if not isinstance(texts, (list, tuple)):
            return []
        return [str(text or "").strip() for text in texts]

    @staticmethod
    def _plain_text_of(result: Any) -> str:
        getter = getattr(result, "get_plain_text", None)
        if not callable(getter):
            return ""
        try:
            return str(getter() or "").strip()
        except Exception:
            return ""

    def _host_segmented_reply_enabled(self, event: AstrMessageEvent) -> bool:
        """宿主内置分段回复开启时让位。

        宿主在装饰钩子之后对我们换好的首段再按标点切一次（它的长度阈值只决定宿主
        自己切不切，管不到我们换链之后的短首段），段数膨胀、分段边界被销毁、补发还
        被宿主的逐条间隔拖后。宿主没有单条结果的开关，只能在调用侧整体让位。
        """
        get_platform_name = getattr(event, "get_platform_name", None)
        if callable(get_platform_name):
            if get_platform_name() in _HOST_SEGMENT_EXEMPT_PLATFORMS:
                return False
        get_config = getattr(self.context, "get_config", None)
        if not callable(get_config):
            return False
        try:
            config = get_config(event.unified_msg_origin)
            platform_settings = config.get("platform_settings", {}) or {}
            segmented = platform_settings.get("segmented_reply", {}) or {}
        except Exception:
            return False
        return bool(isinstance(segmented, dict) and segmented.get("enable"))

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
        if self._stopping:
            # terminate 之后仍可能有 hook 在途：此时不能再装补丁，否则会重新挂上
            # 一个没人卸载的闭包（同步任务已被取消）。
            return
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
            # 上一轮补发进行中，本轮不分段。同时清掉可能残留的旧簿记：若上一轮
            # 登记了 pending 但 RespondStage 被跳过，不清会让本轮
            # after_message_sent 把那条 pending 当成本轮的发出去。
            # 注意：剩余段排在串行链后面，而本轮整条回复由宿主直接发出，不排在链里，
            # 因此旧内容可能落在这条新回复之后——不能为了顺序去阻塞回复。
            self._follow_ups.flush_stale(self._follow_ups.pending_id_of(event))
            self._follow_ups.clear_extras(event)
            return

        result = event.get_result()

        if self._follow_ups.is_result_applied(event, result):
            logger.debug("智能分段跳过：当前发送结果已经处理过")
            return

        # 本轮结果没登记过，残留簿记只可能属于上一轮。上一轮可能因 RespondStage
        # 判空 / 抽走 Reply-At 后链为空而根本没发出去，after_message_sent 不触发，
        # pending 会一直挂在事件上；不清掉，本轮发送成功后就会认领它，把上一轮
        # 剩下的段补发到这一轮（内容串轮）。
        stale_id = self._follow_ups.pending_id_of(event)
        if stale_id:
            self._follow_ups.flush_stale(stale_id)
            self._follow_ups.clear_extras(event)

        if result is None or not is_model_text_result(result):
            return

        if event.get_extra(_STREAMING_FINISHED_EXTRA_KEY, False):
            # 流式回复已经发完，第三方插件又把结果改回 LLM_RESULT：宿主 RespondStage
            # 会直接 return（不发、也不触发 after_message_sent），此时分段只会让剩余段
            # 在兜底超时后被当成孤儿重发一遍。
            logger.debug("智能分段跳过：本轮流式回复已经结束")
            return

        outbound_text = extract_plain_text_chain(result)
        if not outbound_text:
            return

        sent_texts = self._host_already_sent_texts(event)
        if sent_texts:
            plain_text = self._plain_text_of(result)
            if plain_text and plain_text in sent_texts:
                # 原文命中：宿主内置分段关闭时它自己的比对也会命中，交给它抑制。不能动链——
                # 换掉文本它的比对对象就变了，抑制反而失效；换链后只剩首段也是同样问题。
                logger.debug("智能分段跳过：该文本已由 send_message_to_user 发出")
                return
            if outbound_text in sent_texts:
                # 只有剥离 think 后的正文命中：宿主比对的是未剥离的原文，抑制不了。
                # 把链归一成正文，至少不会把 think 原文发给用户；让位分支也要归一。
                logger.debug("智能分段跳过：正文已由 send_message_to_user 发出")
                replace_plain_text_chain(result, outbound_text)
                return

        if self._host_segmented_reply_enabled(event):
            if not self._warned_builtin_segmentation:
                self._warned_builtin_segmentation = True
                logger.warning(
                    "已开启 AstrBot 内置分段回复,本次不启用智能分段"
                    "(两处同时切分会把消息条数放大、顺序打乱)。"
                    "请在 AstrBot 配置 → 平台设置 → 分段回复 中关闭它。",
                )
            plain_text = self._plain_text_of(result)
            if "<think" in plain_text.lower():
                # 让位时也要摘掉 think 原文：宿主会直接按链文本切分发给用户。
                # 原文命中「已发文本」的情形已在上面 return（改链会让宿主抑制失效）。
                replace_plain_text_chain(result, outbound_text)
            return

        if self._stopping:
            # 停止中不要再打一次分段模型：既浪费额度，也会拖住卸载。
            return
        segments = await self._segmenter.segment(event, outbound_text, settings)
        if not segments:
            return
        if self._stopping:
            # terminate 可能落在分段调用期间：此时替换链会被 after_message_sent
            # 丢弃补发，用户只看到首段。停止中保持原样发送。
            logger.debug("智能分段跳过：插件正在停止")
            return

        first_segment = segments[0]
        follow_up_segments = segments[1:]
        # 先换链再登记是刻意顺序：登记成功但换链失败会造成补发 + 原链全文
        # 重复发送，比当前的失败模式（补发缺失）更差。
        if not replace_plain_text_chain(result, first_segment):
            return

        pending_id = self._follow_ups.register(
            session=event.unified_msg_origin,
            segments=follow_up_segments,
            settings=settings,
        )
        self._follow_ups.mark_pending(event, pending_id, result)
        logger.debug(
            "智能分段:共 %s 段,其中 %s 段稍后补发",
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
        try:
            if self._sync_task is not None and not self._sync_task.done():
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            if self._streaming is not None:
                self._streaming.mark_stopping()
            await self._follow_ups.shutdown()
        finally:
            # 补丁必须在本方法退出前卸掉：异常路径若跳过它，补丁会永久留在
            # ResultDecorateStage 上，插件实例也随之无法回收。
            if self._streaming is not None:
                self._streaming.sync(None)
