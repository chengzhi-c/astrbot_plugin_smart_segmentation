"""Outbound text segmentation use-case (LLM + local fallback)."""

from __future__ import annotations

import asyncio

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .chain_utils import should_segment_text
from .errors import brief_exception, is_caller_cancelled
from .segmentation import (
    SegmentationParseError,
    build_segmentation_prompt,
    has_visible_text,
    parse_segments_from_model_output,
    same_content,
    split_text_locally,
)
from .settings import SegmentationSettings


class OutboundSegmenter:
    def __init__(self, context: Context) -> None:
        self._context = context
        self._stopping = False

    def mark_stopping(self) -> None:
        self._stopping = True

    async def segment(
        self,
        event: AstrMessageEvent,
        text: str,
        settings: SegmentationSettings,
    ) -> list[str]:
        if not should_segment_text(text, settings):
            logger.debug("智能分段跳过：文本不满足分段条件")
            return []

        provider_id = await self.resolve_provider_id(event, settings)
        if self._stopping:
            return []
        segments: list[str] | None = None
        if provider_id:
            try:
                segments = await asyncio.wait_for(
                    self._segment_with_llm(
                        text, provider_id=provider_id, settings=settings
                    ),
                    timeout=settings.timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                if self._stopping or is_caller_cancelled():
                    raise
                logger.warning(
                    "智能分段模型调用被取消,本次按标点拆分: %s",
                    brief_exception(exc),
                )
            except TimeoutError:
                logger.warning(
                    "智能分段模型超时(> %.2fs),本次按标点拆分",
                    settings.timeout_seconds,
                )
            except SegmentationParseError as exc:
                # 文案由本模块固定构造（解析失败 / 与原文不一致），不含模型原文。
                logger.warning(
                    "智能分段模型输出不可用,本次按标点拆分: %s",
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "智能分段模型调用失败,本次按标点拆分: %s",
                    brief_exception(exc),
                )
        else:
            logger.warning(
                "未找到可用的分段模型,本次按标点拆分。可在插件配置中指定分段模型。"
            )
        if segments is None:
            segments = split_text_locally(text, max_segments=settings.max_segments)

        # 内容校验后才过滤纯不可见段，正文里的 ZWJ 等格式字符保持原样。
        segments = [segment for segment in segments if has_visible_text(segment)]
        if not segments or len(segments) <= 1:
            logger.debug("智能分段跳过：分段结果不足 2 段")
            return []
        return segments

    async def resolve_provider_id(
        self,
        event: AstrMessageEvent,
        settings: SegmentationSettings,
    ) -> str:
        if settings.provider_id:
            return settings.provider_id

        # 缺失该方法时同样落入 except 降级：AttributeError 不能穿出 hook
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        try:
            provider_id = await self._context.get_current_chat_provider_id(umo=umo)
        except Exception as exc:
            # 调用失败：放弃 provider 解析，本次分段走本地规则兜底
            logger.debug("获取当前会话 provider_id 失败: %s", brief_exception(exc))
            return ""
        return str(provider_id).strip() if provider_id else ""

    async def _segment_with_llm(
        self,
        text: str,
        *,
        provider_id: str,
        settings: SegmentationSettings,
    ) -> list[str]:
        prompt = build_segmentation_prompt(text, settings.style, settings.max_segments)
        response = await self._context.llm_generate(
            prompt=prompt,
            chat_provider_id=provider_id,
        )
        raw_text = str(getattr(response, "completion_text", "") or "").strip()
        if not raw_text:
            return [text]
        segments = parse_segments_from_model_output(
            raw_text,
            max_segments=settings.max_segments,
        )
        if not same_content(segments, text):
            # 模型漏字/改写时不能拿它的结果替换原链：交给调用方走本地兜底
            raise SegmentationParseError("分段结果与原文不一致")
        return segments
