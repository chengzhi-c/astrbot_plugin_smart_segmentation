"""Outbound text segmentation use-case (LLM + local fallback)."""

from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .chain_utils import should_segment_text
from .errors import brief_exception
from .segmentation import (
    SegmentationParseError,
    build_segmentation_prompt,
    parse_segments_from_model_output,
    split_text_locally,
)
from .settings import SegmentationSettings


def _is_current_task_cancelling() -> bool:
    """区分"调用方取消"（必须上抛）与 provider 自己抛的 CancelledError（走兜底）。

    astrbot 要求 Python >=3.12，`Task.cancelling()` 必然存在。
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


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
        if provider_id:
            try:
                segments = await asyncio.wait_for(
                    self._segment_with_llm(text, provider_id=provider_id, settings=settings),
                    timeout=settings.timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                if self._stopping or _is_current_task_cancelling():
                    raise
                logger.warning(
                    "智能分段 LLM 调用被取消，改用本地规则兜底: %s",
                    brief_exception(exc),
                )
                segments = split_text_locally(text, max_segments=settings.max_segments)
            except TimeoutError:
                logger.warning(
                    "智能分段 LLM 调用超时（> %.2fs），改用本地规则兜底",
                    settings.timeout_seconds,
                )
                segments = split_text_locally(text, max_segments=settings.max_segments)
            except SegmentationParseError as exc:
                logger.warning(
                    "智能分段模型输出无法解析，改用本地规则兜底: %s",
                    brief_exception(exc),
                )
                segments = split_text_locally(text, max_segments=settings.max_segments)
            except Exception as exc:
                logger.error(
                    "智能分段 LLM 调用失败，改用本地规则兜底: %s",
                    exc,
                    exc_info=True,
                )
                segments = split_text_locally(text, max_segments=settings.max_segments)
        else:
            logger.warning("智能分段未找到可用 provider_id，改用本地规则兜底")
            segments = split_text_locally(text, max_segments=settings.max_segments)

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

        # 缺失该方法时降级为本地规则：AttributeError 不能穿出 hook
        get_current = getattr(self._context, "get_current_chat_provider_id", None)
        if not callable(get_current):
            return ""
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        try:
            provider_id = await get_current(umo=umo)
        except Exception as exc:
            # 调用失败：放弃 provider 解析，本次分段走本地规则兜底
            logger.debug("获取当前会话 provider_id 失败: %s", exc)
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
        response = await self._llm_generate(
            provider_id=provider_id,
            prompt=prompt,
            settings=settings,
        )
        raw_text = str(getattr(response, "completion_text", "") or "").strip()
        if not raw_text:
            return [text]
        return parse_segments_from_model_output(
            raw_text,
            max_segments=settings.max_segments,
        )

    async def _llm_generate(
        self,
        *,
        provider_id: str,
        prompt: str,
        settings: SegmentationSettings,
    ) -> Any:
        generate = getattr(self._context, "llm_generate", None)
        if not callable(generate):
            raise RuntimeError("当前 AstrBot Context 不支持 llm_generate")
        # astrbot >=4.16,<5 的 llm_generate 签名均为
        # (*, chat_provider_id, prompt=None, ..., **kwargs)，下列参数全部收得下。
        return await generate(
            prompt=prompt,
            chat_provider_id=provider_id,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
