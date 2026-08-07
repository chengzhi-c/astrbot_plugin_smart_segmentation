"""Outbound text segmentation use-case (LLM + local fallback)."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
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

GenerateFn = Callable[..., Any]


@dataclass(slots=True)
class _ProviderApi:
    get_id: GenerateFn | None = None
    generate: GenerateFn | None = None
    generate_supports_provider: bool = False
    generate_supports_temperature: bool = False
    generate_supports_max_tokens: bool = False
    resolved: bool = False


def _accepts_param(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_current_task_cancelling() -> bool:
    current_task = asyncio.current_task()
    if current_task is None:
        return False
    cancelling = getattr(current_task, "cancelling", None)
    if callable(cancelling):
        return cancelling() > 0
    return current_task.cancelled()


class OutboundSegmenter:
    def __init__(self, context: Context) -> None:
        self._context = context
        self._api = _ProviderApi()
        self._stopping = False

    def mark_stopping(self) -> None:
        self._stopping = True

    def _ensure_api(self) -> _ProviderApi:
        api = self._api
        if api.resolved:
            return api

        get_current = getattr(self._context, "get_current_chat_provider_id", None)
        if callable(get_current):
            api.get_id = get_current

        generate = getattr(self._context, "llm_generate", None)
        if callable(generate):
            api.generate = generate
            api.generate_supports_provider = _accepts_param(generate, "chat_provider_id")
            api.generate_supports_temperature = _accepts_param(generate, "temperature")
            api.generate_supports_max_tokens = _accepts_param(generate, "max_tokens")

        api.resolved = True
        return api

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

        api = self._ensure_api()
        if api.get_id is None:
            return ""
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        try:
            if umo:
                provider_id = await _maybe_await(api.get_id(umo=umo))
            else:
                provider_id = await _maybe_await(api.get_id())
        except TypeError:
            provider_id = None
        except Exception as exc:
            logger.debug("获取当前会话 provider_id 失败: %s", exc)
            return ""
        if provider_id:
            return str(provider_id).strip()
        return ""

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
        api = self._ensure_api()
        generate = api.generate
        if generate is None:
            raise RuntimeError("当前 AstrBot Context 不支持 llm_generate")

        kwargs: dict[str, Any] = {"prompt": prompt}
        if api.generate_supports_provider and provider_id:
            kwargs["chat_provider_id"] = provider_id
        if api.generate_supports_temperature:
            kwargs["temperature"] = settings.temperature
        if api.generate_supports_max_tokens:
            kwargs["max_tokens"] = settings.max_tokens

        try:
            return await _maybe_await(generate(**kwargs))
        except TypeError:
            # One fallback: prompt only (locked AstrBot major line).
            if kwargs.keys() != {"prompt"}:
                return await _maybe_await(generate(prompt=prompt))
            raise
