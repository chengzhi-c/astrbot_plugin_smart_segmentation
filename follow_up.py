"""Pending follow-up segment dispatch after the first message is sent."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.star import Context

from .bounds import PENDING_FOLLOW_UP_TTL_SECONDS
from .errors import brief_exception
from .segmentation import calculate_send_delay
from .settings import SegmentationSettings

_PENDING_EXTRA_KEY = "smart_segmentation_pending_id"
_APPLIED_EXTRA_KEY = "smart_segmentation_applied_result_id"


@dataclass(slots=True)
class PendingFollowUp:
    session: str
    segments: tuple[str, ...]
    delay_base: float
    delay_per_char: float
    delay_max: float
    expires_at: float


def is_platform_send_failure(exc: Exception) -> bool:
    if exc.__class__.__name__ == "ActionFailed":
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    result = getattr(exc, "result", None)
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        retcode = result.get("retcode")
        return status == "failed" or retcode not in (None, 0)
    return False


class FollowUpDispatcher:
    def __init__(
        self,
        context: Context,
        *,
        is_enabled: Callable[[], bool],
    ) -> None:
        self._context = context
        self._is_enabled = is_enabled
        self._pending: dict[str, PendingFollowUp] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._guards: dict[str, int] = {}
        self._stopping = False

    def mark_stopping(self) -> None:
        self._stopping = True

    async def shutdown(self) -> None:
        self._stopping = True
        to_cancel = [task for task in self._tasks if not task.done()]
        for task in to_cancel:
            task.cancel()
        if to_cancel:
            await asyncio.gather(*to_cancel, return_exceptions=True)
        self._tasks.clear()
        self._pending.clear()
        self._guards.clear()

    def is_session_guarded(self, session: str) -> bool:
        normalized = str(session or "").strip()
        return bool(normalized and self._guards.get(normalized, 0))

    # --- event bookkeeping (de-dupe + pending handoff) ---

    def mark_pending(self, event: Any, pending_id: str, result: Any) -> None:
        event.set_extra(_PENDING_EXTRA_KEY, pending_id)
        event.set_extra(_APPLIED_EXTRA_KEY, id(result))

    def is_result_applied(self, event: Any, result: Any) -> bool:
        return str(event.get_extra(_APPLIED_EXTRA_KEY, "") or "") == str(id(result))

    def pending_id_of(self, event: Any) -> str:
        return str(event.get_extra(_PENDING_EXTRA_KEY, "") or "").strip()

    def clear_extras(self, event: Any) -> None:
        event.set_extra(_PENDING_EXTRA_KEY, "")
        event.set_extra(_APPLIED_EXTRA_KEY, "")

    def register(
        self,
        *,
        session: str,
        segments: list[str],
        settings: SegmentationSettings,
    ) -> str:
        self._prune_expired()
        pending_id = uuid4().hex
        self._pending[pending_id] = PendingFollowUp(
            session=session,
            segments=tuple(segments),
            delay_base=settings.delay_base,
            delay_per_char=settings.delay_per_char,
            delay_max=settings.delay_max,
            expires_at=time.monotonic() + PENDING_FOLLOW_UP_TTL_SECONDS,
        )
        return pending_id

    def pop(self, pending_id: str) -> PendingFollowUp | None:
        self._prune_expired()
        return self._pending.pop(pending_id, None)

    def discard(self, pending_id: str) -> None:
        self._pending.pop(pending_id, None)

    def schedule(self, pending: PendingFollowUp) -> None:
        task = asyncio.create_task(self._run(pending))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _prune_expired(self) -> None:
        if not self._pending:
            return
        now = time.monotonic()
        expired = [key for key, item in self._pending.items() if item.expires_at <= now]
        for key in expired:
            self._pending.pop(key, None)

    @contextmanager
    def _guard_session(self, session: str):
        normalized = str(session or "").strip()
        if not normalized:
            yield
            return
        self._guards[normalized] = self._guards.get(normalized, 0) + 1
        try:
            yield
        finally:
            remaining = self._guards.get(normalized, 0) - 1
            if remaining > 0:
                self._guards[normalized] = remaining
            else:
                self._guards.pop(normalized, None)

    async def _run(self, pending: PendingFollowUp) -> None:
        try:
            with self._guard_session(pending.session):
                for segment in pending.segments:
                    if self._stopping or not self._is_enabled():
                        return
                    delay = calculate_send_delay(
                        segment,
                        pending.delay_base,
                        pending.delay_per_char,
                        pending.delay_max,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if self._stopping or not self._is_enabled():
                        return
                    try:
                        sent = await self._context.send_message(
                            pending.session,
                            MessageChain([Plain(segment)]),
                        )
                    except Exception as exc:
                        if is_platform_send_failure(exc):
                            logger.warning(
                                "智能分段补发发送失败，已停止剩余补发，会话: %s，错误: %s",
                                pending.session,
                                brief_exception(exc),
                            )
                            return
                        raise
                    if not sent:
                        logger.warning(
                            "智能分段补发未发送，已停止剩余补发，会话: %s",
                            pending.session,
                        )
                        return
        except asyncio.CancelledError:
            logger.warning("智能分段后台补发任务被取消，会话: %s", pending.session)
            raise
        except Exception as exc:
            if is_platform_send_failure(exc):
                logger.warning(
                    "智能分段后台补发发送失败，已停止补发，会话: %s，错误: %s",
                    pending.session,
                    brief_exception(exc),
                )
                return
            logger.error("智能分段后台补发任务异常: %s", exc, exc_info=True)
