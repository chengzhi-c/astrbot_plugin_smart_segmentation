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
_APPLIED_EXTRA_KEY = "smart_segmentation_applied_result"


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
    if isinstance(exc, OSError):
        # TimeoutError / ConnectionError 都是 OSError 子类
        return True
    result = getattr(exc, "result", None)
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        retcode = result.get("retcode")
        return status == "failed" or retcode not in (None, 0, "0")
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
        self._session_tails: dict[str, asyncio.Task[Any]] = {}
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
        self._session_tails.clear()
        self._pending.clear()
        self._guards.clear()

    def is_session_guarded(self, session: str) -> bool:
        normalized = str(session or "").strip()
        return bool(normalized and self._guards.get(normalized, 0))

    # --- event bookkeeping (de-dupe + pending handoff) ---

    def mark_pending(self, event: Any, pending_id: str, result: Any) -> None:
        event.set_extra(_PENDING_EXTRA_KEY, pending_id)
        # 直接存对象、按身份判定：存 id() 会在 result 释放后地址被同一 event 上
        # 的新 result 复用时，把新回复误判为已处理而静默跳过分段（agent 多步
        # 循环每步都新建 MessageEventResult）。
        event.set_extra(_APPLIED_EXTRA_KEY, result)

    def is_result_applied(self, event: Any, result: Any) -> bool:
        marker = event.get_extra(_APPLIED_EXTRA_KEY, None)
        return marker is not None and marker is result

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
        # 先取再清理：被显式认领的 pending 即使刚过 TTL 也要交付，否则首段已替换进
        # 原链、剩余段却在此静默丢失（用户拿到被插件自己截断的回复）。
        pending = self._pending.pop(pending_id, None)
        self._prune_expired()
        return pending

    def discard(self, pending_id: str) -> None:
        self._pending.pop(pending_id, None)

    def schedule(self, pending: PendingFollowUp) -> None:
        if self._stopping:
            return
        # 同会话的补发必须串行：两个 burst 交错发送会把消息顺序打乱。默认 agent
        # runner 用会话锁把整轮运行串行化，第三方 runner（dify/coze/…）没有，
        # 这里自己保证，不依赖 AstrBot 内部的锁。
        previous = self._session_tails.get(pending.session)
        task = asyncio.create_task(self._run(pending, previous))
        self._session_tails[pending.session] = task

        def _forget(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if self._session_tails.get(pending.session) is finished:
                self._session_tails.pop(pending.session, None)

        self._tasks.add(task)
        task.add_done_callback(_forget)
        # 注意：上一轮结束到本轮被唤醒之间有一个迭代宽的守卫空窗（协程返回即
        # 释放守卫，下一任务的 __wakeup 要等下一个事件循环迭代）。落窗内的新
        # 消息会被正常分段而不是整条发送；补发顺序仍由 _session_tails 保证。

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

    async def _run(
        self,
        pending: PendingFollowUp,
        previous: asyncio.Task[Any] | None = None,
    ) -> None:
        if previous is not None:
            try:
                await previous
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # 上一轮的异常已由它自己记录，不影响本轮补发
        # 循环开始前就抛异常时（理论路径），合并补发从第一段算起
        index = 0
        try:
            with self._guard_session(pending.session):
                for index, segment in enumerate(pending.segments):
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
                                "智能分段补发发送失败，剩余 %s 段将合并补发，会话: %s，错误: %s",
                                len(pending.segments) - index,
                                pending.session,
                                brief_exception(exc),
                            )
                            await self._send_merged_remainder(pending, index)
                            return
                        raise
                    if not sent:
                        logger.warning(
                            "智能分段补发未发送，剩余 %s 段将合并补发，会话: %s",
                            len(pending.segments) - index,
                            pending.session,
                        )
                        await self._send_merged_remainder(pending, index)
                        return
        except asyncio.CancelledError:
            logger.debug("智能分段后台补发任务被取消（预期路径），会话: %s", pending.session)
            raise
        except Exception as exc:
            # 无论是不是平台类失败，未送出的段都不能丢：首段已被破坏性替换进
            # 原结果链，剩余段只存在于 pending 里。
            logger.error(
                "智能分段后台补发任务异常，剩余 %s 段将合并补发，会话: %s，错误: %s",
                len(pending.segments) - index,
                pending.session,
                brief_exception(exc),
            )
            await self._send_merged_remainder(pending, index)

    async def _send_merged_remainder(
        self,
        pending: PendingFollowUp,
        index: int,
    ) -> None:
        """补发中断时把未送出的段合并为一条发出。

        首段已经被替换进原结果链，剩余段只存在于 pending 里：直接丢弃等于用
        户收到一条被插件自己截断的回复。
        """
        if self._stopping or not self._is_enabled():
            return
        remainder = "\n".join(
            segment for segment in pending.segments[index:] if segment.strip()
        )
        if not remainder:
            return
        try:
            await self._context.send_message(
                pending.session,
                MessageChain([Plain(remainder)]),
            )
        except Exception as exc:
            logger.error(
                "智能分段剩余内容合并补发失败，会话: %s，错误: %s",
                pending.session,
                brief_exception(exc),
            )
