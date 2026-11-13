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
from .errors import brief_exception, is_caller_cancelled
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
        self.reap_expired()
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
        self.reap_expired()
        return pending

    def discard(self, pending_id: str) -> None:
        self._pending.pop(pending_id, None)

    def schedule(self, pending: PendingFollowUp) -> None:
        if self._stopping:
            return
        # 同会话的补发必须串行：两个 burst 交错发送会把消息顺序打乱。默认 agent
        # runner 用会话锁把整轮运行串行化，第三方 runner（dify/coze/…）没有，
        # 这里自己保证，不依赖 AstrBot 内部的锁。
        self._track(
            self._run(pending, self._session_tails.get(pending.session)),
            pending.session,
        )
        # 注意：上一轮结束到本轮被唤醒之间有一个迭代宽的守卫空窗（协程返回即
        # 释放守卫，下一任务的 __wakeup 要等下一个事件循环迭代）。落窗内的新
        # 消息会被正常分段而不是整条发送；补发顺序仍由 _session_tails 保证。

    def _track(self, coro: Any, session: str) -> None:
        """登记补发任务并接进同会话串行链。"""
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            # 同步调用路径（没有运行中的事件循环）没有地方执行补发：关掉协程即可，
            # 不能把 RuntimeError 甩给 register/pop 的调用方。
            coro.close()
            logger.debug("智能分段补发任务无法创建：当前没有运行中的事件循环")
            return
        self._session_tails[session] = task

        def _forget(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if self._session_tails.get(session) is finished:
                self._session_tails.pop(session, None)

        self._tasks.add(task)
        task.add_done_callback(_forget)

    def reap_expired(self) -> None:
        """兜底交付过期未被认领的 pending。

        after_message_sent 是唯一的交接点：第三方插件只在 DELIVERED 后调它（主动
        回复插件遇 UNKNOWN 就不调），宿主 RespondStage 判空/去重时也会直接 return。
        首段是破坏性替换，剩余段只留在 pending 里，静默丢弃等于把回复截断，所以
        过期后合并补发一次。
        """
        if not self._pending:
            return
        now = time.monotonic()
        expired = [
            (key, item) for key, item in self._pending.items() if item.expires_at <= now
        ]
        for key, item in expired:
            self._pending.pop(key, None)
            self._deliver_orphan(item)

    def flush_stale(self, pending_id: str) -> None:
        """立即交付没人认领的旧 pending（同一事件进入下一轮时用）。

        对应 bug：上一轮结果链已被替换成首段，而宿主/第三方插件没触发
        after_message_sent，pending 会一直滞留。丢弃等于把回复截断。
        """
        pending = self._pending.pop(str(pending_id or "").strip(), None)
        if pending is not None:
            self._deliver_orphan(pending)

    def _deliver_orphan(self, pending: PendingFollowUp) -> None:
        if self._stopping:
            return
        logger.warning(
            "智能分段:上一轮剩余 %s 段未及时发出,已合并补发,会话: %s",
            len(pending.segments),
            pending.session,
        )
        # 必须排在同会话串行链后面：正常补发正在进行时插进去会把上一轮的 burst
        # 打断（用户看到的消息顺序被打乱）。
        self._track(
            self._run_orphan(pending, self._session_tails.get(pending.session)),
            pending.session,
        )

    async def _run_orphan(
        self,
        pending: PendingFollowUp,
        previous: asyncio.Task[Any] | None = None,
    ) -> None:
        await self._wait_previous(previous)
        try:
            with self._guard_session(pending.session):
                await self._send_merged_remainder(pending, 0)
        except asyncio.CancelledError:
            if is_caller_cancelled():
                raise
            # 这已经是最后一道兜底，没有可再兜的东西：记一条日志，别让它打断
            # 同会话后面排队的任务。
            logger.warning("智能分段合并补发被平台取消,会话: %s", pending.session)

    @classmethod
    async def _wait_previous(cls, previous: asyncio.Task[Any] | None) -> None:
        if previous is None:
            return
        try:
            await previous
        except asyncio.CancelledError:
            if is_caller_cancelled():
                raise
            # 上一轮是被平台取消的：本轮照常发自己的段
        except Exception:
            pass  # 上一轮的异常已由它自己记录，不影响本轮补发

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
        await self._wait_previous(previous)
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
                        # False 的语义是「没找到可用的平台」，合并重试同样发不出去，
                        # 只会把真实原因掩盖成「发送失败」。
                        logger.warning(
                            "智能分段补发失败:该会话没有可用的平台适配器,剩余 %s 段未发送,"
                            "会话: %s。请确认平台适配器已在运行",
                            len(pending.segments) - index,
                            pending.session,
                        )
                        return
        except asyncio.CancelledError:
            if is_caller_cancelled():
                logger.debug(
                    "智能分段后台补发任务被取消（预期路径），会话: %s",
                    pending.session,
                )
                raise
            # 平台适配器被取消时 CancelledError 会穿过 send_message：剩余段不能
            # 跟着一起丢，走合并兜底（任务本身没有被取消，可以继续 await）。
            logger.warning(
                "智能分段补发被平台取消，剩余 %s 段将合并补发，会话: %s",
                len(pending.segments) - index,
                pending.session,
            )
            await self._send_merged_remainder(pending, index)
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
