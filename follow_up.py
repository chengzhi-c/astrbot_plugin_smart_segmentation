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
from astrbot.api.message_components import At, Plain, Record, Reply
from astrbot.api.star import Context

from .bounds import PENDING_FOLLOW_UP_TTL_SECONDS, SHUTDOWN_TIMEOUT_SECONDS
from .errors import brief_exception, is_caller_cancelled
from .segmentation import calculate_send_delay, split_oversized_segments
from .settings import SegmentationSettings

_PENDING_EXTRA_KEY = "smart_segmentation_pending_id"
_APPLIED_EXTRA_KEY = "smart_segmentation_applied_result"
_FIRST_SEND_DISPATCHER_EXTRA_KEY = "smart_segmentation_first_send_dispatcher"
_FIRST_PLAIN_ATTR = "_smart_segmentation_first_plain"


@dataclass(slots=True)
class PendingFollowUp:
    session: str
    segments: tuple[str, ...]
    delay_base: float
    delay_per_char: float
    delay_max: float
    expires_at: float
    first_send_delivered: bool = False
    first_send_failed: bool = False


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
            # 平台适配器可能吞掉取消，等待必须有上限，否则 terminate 无法完成。
            _, pending = await asyncio.wait(
                to_cancel,
                timeout=SHUTDOWN_TIMEOUT_SECONDS,
            )
            if pending:
                logger.warning(
                    "智能分段补发任务未在关停窗口内退出: %s 个",
                    len(pending),
                )
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
        # 宿主随后会原地加回复前缀，再将此文本复制到 TTS Record.text。
        setattr(
            result,
            _FIRST_PLAIN_ATTR,
            next(
                component for component in result.chain if isinstance(component, Plain)
            ),
        )

    def track_first_send(self, event: Any) -> None:
        dispatcher = event.get_extra(_FIRST_SEND_DISPATCHER_EXTRA_KEY, None)
        event.set_extra(_FIRST_SEND_DISPATCHER_EXTRA_KEY, self)
        if dispatcher is not None:
            return
        original_send = event.send

        async def tracked_send(message: Any) -> Any:
            dispatcher = event.get_extra(_FIRST_SEND_DISPATCHER_EXTRA_KEY)
            pending_id = dispatcher.pending_id_of(event)
            pending = (
                dispatcher._pending.get(pending_id)
                if pending_id
                and dispatcher.is_first_segment_send(event, pending_id, message)
                else None
            )
            result = await original_send(message)
            if pending is not None:
                if result is False:
                    pending.first_send_failed = True
                else:
                    pending.first_send_delivered = True
            return result

        event.send = tracked_send

    def is_first_segment_send(self, event: Any, pending_id: str, message: Any) -> bool:
        if pending_id not in self._pending or not isinstance(message, MessageChain):
            return False
        applied = event.get_extra(_APPLIED_EXTRA_KEY, None)
        if event.get_result() is not applied:
            return False
        if message is applied:
            return True
        first_plain = getattr(applied, _FIRST_PLAIN_ATTR)
        # Record 会从原链抽走（4.16 没有 text）；Image/Node 仍共享组件身份。
        return any(
            (
                not isinstance(component, (At, Reply))
                and any(component is source for source in applied.chain)
            )
            or (
                isinstance(component, Record)
                and getattr(component, "text", first_plain.text) == first_plain.text
            )
            for component in message.chain
        )

    def is_result_applied(self, event: Any, result: Any) -> bool:
        marker = event.get_extra(_APPLIED_EXTRA_KEY, None)
        return marker is not None and marker is result

    def pending_id_of(self, event: Any) -> str:
        return str(event.get_extra(_PENDING_EXTRA_KEY, "") or "").strip()

    def clear_extras(self, event: Any) -> None:
        event.set_extra(_PENDING_EXTRA_KEY, "")
        event.set_extra(_APPLIED_EXTRA_KEY, "")
        # 保留包装标记；同一事件复用包装，热重载时只更新 dispatcher 归属。

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
        # 先取再清理：被显式认领的 pending 即使刚过 TTL 也要交付。
        pending = self._pending.pop(pending_id, None)
        self.reap_expired()
        return pending

    def discard(self, pending_id: str) -> None:
        self._pending.pop(pending_id, None)

    def schedule(self, pending: PendingFollowUp, *, start_index: int = 1) -> None:
        if self._stopping:
            return
        # 同会话的补发必须串行：两个 burst 交错发送会把消息顺序打乱。默认 agent
        # runner 用会话锁把整轮运行串行化，第三方 runner（dify/coze/…）没有，
        # 这里自己保证，不依赖 AstrBot 内部的锁。
        self._track(
            self._run(
                pending,
                self._session_tails.get(pending.session),
                start_index=start_index,
            ),
            pending.session,
        )
        # 注意：上一轮结束到本轮被唤醒之间有一个迭代宽的守卫空窗（协程返回即
        # 释放守卫，下一任务的 __wakeup 要等下一个事件循环迭代）。落窗内的新
        # 消息会被正常分段而不是整条发送；补发顺序仍由 _session_tails 保证。

    def _track(self, coro: Any, session: str) -> None:
        """登记补发任务并接进同会话串行链。"""
        task = asyncio.create_task(coro)
        self._session_tails[session] = task

        def _forget(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if self._session_tails.get(session) is finished:
                self._session_tails.pop(session, None)

        self._tasks.add(task)
        task.add_done_callback(_forget)

    def reap_expired(self) -> None:
        """回收过期 pending；首段已送达时只补发剩余段。

        after_message_sent 是唯一的交接点，但它只代表发送流程结束：发送循环里的异常
        宿主只记 error 日志，随后照样派发，所以不代表送达成功；第三方插件同样不保证
        调它（主动回复插件遇 UNKNOWN 就不调），宿主 RespondStage 判空/去重时也会直接
        return。只对已确认送达的首段补发余段；明确发送失败时恢复整轮，状态未知时丢弃。
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
        """回收没人认领的旧 pending（同一事件进入下一轮时用）。

        对应 bug：上一轮结果链已被替换成首段，而 after_message_sent 根本没被调用
        （例如装饰后被代次/停止检查抑制），旧回复不得因此重新发出。
        """
        pending = self._pending.pop(str(pending_id or "").strip(), None)
        if pending is not None:
            self._deliver_orphan(pending)

    def _deliver_orphan(self, pending: PendingFollowUp) -> None:
        if self._stopping or not (
            pending.first_send_delivered or pending.first_send_failed
        ):
            return
        start_index = 1 if pending.first_send_delivered else 0
        logger.warning(
            "智能分段:上一轮 %s 段未及时发出，已合并补发剩余内容",
            len(pending.segments) - start_index,
        )
        # 必须排在同会话串行链后面：正常补发正在进行时插进去会把上一轮的 burst
        # 打断（用户看到的消息顺序被打乱）。
        self._track(
            self._run_orphan(
                pending,
                self._session_tails.get(pending.session),
                start_index=start_index,
            ),
            pending.session,
        )

    async def _run_orphan(
        self,
        pending: PendingFollowUp,
        previous: asyncio.Task[Any] | None = None,
        *,
        start_index: int = 1,
    ) -> None:
        await self._wait_previous(previous)
        try:
            with self._guard_session(pending.session):
                await self._send_merged_remainder(pending, start_index)
        except asyncio.CancelledError:
            if is_caller_cancelled():
                raise
            # 这已经是最后一道兜底，没有可再兜的东西：记一条日志，别让它打断
            # 同会话后面排队的任务。
            logger.warning("智能分段合并补发被平台取消")

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
        except Exception as exc:
            logger.debug("上一轮补发任务异常已由自身记录: %s", brief_exception(exc))

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
        *,
        start_index: int = 0,
    ) -> None:
        await self._wait_previous(previous)
        index = start_index
        try:
            with self._guard_session(pending.session):
                for index, segment in enumerate(
                    pending.segments[start_index:],
                    start=start_index,
                ):
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
                    sent = await self._context.send_message(
                        pending.session,
                        MessageChain([Plain(segment)]),
                    )
                    if not sent:
                        # False 的语义是「没找到可用的平台」，合并重试同样发不出去，
                        # 只会把真实原因掩盖成「发送失败」。
                        logger.warning(
                            "智能分段补发失败:没有可用的平台适配器，剩余 %s 段未发送。"
                            "请确认平台适配器已在运行",
                            len(pending.segments) - index,
                        )
                        return
        except asyncio.CancelledError:
            if is_caller_cancelled():
                logger.debug(
                    "智能分段后台补发任务被取消（预期路径）",
                )
                raise
            # 平台取消可能发生在提交之后：当前段不重发，其余段继续补发。
            logger.warning(
                "智能分段补发被平台取消，当前段发送状态未知，后续 %s 段将合并补发",
                len(pending.segments) - index - 1,
            )
            await self._send_merged_remainder(pending, index + 1)
        except Exception as exc:
            # send_message 抛错不能证明未提交，合并补发只包含后续段。
            logger.error(
                "智能分段后台补发任务异常，当前段发送状态未知，后续 %s 段将合并补发，错误: %s",
                len(pending.segments) - index - 1,
                brief_exception(exc),
            )
            await self._send_merged_remainder(pending, index + 1)

    async def _send_merged_remainder(
        self,
        pending: PendingFollowUp,
        index: int,
    ) -> None:
        """把 start_index 之后尚未送出的段合并后补发。"""
        remainder = "\n".join(
            segment for segment in pending.segments[index:] if segment.strip()
        )
        if not remainder:
            return
        for segment in split_oversized_segments([remainder]):
            if self._stopping or not self._is_enabled():
                return
            try:
                sent = await self._context.send_message(
                    pending.session,
                    MessageChain([Plain(segment)]),
                )
                if sent is False:
                    logger.warning("智能分段剩余内容合并补发失败:没有可用的平台适配器")
                    return
            except asyncio.CancelledError:
                if is_caller_cancelled():
                    raise
                logger.warning(
                    "智能分段剩余内容合并补发再次被平台取消，当前片段发送状态未知，继续后续片段"
                )
            except Exception as exc:
                logger.error(
                    "智能分段剩余内容合并补发失败，当前片段发送状态未知，继续后续片段，错误: %s",
                    brief_exception(exc),
                )
