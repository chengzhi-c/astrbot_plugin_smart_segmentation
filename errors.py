"""Small shared error formatting helpers."""

from __future__ import annotations

import asyncio


def is_caller_cancelled() -> bool:
    """当前任务是否真的处于取消流程中。

    子协程（provider、平台适配器、子任务）抛出的 CancelledError 不会让当前任务进入
    cancelling 状态：只有那种情况才该走兜底，真正的调用方取消必须原样上抛。
    宿主要求 Python >=3.12，Task.cancelling() 必然存在。
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def brief_exception(exc: BaseException) -> str:
    """Return a log-safe exception summary without message or URL contents."""
    return type(exc).__name__
