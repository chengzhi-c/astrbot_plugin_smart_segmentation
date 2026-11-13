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
    """一行异常摘要。只在 except 块里调用，它自己绝不能再抛。"""
    try:
        return _describe(exc)
    except BaseException:
        return exc.__class__.__name__


def _describe(exc: BaseException) -> str:
    result = getattr(exc, "result", None)
    if isinstance(result, dict):
        retcode = result.get("retcode", "")
        message = result.get("wording") or result.get("message") or str(exc)
        message = " ".join(str(message).split())
        if len(message) > 180:
            message = f"{message[:177]}..."
        return f"{exc.__class__.__name__}(retcode={retcode}, message={message})"
    text = " ".join(str(exc).split()).strip()
    if len(text) > 180:
        text = f"{text[:177]}..."
    if not text:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"
