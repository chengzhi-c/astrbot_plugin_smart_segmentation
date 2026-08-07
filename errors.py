"""Small shared error formatting helpers."""

from __future__ import annotations


def brief_exception(exc: BaseException) -> str:
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
