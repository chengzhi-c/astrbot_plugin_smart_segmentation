"""Config loading and typed settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .bounds import (
    ALLOWED_STYLES,
    MAX_DELAY_BASE,
    MAX_DELAY_MAX,
    MAX_DELAY_PER_CHAR,
    MAX_MIN_LENGTH,
    MAX_SEGMENTS,
    MAX_STREAMING_MAX_CHARS,
    MAX_STREAMING_MIN_CHARS,
    MAX_TIMEOUT_SECONDS,
)


@dataclass(slots=True)
class SegmentationSettings:
    provider_id: str = ""
    style: str = "active"
    min_length: int = 12
    max_segments: int = 5
    timeout_seconds: float = 30.0
    delay_base: float = 0.8
    delay_per_char: float = 0.025
    delay_max: float = 2.5
    streaming_compat_enabled: bool = False
    streaming_min_chars: int = 18
    streaming_max_chars: int = 90


_DEFAULTS = SegmentationSettings()


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        # 读配置失败必须降级：配置对象抛异常不能打断 hook 或补丁路径
        pass
    return default


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 手改 JSON 可能写成 1/0 甚至 1.0/0.0：按布尔处理，不要静默回落默认值
        if isinstance(value, float) and math.isnan(value):
            return default
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    # NaN 参与 min/max 钳制会被当成「不大于任何值」而落到下界:手改 JSON 写出 NaN
    # 时按默认值处理,避免配置变成 0.1 秒的瞬时超时。
    return default if math.isnan(parsed) else parsed


def _clamped(
    config: Any,
    key: str,
    default: Any,
    lower: Any,
    upper: Any,
    cast: Any,
) -> Any:
    value = cast(_get_config_value(config, key, default), default)
    return min(upper, max(lower, value))


def _bounded_int(config: Any, key: str, default: int, lower: int, upper: int) -> int:
    return _clamped(config, key, default, lower, upper, as_int)


def _bounded_float(
    config: Any,
    key: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    return _clamped(config, key, default, lower, upper, as_float)


def load_settings(config: Any) -> SegmentationSettings | None:
    """Return settings when enabled, otherwise None."""
    defaults = _DEFAULTS
    if not as_bool(_get_config_value(config, "enabled", True), True):
        return None

    style = str(_get_config_value(config, "style", defaults.style) or defaults.style).strip()
    if style not in ALLOWED_STYLES:
        style = defaults.style

    streaming_min = _bounded_int(
        config,
        "streaming_min_chars",
        defaults.streaming_min_chars,
        1,
        MAX_STREAMING_MIN_CHARS,
    )
    streaming_max = max(
        streaming_min,
        _bounded_int(
            config,
            "streaming_max_chars",
            defaults.streaming_max_chars,
            20,
            MAX_STREAMING_MAX_CHARS,
        ),
    )

    return SegmentationSettings(
        provider_id=str(
            _get_config_value(config, "provider_id", defaults.provider_id) or "",
        ).strip(),
        style=style,
        min_length=_bounded_int(config, "min_length", defaults.min_length, 0, MAX_MIN_LENGTH),
        max_segments=_bounded_int(config, "max_segments", defaults.max_segments, 1, MAX_SEGMENTS),
        timeout_seconds=_bounded_float(
            config,
            "timeout_seconds",
            defaults.timeout_seconds,
            0.1,
            MAX_TIMEOUT_SECONDS,
        ),
        delay_base=_bounded_float(
            config,
            "delay_base",
            defaults.delay_base,
            0.0,
            MAX_DELAY_BASE,
        ),
        delay_per_char=_bounded_float(
            config,
            "delay_per_char",
            defaults.delay_per_char,
            0.0,
            MAX_DELAY_PER_CHAR,
        ),
        delay_max=_bounded_float(
            config,
            "delay_max",
            defaults.delay_max,
            0.0,
            MAX_DELAY_MAX,
        ),
        streaming_compat_enabled=as_bool(
            _get_config_value(
                config,
                "streaming_compat_enabled",
                defaults.streaming_compat_enabled,
            ),
            defaults.streaming_compat_enabled,
        ),
        streaming_min_chars=streaming_min,
        streaming_max_chars=streaming_max,
    )

