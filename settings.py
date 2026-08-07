"""Config loading and typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from .bounds import (
    ALLOWED_STYLES,
    MAX_DELAY_BASE,
    MAX_DELAY_MAX,
    MAX_DELAY_PER_CHAR,
    MAX_MIN_LENGTH,
    MAX_SEGMENTS,
    MAX_STREAMING_MAX_CHARS,
    MAX_STREAMING_MIN_CHARS,
    MAX_TEMPERATURE,
    MAX_TIMEOUT_SECONDS,
    MAX_TOKENS,
)

_CONFIG_KEYS = (
    "enabled",
    "provider_id",
    "style",
    "min_length",
    "max_segments",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "delay_base",
    "delay_per_char",
    "delay_max",
    "streaming_compat_enabled",
    "streaming_min_chars",
    "streaming_max_chars",
)


@dataclass(slots=True)
class SegmentationSettings:
    enabled: bool = True
    provider_id: str = ""
    style: str = "active"
    min_length: int = 12
    max_segments: int = 5
    temperature: float = 0.0
    max_tokens: int = 500
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
    except (TypeError, KeyError, AttributeError) as exc:
        logger.debug("读取智能分段配置 %s 失败: %s", key, exc)
    return default


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
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
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    enabled = as_bool(_get_config_value(config, "enabled", defaults.enabled), defaults.enabled)
    if not enabled:
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
        enabled=enabled,
        provider_id=str(
            _get_config_value(config, "provider_id", defaults.provider_id) or "",
        ).strip(),
        style=style,
        min_length=_bounded_int(config, "min_length", defaults.min_length, 0, MAX_MIN_LENGTH),
        max_segments=_bounded_int(config, "max_segments", defaults.max_segments, 1, MAX_SEGMENTS),
        temperature=_bounded_float(
            config,
            "temperature",
            defaults.temperature,
            0.0,
            MAX_TEMPERATURE,
        ),
        max_tokens=_bounded_int(config, "max_tokens", defaults.max_tokens, 1, MAX_TOKENS),
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


class SettingsLoader:
    """Cache parsed settings while raw config values stay unchanged."""

    def __init__(self, config: Any) -> None:
        self._config = config if config is not None else {}
        self._cached: SegmentationSettings | None = None
        self._fingerprint: tuple[Any, ...] | None = None

    def get(self) -> SegmentationSettings | None:
        fingerprint = tuple(
            _get_config_value(self._config, key, None) for key in _CONFIG_KEYS
        )
        if fingerprint == self._fingerprint:
            return self._cached

        self._cached = load_settings(self._config)
        self._fingerprint = fingerprint
        return self._cached
