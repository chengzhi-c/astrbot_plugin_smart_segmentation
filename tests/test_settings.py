from __future__ import annotations

from astrbot_plugin_smart_segmentation.settings import load_settings
from astrbot_plugin_smart_segmentation.bounds import (
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


class TestLoadSettings:
    def test_disabled_returns_none(self) -> None:
        assert load_settings({"enabled": False}) is None

    def test_defaults_and_bounds(self) -> None:
        settings = load_settings({"enabled": True, "max_segments": 999, "style": "nope"})
        assert settings is not None
        assert settings.max_segments == 20
        assert settings.style == "active"

    def test_bool_string(self) -> None:
        settings = load_settings({"enabled": "true", "streaming_compat_enabled": "off"})
        assert settings is not None
        assert settings.streaming_compat_enabled is False

    def test_infinite_config_value_falls_back_to_default(self) -> None:
        """JSON 的 1e999/Infinity 解析成 inf：int(inf) 抛 OverflowError，
        曾让插件加载失败、之后每条消息的 hook 都抛异常。"""
        settings = load_settings({"enabled": True, "max_tokens": float("inf")})
        assert settings is not None
        assert settings.max_tokens == 500
        # as_float(inf) 不抛异常，由钳制收敛到上界
        settings = load_settings({"enabled": True, "timeout_seconds": float("inf")})
        assert settings is not None
        assert settings.timeout_seconds == MAX_TIMEOUT_SECONDS

    def test_streaming_max_not_below_min(self) -> None:
        settings = load_settings(
            {
                "enabled": True,
                "streaming_min_chars": 80,
                "streaming_max_chars": 20,
            },
        )
        assert settings is not None
        assert settings.streaming_max_chars >= settings.streaming_min_chars

    def test_lower_bounds_clamp_negative_values(self) -> None:
        """负值/零值配置必须被钳到下界，不能直穿进运行时段计算。"""
        settings = load_settings(
            {
                "enabled": True,
                "max_segments": -5,
                "temperature": -2,
                "timeout_seconds": -5,
                "delay_base": -1,
                "delay_per_char": -1,
                "delay_max": -1,
                "min_length": -1,
                "max_tokens": 0,
                "streaming_min_chars": 0,
            },
        )
        assert settings is not None
        assert settings.max_segments == 1
        assert settings.temperature == 0.0
        assert settings.timeout_seconds == 0.1
        assert settings.delay_base == 0.0
        assert settings.delay_per_char == 0.0
        assert settings.delay_max == 0.0
        assert settings.min_length == 0
        assert settings.max_tokens == 1
        assert settings.streaming_min_chars == 1

    def test_upper_bounds_clamp(self) -> None:
        """每个数值字段都要被钳到 bounds 上界：bounds.py 自称配置上限单一来源，
        任何字段的 upper 被删/写错都必须有测试变红。"""
        for key, upper in (
            ("min_length", MAX_MIN_LENGTH),
            ("max_segments", MAX_SEGMENTS),
            ("max_tokens", MAX_TOKENS),
            ("timeout_seconds", MAX_TIMEOUT_SECONDS),
            ("temperature", MAX_TEMPERATURE),
            ("delay_base", MAX_DELAY_BASE),
            ("delay_per_char", MAX_DELAY_PER_CHAR),
            ("delay_max", MAX_DELAY_MAX),
            ("streaming_min_chars", MAX_STREAMING_MIN_CHARS),
            ("streaming_max_chars", MAX_STREAMING_MAX_CHARS),
        ):
            settings = load_settings({"enabled": True, key: 10**9})
            assert settings is not None, key
            assert getattr(settings, key) == upper, key

    def test_bool_false_words(self) -> None:
        for word in ("0", "false", "no", "off"):
            assert load_settings({"enabled": word}) is None, word

    def test_bool_true_words(self) -> None:
        for word in ("1", "true", "yes", "on"):
            settings = load_settings(
                {"enabled": True, "streaming_compat_enabled": word}
            )
            assert settings is not None
            assert settings.streaming_compat_enabled is True, word

    def test_bool_accepts_integers_and_floats(self) -> None:
        """手改 JSON 写成 1/0（甚至 1.0/0.0）时必须按布尔处理，不能静默回落默认值。"""
        assert load_settings({"enabled": 1}) is not None
        assert load_settings({"enabled": 0}) is None
        assert load_settings({"enabled": 1.0}) is not None
        assert load_settings({"enabled": 0.0}) is None
        settings = load_settings({"enabled": True, "streaming_compat_enabled": 1})
        assert settings is not None
        assert settings.streaming_compat_enabled is True
        settings = load_settings({"enabled": True, "streaming_compat_enabled": 0.0})
        assert settings is not None
        assert settings.streaming_compat_enabled is False

    def test_config_object_exception_falls_back_to_defaults(self) -> None:
        """配置对象 .get() 抛异常必须降级：异常会直接穿到 hook 与补丁路径。"""

        class BrokenConfig:
            def get(self, key, default=None):
                raise RuntimeError("config backend down")

        settings = load_settings(BrokenConfig())
        assert settings is not None
        assert settings.max_segments == 5

    def test_unparsable_float_falls_back_to_default(self) -> None:
        """非法浮点配置回落默认值（不能抛异常让插件加载失败）。"""
        settings = load_settings({"enabled": True, "delay_max": "abc"})
        assert settings is not None
        assert settings.delay_max == 2.5

    def test_settings_track_config_changes(self) -> None:
        """每次调用都按当前配置值解析：热改配置必须立刻生效。"""
        cfg = {"enabled": True, "max_segments": 4}
        assert load_settings(cfg).max_segments == 4
        cfg["max_segments"] = 6
        assert load_settings(cfg).max_segments == 6
        cfg["enabled"] = False
        assert load_settings(cfg) is None
