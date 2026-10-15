from __future__ import annotations

from astrbot_plugin_smart_segmentation.settings import SettingsLoader, load_settings
from astrbot_plugin_smart_segmentation.bounds import MAX_TIMEOUT_SECONDS


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

    def test_unparsable_float_falls_back_to_default(self) -> None:
        """非法浮点配置回落默认值（不能抛异常让插件加载失败）。"""
        settings = load_settings({"enabled": True, "delay_max": "abc"})
        assert settings is not None
        assert settings.delay_max == 2.5


class TestSettingsLoaderCache:
    def test_cache_hit_same_values(self) -> None:
        cfg = {"enabled": True, "max_segments": 4}
        loader = SettingsLoader(cfg)
        first = loader.get()
        second = loader.get()
        assert first is second

    def test_cache_invalidates_on_change(self) -> None:
        cfg = {"enabled": True, "max_segments": 4}
        loader = SettingsLoader(cfg)
        first = loader.get()
        cfg["max_segments"] = 6
        second = loader.get()
        assert first is not None and second is not None
        assert first is not second
        assert second.max_segments == 6

    def test_disable_toggle(self) -> None:
        cfg = {"enabled": True}
        loader = SettingsLoader(cfg)
        assert loader.get() is not None
        cfg["enabled"] = False
        assert loader.get() is None
