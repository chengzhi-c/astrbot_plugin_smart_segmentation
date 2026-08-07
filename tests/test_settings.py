from __future__ import annotations

from astrbot_plugin_smart_segmentation.settings import SettingsLoader, load_settings


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
