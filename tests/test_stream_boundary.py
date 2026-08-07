from __future__ import annotations

from astrbot_plugin_smart_segmentation.streaming.wrapper import (
    find_stream_boundary,
    pop_stream_segments,
)


class TestStreamBoundary:
    def test_split_on_strong_mark_after_min(self) -> None:
        text = "你好啊。还有吗"
        boundary = find_stream_boundary(text, min_chars=3, max_chars=90)
        assert boundary == 4  # 「你好啊。」

    def test_wait_if_shorter_than_min(self) -> None:
        text = "嗨。"
        assert find_stream_boundary(text, min_chars=5, max_chars=90) is None

    def test_newline_boundary(self) -> None:
        text = "第一行\n第二行"
        assert find_stream_boundary(text, min_chars=1, max_chars=90) == 3

    def test_max_chars_weak_boundary(self) -> None:
        text = "啊" * 30 + "，" + "吧" * 10
        boundary = find_stream_boundary(text, min_chars=5, max_chars=35)
        assert boundary is not None
        assert boundary <= 36

    def test_pop_multiple(self) -> None:
        segments, rest = pop_stream_segments(
            "一句。两句。未完",
            min_chars=2,
            max_chars=90,
        )
        assert segments == ["一句。", "两句。"]
        assert rest == "未完"

    def test_newline_at_zero_returns_end(self) -> None:
        assert find_stream_boundary("\nhello", min_chars=2, max_chars=90) == 1

    def test_weak_boundary_before_min_no_cut(self) -> None:
        assert find_stream_boundary("你，好", min_chars=5, max_chars=90) is None

    def test_hard_cut_at_max_without_boundary(self) -> None:
        assert find_stream_boundary("a" * 100, min_chars=10, max_chars=50) == 50

    def test_weak_boundary_picks_latest_not_first_declared(self) -> None:
        """弱边界应取最晚位置，而非常量声明顺序最靠前的（O-13）。"""
        # 逗号 index 1，分号 index 4；应在分号后切
        assert find_stream_boundary("啊，这样；好吧", min_chars=1, max_chars=7) == 5
