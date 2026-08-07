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
