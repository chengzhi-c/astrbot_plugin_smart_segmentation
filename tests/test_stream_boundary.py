from __future__ import annotations

import pytest

from astrbot_plugin_smart_segmentation.streaming.wrapper import (
    find_stream_boundary,
    pop_stream_segments,
)


class TestStreamBoundary:
    def test_split_on_strong_mark_after_min(self) -> None:
        text = "你好啊。还有吗"
        boundary = find_stream_boundary(text, min_chars=3, max_chars=90)
        assert boundary == 4  # 「你好啊。」

    def test_strong_mark_respects_max_chars(self) -> None:
        """强句读符也受 max_chars 约束：只查下界会让超长句子整段穿过，
        且同一文本的分段结果会随 provider 的 chunk 大小漂移。"""
        text = "x" * 200 + "。"
        boundary = find_stream_boundary(text, min_chars=18, max_chars=90)
        assert boundary == 90  # 超过 max_chars 的句号不能作为边界
        segments, rest = pop_stream_segments(text, min_chars=18, max_chars=90)
        assert all(len(s) <= 90 for s in segments)
        assert "".join(segments) + rest == text

    def test_wait_if_shorter_than_min(self) -> None:
        text = "嗨。"
        assert find_stream_boundary(text, min_chars=5, max_chars=90) is None

    def test_newline_boundary_includes_the_newline(self) -> None:
        """边界含换行符本身：平台把各段 Plain 无分隔符拼接，丢边界换行会把
        markdown 列表/分步说明黏成一行。"""
        text = "第一行\n第二行"
        assert find_stream_boundary(text, min_chars=1, max_chars=90) == 4

    def test_max_chars_weak_boundary(self) -> None:
        text = "啊" * 30 + "，" + "吧" * 10
        boundary = find_stream_boundary(text, min_chars=5, max_chars=35)
        assert boundary == 31  # 窗口内最晚弱边界（逗号）之后

    def test_pop_multiple(self) -> None:
        segments, rest = pop_stream_segments(
            "一句。两句。未完",
            min_chars=2,
            max_chars=90,
        )
        assert segments == ["一句。", "两句。"]
        assert rest == "未完"

    def test_newline_at_zero_returns_end(self) -> None:
        assert find_stream_boundary("\nhello", min_chars=1, max_chars=90) == 1

    def test_newline_respects_min_chars(self) -> None:
        """换行边界曾完全绕过 min_chars，短行被切成 3~5 字碎消息。"""
        text = "第一行\n第二行\n第三行内容更长一些"
        assert find_stream_boundary(text, min_chars=18, max_chars=90) is None

    def test_newline_respects_max_chars(self) -> None:
        """超过 max_chars 的换行不能作为边界（会等太久）。

        输入让换行 run 起点落在窗口内、末尾超过 max_chars，确保真的走到
        换行分支的 max 闸门（而不是被弱边界回退顺带满足）。
        """
        text = "啊" * 20 + "\n" * 20 + "吧" * 10
        boundary = find_stream_boundary(text, min_chars=5, max_chars=25)
        assert boundary == 25  # 换行被拒，落到硬切

    def test_newline_run_scan_stops_at_max_chars(self) -> None:
        """run 末尾扫描的上限是 max_chars+1：超长 run 必须被拒（不能被截断成
        恰好 max_chars 而当成合法边界），回落到更早的强句读符。"""
        assert find_stream_boundary("。\n\n\n\n\nabc", min_chars=1, max_chars=5) == 1

    def test_newline_skips_to_next_qualifying_one(self) -> None:
        """第一个换行不满足 min/max 时必须继续看后面的换行。

        否则"短首行 + 按行列举"的常见回复形状会完全失去实时分段。
        """
        text = "步骤如下\n1. 打开设置页面，找到通用选项\n2. 点击分段开关并保存"
        boundary = find_stream_boundary(text, min_chars=18, max_chars=90)
        assert boundary is not None
        assert 18 <= boundary <= 90
        # 必须跳过第一个换行（index 4）、落在第二个换行上：段尾是换行符，
        # 且段内恰好保留第一个换行
        assert text[boundary - 1] == "\n"
        assert text[:boundary].count("\n") == 2

    def test_pop_keeps_newline_in_segment(self) -> None:
        """分段必须保留边界换行符（平台无分隔符拼接各段）。"""
        segments, rest = pop_stream_segments(
            "第一行\n第二行\n第三行",
            min_chars=1,
            max_chars=90,
        )
        assert segments == ["第一行\n", "第二行\n"]
        assert rest == "第三行"

    def test_pop_strips_spaces_but_not_newlines(self) -> None:
        segments, rest = pop_stream_segments(
            "  一句。  两句。  ",
            min_chars=2,
            max_chars=90,
        )
        assert segments == ["一句。", "两句。"]
        assert rest == ""

    def test_hard_cut_keeps_following_newline(self) -> None:
        """硬切位置上的换行不能丢：find_stream_boundary 只在窗口内找换行，
        恰好落在 max_chars 处的换行只能靠 pop 的 lstrip 保住。"""
        segments, rest = pop_stream_segments(
            "x" * 90 + "\nnext",
            min_chars=18,
            max_chars=90,
        )
        assert segments == ["x" * 90]
        assert rest == "\nnext"

    def test_pop_skips_whitespace_only_segments(self) -> None:
        assert pop_stream_segments("     ", min_chars=1, max_chars=3) == ([], "")
        assert pop_stream_segments("\n\n\nhello", min_chars=3, max_chars=90) == (
            [],
            "hello",
        )

    def test_newline_run_cannot_pad_min_chars(self) -> None:
        """min_chars 按内容字符计：长换行 run 会把原始长度垫过闸门，切出只有
        两三个字的消息。此时应继续等待后续内容。"""
        segments, rest = pop_stream_segments(
            "好的" + "\n" * 20 + "后面",
            min_chars=18,
            max_chars=90,
        )
        assert segments == []
        assert rest == "好的" + "\n" * 20 + "后面"  # 等待中不能丢内容

    def test_leading_whitespace_cannot_pad_min_chars(self) -> None:
        """段首空白同样不能把强句读符边界的长度垫过闸门。"""
        text = "\n" * 11 + "？开、;吧;存。"
        assert find_stream_boundary(text, min_chars=18, max_chars=90) is None

    def test_weak_boundary_before_min_no_cut(self) -> None:
        assert find_stream_boundary("你，好", min_chars=5, max_chars=90) is None

    def test_weak_boundary_at_min_is_accepted(self) -> None:
        """弱边界与强句读符/换行共用同一道 min 闸门：逗号恰好落在 min_chars-1
        处时必须被接受，否则会退化成硬切。"""
        text = "x" * 17 + "，" + "y" * 80
        assert find_stream_boundary(text, min_chars=18, max_chars=90) == 18

    def test_hard_cut_at_max_without_boundary(self) -> None:
        assert find_stream_boundary("a" * 100, min_chars=10, max_chars=50) == 50

    @pytest.mark.parametrize(
        "cluster",
        ["👨‍👩‍👧", "🇨🇳", "❤️", "👍🏽"],
    )
    def test_hard_cut_does_not_split_emoji_cluster(self, cluster: str) -> None:
        """硬切落在字形簇中间会把一个 emoji 裂成两半。四类接缝各自钉：
        ZWJ 序列、成对区域指示符（国旗）、变体选择符、肤色修饰符。"""
        text = "a" * 89 + cluster + "b" * 5
        boundary = find_stream_boundary(text, min_chars=18, max_chars=90)
        assert boundary is not None
        assert text[boundary:] == cluster + "b" * 5

    def test_boundary_search_is_bounded_by_window(self) -> None:
        """边界查找必须限定在 [min_chars, max_chars] 窗口内：全文扫描会让单个
        大 chunk 的耗时随长度线性膨胀（实测差三个数量级）。"""
        import time

        text = "啊" * 300000  # 无任何边界：最坏情况
        start = time.perf_counter()
        for _ in range(20):
            assert find_stream_boundary(text, min_chars=18, max_chars=90) == 90
        elapsed = time.perf_counter() - start
        # 窗口化实测 0.2ms，全文扫描实测 508ms：阈值取两者之间
        assert elapsed < 0.05, f"find_stream_boundary 退化为全文扫描：{elapsed:.3f}s"

    def test_weak_boundary_picks_latest_not_first_declared(self) -> None:
        """弱边界应取最晚位置，而非常量声明顺序最靠前的（O-13）。"""
        # 逗号 index 1，分号 index 4；应在分号后切
        assert find_stream_boundary("啊，这样；好吧", min_chars=1, max_chars=7) == 5
