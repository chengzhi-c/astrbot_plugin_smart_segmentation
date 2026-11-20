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

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/a?b=1&c=2",
            "http://example.com/a!b?q=1&next=2",
        ],
    )
    def test_url_query_is_indivisible(self, url: str) -> None:
        text = "详情见 " + url + " 继续阅读"
        assert find_stream_boundary(text, min_chars=6, max_chars=90) is None
        line = "详情见 " + url + "\n"
        assert find_stream_boundary(line + "正文", min_chars=6, max_chars=90) == len(
            line
        )

    @pytest.mark.parametrize("closing", ["", "\n```"])
    def test_fenced_code_is_indivisible(self, closing: str) -> None:
        text = "```python\nprint('hello?')\nprint('world!')" + closing
        assert find_stream_boundary(text, min_chars=6, max_chars=90) is None
        if closing:
            assert (
                find_stream_boundary(text + "\n正文", min_chars=6, max_chars=90)
                == len(text) + 1
            )

    def test_tilde_fence_hides_internal_boundary(self) -> None:
        text = "~~~python\nprint(hello!)\n~~~\nvisible。"
        boundary = find_stream_boundary(text, min_chars=6, max_chars=90)
        assert boundary == text.index("visible。")

    def test_double_backtick_span_hides_internal_boundary(self) -> None:
        text = "Run ``a!=b()`` now? Then visible。"
        boundary = find_stream_boundary(text, min_chars=6, max_chars=90)
        assert boundary == text.index(" Then")

    def test_max_boundary_stops_before_url(self) -> None:
        atom = "https://example.com/a?b=1&c=2&d=3"
        prefix = "prefix text "
        assert len(atom) < 40 < len(prefix + atom)
        assert find_stream_boundary(prefix + atom, min_chars=6, max_chars=40) == len(
            prefix
        )

    @pytest.mark.parametrize(
        "atom",
        [
            "https://x.test/a?b=1&c=2" + "x" * 100,
            "```py\nx = 1; y = 2!\n" + "x" * 100,
        ],
    )
    def test_oversized_atom_hard_cut_makes_progress(self, atom: str) -> None:
        assert find_stream_boundary(atom, min_chars=6, max_chars=20) == 20
        segments, rest = pop_stream_segments(atom, min_chars=6, max_chars=20)
        assert segments
        assert all(6 <= len(segment) <= 20 for segment in segments)
        assert len(rest) < 20
        assert "".join(segments) + rest == atom

    @pytest.mark.parametrize(
        "text",
        [
            "这个装法很好用 `pip install a!=b` 这样装，然后重启插件就可以了。",
            "执行 `rm -rf /tmp/x!` 之前一定要先备份，出事了很难恢复。",
            "先跑 `npm run build` 再跑 `npm test` 就完事了，别的不需要动。",
        ],
        ids=["bang-in-code", "bang-after-path", "two-codes"],
    )
    def test_inline_code_is_indivisible(self, text: str) -> None:
        """行内代码里的 ! ? 是代码字符:不遮住就会在中间切开,两条消息的反引号
        都不配对。与非流式路径的 _mask_atoms 保持同一口径。"""
        boundary = find_stream_boundary(text + "\n后续", min_chars=18, max_chars=90)
        assert boundary == len(text) + 1, f"未在完整代码后分段：{boundary}"

    def test_long_inline_code_is_indivisible(self) -> None:
        code = "`" + "a" * 220 + "!" + "b" * 20 + "`"
        text = "前文 " + code + " 后文。"
        boundary = find_stream_boundary(text, min_chars=6, max_chars=300)
        assert boundary == len(text), f"切进了行内代码：{boundary}"

    def test_table_tail_row_is_not_a_cut_point(self) -> None:
        """流式缓冲区末尾的表格残行不能成为切点：那行的句号会把表切开，
        两条消息都不是完整表格行。"""
        table = "数据如下：\n| 城市 | 数量 |\n| --- | --- |\n| 上海 | 98。 |\n| 北京 | 102 |"
        assert find_stream_boundary(table, min_chars=18, max_chars=90) is None
        boundary = find_stream_boundary(table + "\n后续。", min_chars=18, max_chars=90)
        assert boundary is not None and boundary > len(table)

    @pytest.mark.parametrize(
        "text",
        [
            "参考这个链接 [真的吗？](https://example.com/a?b=1) 就明白了，还有别的内容。",
            "我推荐 [这个方法！](https://example.com/x) 你试试看，应该能解决问题。",
            "看 [a [b] c](https://example.com/y) 这个嵌套的例子，说明更清楚一些。",
        ],
        ids=["quest-in-text", "bang-in-text", "nested"],
    )
    def test_markdown_link_is_indivisible(self, text: str) -> None:
        """链接文本里的问号、叹号是链接的一部分。流式窗口不遮 markdown 链接时，
        切点会落进 `[...]` 与 `(...)` 之间，两条消息的链接都不成立。"""
        found = False
        for end in range(18, len(text) + 1):
            chunk = text[:end]
            boundary = find_stream_boundary(chunk, min_chars=18, max_chars=90)
            if boundary is None:
                continue
            found = True
            cut = chunk[:boundary]
            assert not ("[" in cut and ")" not in cut), f"切进了链接：{cut!r}"
        assert found, "整个链接缓冲中没有可验证的边界"

    def test_nested_markdown_link_with_strong_punctuation_is_indivisible(self) -> None:
        text = "看[外[内?]层](https://a.com/x)后文。"
        link_start = text.index("[")
        link_end = text.index(")后文") + 1
        boundary = find_stream_boundary(text, min_chars=6, max_chars=90)
        assert boundary is not None and not link_start < boundary < link_end

    def test_unclosed_inline_code_is_not_a_cut_point(self) -> None:
        """闭反引号还没到的那一帧，代码里的 ! 仍是边界标记。不按未闭单元遮到
        缓冲区末尾，就会在这里切开，两条消息的反引号都不配对。"""
        text = "先跑 `pip install a!=b` 这样装，然后重启插件就可以生效了。"
        closing = text.index("`", text.index("`") + 1)
        for end in range(18, closing + 1):
            chunk = text[:end]
            assert find_stream_boundary(chunk, min_chars=18, max_chars=90) is None
        assert find_stream_boundary(text, min_chars=18, max_chars=90) == len(text)

    def test_fence_wins_over_url_ending_at_closing_marker(self) -> None:
        line = "```text\nhttps://x.test/path```\n```\n"
        assert find_stream_boundary(line + "正文", min_chars=6, max_chars=90) == len(
            line
        )

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

    def test_pop_keeps_boundary_spaces_and_indentation(self) -> None:
        segments, rest = pop_stream_segments(
            "  一句。  两句。  ",
            min_chars=2,
            max_chars=90,
        )
        assert segments == ["  一句。", "  两句。"]
        assert rest == "  "

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
        assert pop_stream_segments("     ", min_chars=1, max_chars=3) == ([], "  ")
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

    @pytest.mark.parametrize("prefix_length, expected", [(16, 90), (17, 18)])
    def test_weak_boundary_respects_min(self, prefix_length, expected) -> None:
        text = "x" * prefix_length + "，" + "y" * 80
        assert find_stream_boundary(text, min_chars=18, max_chars=90) == expected

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

    def test_weak_boundary_picks_latest_not_first_declared(self) -> None:
        """弱边界应取最晚位置，而非常量声明顺序最靠前的（O-13）。"""
        # 逗号 index 1，分号 index 4；应在分号后切
        assert find_stream_boundary("啊，这样；好吧", min_chars=1, max_chars=7) == 5

    def test_flag_run_stays_above_min_chars(self) -> None:
        """连续国旗 emoji 会让簇边界一路回退到 1；退到不足 min_chars 时必须硬切在
        max_chars，否则会连发一串单字消息。"""
        segments, _rest = pop_stream_segments("🇨🇳🇺🇸" * 30, min_chars=5, max_chars=25)
        assert segments
        assert all(len(segment) >= 5 for segment in segments)
