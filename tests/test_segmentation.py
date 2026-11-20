from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
from astrbot_plugin_smart_segmentation.bounds import (
    ALLOWED_STYLES,
    DELAY_JITTER_SECONDS,
    MAX_DELAY_BASE,
    MAX_DELAY_MAX,
    MAX_DELAY_PER_CHAR,
    MAX_MIN_LENGTH,
    MAX_SEGMENTS,
    MAX_STREAMING_MAX_CHARS,
    MAX_STREAMING_MIN_CHARS,
    MAX_TIMEOUT_SECONDS,
)
from astrbot_plugin_smart_segmentation.segmentation import (
    SegmentationParseError,
    build_segmentation_prompt,
    calculate_send_delay,
    cap_segments,
    clean_segment_tail,
    extract_json_array_text,
    is_action_only_text,
    merge_segments_balancing_brackets,
    parse_segments_from_model_output,
    same_content,
    split_oversized_segments,
    split_text_at_brackets,
    split_text_locally,
    strip_thinking_content,
)
from astrbot_plugin_smart_segmentation.settings import (
    SegmentationSettings,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("第一句，第二句，。", "第一句，第二句"),
        ("收尾.,，。;；:：、", "收尾"),
        ("  缩进保留，\r\n", "  缩进保留\r\n"),
        ("等等...", "等等"),
        ("等等…", "等等…"),
        ("真的？！?!~", "真的？！?!~"),
        ("（动作。）", "（动作。）"),
        ("数值为 3.14。", "数值为 3.14"),
        ("👩‍💻，", "👩‍💻"),
        ("`print('hello');`，", "`print('hello');`"),
        ("```python\nx = 1;\n```", "```python\nx = 1;\n```"),
        ("```python\nx = 1;\n", "```python\nx = 1;\n"),
        ("`unfinished;", "`unfinished;"),
        ("https://example.com/path;", "https://example.com/path;"),
        ("https://example.com/path;，", "https://example.com/path;"),
        ("[链接](https://example.com/)，", "[链接](https://example.com/)"),
        ("，。\n", "\n"),
        ("", ""),
    ],
)
def test_clean_segment_tail(text: str, expected: str) -> None:
    assert clean_segment_tail(text) == expected


class TestStripThinking:
    def test_closed_think_tag(self) -> None:
        text = "<think>secret</think>你好"
        assert strip_thinking_content(text) == "你好"

    def test_thinking_tag_case_insensitive(self) -> None:
        text = "<THINKING>x</THINKING>正文"
        assert strip_thinking_content(text) == "正文"

    def test_mismatched_tag_names_do_not_swallow_the_reply(self) -> None:
        """开闭标签名不一致时，未闭合正则会一路吃到文末，把可见答案一起删掉。"""
        assert strip_thinking_content("<think>secret</think>答案") == "答案"
        assert strip_thinking_content("<thinking>secret</think>答案") == "答案"

    def test_unclosed_think_with_blank_line_is_discarded(self) -> None:
        assert strip_thinking_content("<think>还在想\n\n仍在推理") == ""

    def test_unclosed_think_without_boundary_is_discarded(self) -> None:
        """没有可见分界时不能把未闭合 think 后的内容当成正文发送。"""
        assert strip_thinking_content("<think>推理\n答案") == ""
        assert strip_thinking_content("<think>想想答案") == ""

    def test_unclosed_think_after_visible_prefix_discards_only_the_tail(self) -> None:
        assert strip_thinking_content("前文<think>内部推理") == "前文"

    def test_think_lookalike_in_body_is_kept(self) -> None:
        """正文里出现「<think 然后写推理 >」这种写法时不能把中间内容当标签删掉。"""
        text = "格式是 <think 然后写推理 > 结束"
        assert strip_thinking_content(text) == text

    def test_boundary_tags_removed(self) -> None:
        assert strip_thinking_content("前</think>后") == "前后"

    def test_code_inside_thought_does_not_hide_the_closing_tag(self) -> None:
        assert (
            strip_thinking_content("<think>```text\nprivate</think>visible")
            == "visible"
        )

    def test_nested_thinking_example_stays_hidden(self) -> None:
        text = "<think>reasoning `<think>example</think>` SECRET</think>answer"
        assert strip_thinking_content(text) == "answer"

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_longer_closing_fence_does_not_hide_later_thinking(
        self, fence: str
    ) -> None:
        text = f"{fence}text\nvisible\n{fence}{fence[0]}\n<think>SECRET</think>answer"
        assert (
            strip_thinking_content(text)
            == f"{fence}text\nvisible\n{fence}{fence[0]}\nanswer"
        )

    def test_marker_inside_code_line_does_not_close_fence(self) -> None:
        text = "```text\nthis ``` is literal\n<think>VISIBLE</think>\n```\nanswer"
        assert strip_thinking_content(text) == text

    @pytest.mark.parametrize("ticks", ["```", "````"])
    def test_inline_code_at_line_start_does_not_hide_later_thinking(
        self, ticks: str
    ) -> None:
        assert (
            strip_thinking_content(f"{ticks}foo{ticks} <think>SECRET</think>answer")
            == f"{ticks}foo{ticks} answer"
        )

    def test_closed_thinking_keeps_answer_before_later_unclosed_block(self) -> None:
        assert (
            strip_thinking_content(
                "<think>SECRET1</think>answer<think>SECRET2\n\nSECRET3"
            )
            == "answer"
        )

    @pytest.mark.parametrize("ticks", ["``", "```"])
    def test_unmatched_inline_recovery_keeps_answer_after_later_marker(
        self, ticks: str
    ) -> None:
        text = f"`<think>SECRET</think> xx {ticks} <think>PRIVATE</think>end"
        assert strip_thinking_content(text) == f"` xx {ticks} end"

    def test_multiline_tag_candidate_preserves_character_order(self) -> None:
        text = "`<think\n>SECRET</think>answer"
        assert strip_thinking_content(text) == "`<think\n>SECRETanswer"

    def test_partial_tag_after_pending_inline_is_not_duplicated(self) -> None:
        text = "`<think>SECRET</think>answer ``<think>LITERAL</think>`` <thi\nnext"
        assert (
            strip_thinking_content(text)
            == "`answer ``<think>LITERAL</think>`` <thi\nnext"
        )


class TestJsonExtractAndParse:
    def test_fenced_json(self) -> None:
        raw = '废话\n```json\n["一", "二"]\n```\n尾 ["三"]'
        assert json.loads(extract_json_array_text(raw)) == ["一", "二"]

    def test_embedded_array(self) -> None:
        raw = '结果如下: ["甲", "乙"] 谢谢'
        assert json.loads(extract_json_array_text(raw)) == ["甲", "乙"]

    def test_fenced_array_with_trailing_array(self) -> None:
        raw = '```\n["一", "二"]\n```\n["三", "四"]'
        assert json.loads(extract_json_array_text(raw)) == ["一", "二"]

    def test_code_fence_with_outside_array(self) -> None:
        raw = '```\nprint("hi")\n```\n结果: ["甲", "乙"]'
        assert json.loads(extract_json_array_text(raw)) == ["甲", "乙"]

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(SegmentationParseError):
            parse_segments_from_model_output("not-json", max_segments=5)

    def test_parse_empty_array_raises(self) -> None:
        with pytest.raises(SegmentationParseError):
            parse_segments_from_model_output("[]", max_segments=5)

    def test_parse_scalar_json_raises(self) -> None:
        """模型返回标量 JSON（字符串/数字）必须抛解析错误，不能被逐字符拆分。"""
        for raw in ('"hello"', "123", "null", "true"):
            with pytest.raises(SegmentationParseError):
                parse_segments_from_model_output(raw, max_segments=5)

    def test_parse_blank_segments_raise(self) -> None:
        with pytest.raises(SegmentationParseError):
            parse_segments_from_model_output('["", " "]', max_segments=5)

    def test_parse_balances_brackets_across_segments(self) -> None:
        """跨段括号配平：模型把一对括号拆到两个数组元素里时要合并。"""
        assert parse_segments_from_model_output('["（前", "后）"]', max_segments=5) == [
            "（前后）"
        ]

    def test_parse_keeps_newline_inside_bracket_part(self) -> None:
        raw = '["（哈\\n哈）", "继续"]'
        assert parse_segments_from_model_output(raw, max_segments=1) == [
            "（哈\n哈）\n继续"
        ]

    def test_parse_code_fence_kept_whole(self) -> None:
        """模型把整块代码放进一个数组元素时，行/括号 stage 不得再把它拆碎。"""
        code = "```python\ndef main():\n    print(1)\n```"
        raw = json.dumps(["给你个脚本", code, "跑完贴我"], ensure_ascii=False)
        assert parse_segments_from_model_output(raw, max_segments=20) == [
            "给你个脚本",
            code,
            "跑完贴我",
        ]

    def test_parse_code_fence_spanning_model_segments(self) -> None:
        """模型按行返回围栏时，代码块仍须保持完整并保留缩进。"""
        raw = json.dumps(
            [
                "说明。",
                "```python",
                "for i in range(3):",
                "    print(i)",
                "```",
                "后续内容。",
            ],
            ensure_ascii=False,
        )
        assert parse_segments_from_model_output(raw, max_segments=20) == [
            "说明",
            "```python\nfor i in range(3):\n    print(i)\n```",
            "后续内容",
        ]

    def test_parse_table_spanning_model_segments(self) -> None:
        raw = json.dumps(
            ["说明。", "| 城市 | 数量 |", "| --- | --- |", "| 上海 | 98。 |", "后续。"],
            ensure_ascii=False,
        )
        result = parse_segments_from_model_output(raw, max_segments=20)
        assert any(
            "| 城市 | 数量 |\n| --- | --- |\n| 上海 | 98。 |" in item for item in result
        )

    def test_parse_nested_short_fence_inside_long_fence(self) -> None:
        raw = json.dumps(
            [
                "intro",
                "````md",
                "```python",
                "x = (",
                "    1,",
                ")",
                "```",
                "````",
                "outro",
            ],
            ensure_ascii=False,
        )
        result = parse_segments_from_model_output(raw, max_segments=20)
        assert "````md\n```python\nx = (\n    1,\n)\n```\n````" in result[1]

    @pytest.mark.parametrize("ticks", ["`", "``"])
    def test_parse_inline_code_spanning_model_segments(self, ticks: str) -> None:
        raw = json.dumps(
            [f"run {ticks}foo", "()", f"{ticks} then continue"], ensure_ascii=False
        )
        assert parse_segments_from_model_output(raw, max_segments=20) == [
            f"run {ticks}foo(){ticks} then continue",
        ]

    def test_parse_markdown_link_spanning_model_segments(self) -> None:
        raw = json.dumps(
            ["look [doc]", "(https://example.com)", " after"],
            ensure_ascii=False,
        )
        result = parse_segments_from_model_output(raw, max_segments=20)
        assert any("[doc](https://example.com)" in item for item in result)

    def test_extract_no_bracket_returns_raw(self) -> None:
        raw = "模型说分段如下"
        assert extract_json_array_text(raw) == raw

    def test_extract_array_after_earlier_bracket(self) -> None:
        """首个 "[" 之前还有别的 "[" 时，first..last 会拼成非法 JSON：
        必须回退到最后一个 "[" 才能取到真正的数组。"""
        for raw in (
            '{"note": "[x]", "s": ["甲", "乙"]}',
            'Step [1]: ["甲", "乙"]',
        ):
            assert json.loads(extract_json_array_text(raw)) == ["甲", "乙"]

    def test_parse_deeply_nested_raises_parse_error(self) -> None:
        """深层嵌套 JSON 的 RecursionError 必须收敛为 SegmentationParseError。"""
        with pytest.raises(SegmentationParseError) as excinfo:
            parse_segments_from_model_output("[" * 50000, max_segments=5)
        assert isinstance(excinfo.value.__cause__, RecursionError)


class TestTrailingPeriodAndLocalSplit:
    @pytest.mark.parametrize("mark", ["~", "…", "!", "?"])
    def test_local_splits_on_every_strong_mark(self, mark: str) -> None:
        """这几个强句读符只有这里钉（金样只用全角，且 `~` 从不出现在句中）；
        全角 `。！？` 由金样 two_sentences / exclaim_question 覆盖。

        句末的单个「。」会被有意剥掉（更像真人聊天），其余情绪标点保留。
        """
        text = f"真的吗{mark}好吧。"
        expected_head = f"真的吗{mark}"
        assert split_text_locally(text, max_segments=5) == [expected_head, "好吧"]

    def test_local_keeps_leading_marks(self) -> None:
        """段首句读符曾因正则匹配规则被静默丢弃（内容丢失），现拼回首句。

        其余三种形状由金样 leading_marks_attached / leading_marks_after_bracket 覆盖。
        """
        assert split_text_locally("。你好。世界。", max_segments=5) == [
            "。你好",
            "世界",
        ]

    def test_local_keeps_url_whole(self) -> None:
        """半角 ? 是强句读符：不保护 URL 会把链接拆到两条消息里。"""
        assert split_text_locally(
            "https://example.com/a?b=1 看看这个。",
            max_segments=5,
        ) == ["https://example.com/a?b=1 看看这个"]
        # 中文不加空格：URL 匹配不能吃掉后面的句读和正文，否则整行再也切不动
        assert split_text_locally(
            "详情见https://a.com/x。希望对你有帮助！",
            max_segments=5,
        ) == ["详情见https://a.com/x", "希望对你有帮助！"]
        # 反过来，URL 路径里的中文不能被切断（切在中间等于把链接发坏）
        assert split_text_locally(
            "https://zh.wikipedia.org/wiki/中文 看看。",
            max_segments=5,
        ) == ["https://zh.wikipedia.org/wiki/中文 看看"]
        # 括号型 URL（维基常见）：括号 stage 不能把它拆开
        assert split_text_locally(
            "看 https://en.wikipedia.org/wiki/Foo_(bar) 谢谢。",
            max_segments=5,
        ) == ["看 https://en.wikipedia.org/wiki/Foo_(bar) 谢谢"]

    def test_same_content_only_tolerates_whitespace_and_periods(self) -> None:
        """模型 segmentation 允许去掉每条消息末尾的「。」、允许重排空白。

        这两条容差是 LLM 路径的命门：去掉任何一条，带句号或换行的回复都会静默
        永久退回本地兜底。
        """
        assert same_content(["你好", "世界"], "你好。世界。")
        assert same_content(["第一行", "第二行"], "第一行\n第二行")
        assert not same_content(["你好", "世界"], "你好。世界。还有一句。")
        assert not same_content(["你好", "世界"], "你好。世界！")
        assert not same_content(["第一第二"], "第一。第二。")
        assert same_content(["[50]", "。)"], "[50]。)")

    def test_same_content_preserves_code_fence_whitespace(self) -> None:
        source = "```py\nif x:\n    return 1\n```"
        changed = "```py\nif x:\nreturn 1\n```"
        assert same_content([source], source)
        assert not same_content([changed], source)

    @pytest.mark.parametrize("ticks", ["`", "``"])
    def test_same_content_preserves_inline_code_whitespace(self, ticks: str) -> None:
        source = f'运行 {ticks}print("a b"){ticks} 然后继续'
        changed = [f'运行 {ticks}print("ab"){ticks}', "然后继续"]
        assert same_content([source], source)
        assert not same_content(changed, source)

    def test_pua_placeholder_lookalike_does_not_raise(self) -> None:
        """正文里恰好出现同形的私用区字符时不能炸：越界的占位符原样保留。"""
        text = "看这个 https://example.com/a?b=1 然后9 结束。"
        segments = split_text_locally(text, max_segments=5)
        assert "9" in "".join(segments)
        assert "https://example.com/a?b=1" in "".join(segments)

    def test_local_no_empty_segment(self) -> None:
        """段本身就是「。」时去掉句末句号会得到空串，补发会为它单发一条空消息。"""
        assert split_text_locally("他笑了（笑）。", max_segments=5) == [
            "他笑了",
            "（笑）",
        ]
        assert split_text_locally("（笑）。", max_segments=5) == ["（笑）"]

    def test_local_action_only_with_marks_kept_whole(self) -> None:
        assert split_text_locally("（笑死我了！）", max_segments=5) == [
            "（笑死我了！）"
        ]

    def test_local_action_only_with_newline_kept_whole(self) -> None:
        assert split_text_locally("（哈\n哈）", max_segments=5) == ["（哈\n哈）"]

    def test_code_fence_kept_whole(self) -> None:
        """围栏被行/括号 stage 拆开会丢缩进，把 def main(): 拆成「def main」「()」「:」。"""
        text = (
            "给你个脚本，记得改路径\n"
            "```python\ndef main():\n    for i in range(3):\n        print(i)\n\nmain()\n```\n"
            "跑完把输出贴我"
        )
        segments = split_text_locally(text, max_segments=20)
        assert len(segments) <= 3, segments
        assert "()" not in segments and ":" not in segments, segments
        assert same_content(segments, text), segments
        code = [segment for segment in segments if "```python" in segment]
        assert len(code) == 1, segments
        assert "    for i in range(3):" in code[0] and "print(i)" in code[0], code

    def test_tilde_code_fence_kept_whole(self) -> None:
        text = "~~~python\nprint(hello!)\n~~~"
        assert split_text_locally(text, max_segments=10) == [text]

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_code_marker_inside_string_does_not_split_block(self, fence: str) -> None:
        text = f'{fence}python\nx = "{fence}"\n    print(1)\n{fence}'
        assert split_text_locally(text, max_segments=10) == [text]
        assert parse_segments_from_model_output(
            json.dumps([text]), max_segments=10
        ) == [text]

    def test_markdown_link_kept_whole(self) -> None:
        """只遮 URL 时 [] () 仍是结构字符，链接会被括号 stage 拆成三截。"""
        text = "看这个[文档](https://example.com/a?b=1)吧，里面有说明"
        assert split_text_locally(text, max_segments=5) == [text]

    def test_capped_tail_stays_under_platform_limit(self) -> None:
        """尾部合并必须受单段长度约束。全部并进一段会产出一条超过平台上限的消息，
        宿主侧（如 Discord 2000 字）直接静默截断——那正是分段要避免的。"""
        text = "这是一句完整的话。" * 900  # 8100 字，可切
        segments = split_text_locally(text, max_segments=5)
        assert max(len(seg) for seg in segments) <= 1500, [len(s) for s in segments]
        assert len(segments) > 5, "内容装不进 5 段时必须多分几条而不是超长一段"

    def test_unpunctuated_long_segment_stays_under_platform_limit(self) -> None:
        text = "A" * 5000
        segments = split_text_locally(text, max_segments=5)
        assert max(map(len, segments)) <= 1500, [len(segment) for segment in segments]
        assert "".join(segments) == text

    def test_long_segment_split_keeps_placeholder_lookalikes(self) -> None:
        text = "a" * 1499 + "\ue0000" + "b" * 1500 + "\ue001"
        assert "".join(split_oversized_segments([text])) == text

    def test_single_row_table_at_buffer_end_kept(self) -> None:
        """只有一行的表、又恰好处在缓冲区末尾（还没有行尾换行）时，整条正则必须
        还能命中：主体那一支要求至少一条以换行收尾的完整行，单行表只有尾部一支。"""
        assert split_text_locally("| a。 | b |", max_segments=5) == ["| a。 | b |"]

    @pytest.mark.parametrize(
        "table",
        [
            "销售情况如下：\n| 城市 | 数量 |\n| --- | --- |\n| 上海 | 98。 |\n| 北京 | 102 |",
            "| 项目 | 状态 |\n| --- | --- |\n| 备份 | 完成。 |\n| 升级 | 进行中 |",
            "对比一下：\n| a | b |\n|---|---|\n| 1 | 2 |",
        ],
        ids=["with-period", "two-rows", "ascii"],
    )
    def test_table_kept_whole(self, table: str) -> None:
        """表格行是结构化单位:逐行拆开会把一张表拆成 N 条消息,句号还会把某行
        切成两半、留下一条只有 `|` 的碎片。"""
        segments = split_text_locally(table, max_segments=10)
        assert not any(
            seg.strip().strip("|").strip() == ""
            or (seg.strip().startswith("|") and "\n" not in seg)
            for seg in segments
        ), segments
        joined = "\n".join(segments)
        for row in (
            line for line in table.split("\n") if line.lstrip().startswith("|")
        ):
            assert row.replace("。", "") in joined.replace("。", ""), segments

    def test_table_tail_row_without_closing_pipe_kept(self) -> None:
        """流式缓冲区末尾那行常还没收到行尾竖线。它必须并进同一张表：否则行内的
        句号就是切点，切开后两条消息都不是完整表格行。"""
        table = "| 城市 | 数量 |\n| --- | --- |\n| 上海 | 98。"
        assert split_text_locally(table, max_segments=10) == [table]

    def test_table_with_crlf_endings_kept(self) -> None:
        """MULTILINE 的 $ 停在 \n 之前，CRLF 文本的行尾是 \r。字符类不认它的话
        整张表漏遮，被逐行拆开。"""
        table = "前一句。\r\n| a | b |\r\n| c | d |\r\n结尾。"
        segments = split_text_locally(table, max_segments=10)
        joined = "\n".join(segments)
        assert "| a | b |\n| c | d |" in joined.replace("\r\n", "\n"), segments

    @pytest.mark.parametrize(
        "text",
        [
            "运行 `pip install a!=b` 这样不行吗？",
            "用 `foo()` 和 `bar?` 两个函数就够了。",
            "改 `rm -rf /tmp/x!` 之前先备份。",
            "这条 `a[0](b)` 会踩到括号 stage",
        ],
        ids=["bang", "paren-question", "bang-after-path", "brackets"],
    )
    def test_inline_code_kept_whole(self, text: str) -> None:
        """行内代码里的 ! ? () 是代码字符:不遮罩就会被标点与括号 stage 拦腰切开,
        两条消息的反引号都不配对,用户看到的是坏掉的 markdown。
        句末句号会被剥掉,这是产品规则,比对时排除。"""
        out = split_text_locally(text, max_segments=5)
        assert len(out) == 1, out
        assert out[0].replace("。", "") == text.replace("。", "")

    def test_double_backtick_inline_code_kept_whole(self) -> None:
        text = "Run ``a!=b()`` now?"
        assert split_text_locally(text, max_segments=5) == [text]

    @pytest.mark.parametrize(
        "text",
        [
            "```html\n<think>visible</think>\n```",
            "`<think>visible</think>`",
        ],
        ids=["fenced-code", "inline-code"],
    )
    def test_thinking_tags_inside_markdown_code_are_visible(self, text: str) -> None:
        assert split_text_locally(text, max_segments=5) == [text]

    @pytest.mark.parametrize("prefix", ["before `\n", "before ` unmatched "])
    def test_unmatched_inline_tick_does_not_hide_think(self, prefix: str) -> None:
        text = prefix + "<think>SECRET</think>answer"
        result = split_text_locally(text, max_segments=5)
        assert "answer" in "".join(result)
        assert "SECRET" not in "".join(result)

    def test_long_inline_code_kept_whole(self) -> None:
        code = "`" + "a" * 220 + "!" + "b" * 20 + "`"
        segments = split_text_locally(f"前文 {code} 后文。", max_segments=10)
        assert sum(code in segment for segment in segments) == 1, segments

    @pytest.mark.parametrize(
        "atom",
        [
            "```text\n" + "x" * 1600 + "\n```",
            "[文档](https://example.com/" + "x" * 1600 + ")",
        ],
        ids=["code-fence", "markdown-link"],
    )
    def test_overlong_markdown_atoms_still_respect_message_limit(
        self, atom: str
    ) -> None:
        segments = split_oversized_segments([atom])
        assert all(len(segment) <= 1500 for segment in segments)
        assert "".join(segments) == atom

    @pytest.mark.parametrize(
        "segment",
        [
            "https://example.com/"
            + "a" * 780
            + " "
            + "https://example.com/"
            + "b" * 780,
            "```text\n" + "a" * 740 + "\n```\n```text\n" + "b" * 740 + "\n```",
        ],
        ids=["multiple-urls", "multiple-code-fences"],
    )
    def test_multiple_short_markdown_atoms_still_respect_message_limit(
        self, segment: str
    ) -> None:
        segments = split_oversized_segments([segment])
        assert all(len(item) <= 1500 for item in segments)
        assert "".join(segments) == segment

    def test_unclosed_backtick_is_not_masked(self) -> None:
        """未闭合的反引号不能被当行内代码:否则整段正文会被吞进一个原子、
        该切的地方全不切。"""
        text = "他说 `这里没闭合。第二句在这里。"
        out = split_text_locally(text, max_segments=5)
        assert "".join(out).replace(" ", "").count("第二句") == 1
        assert len(out) > 1

    @pytest.mark.parametrize(
        "link",
        [
            "[" + "文" * 201 + "](https://example.com/a)",
            "[文档](https://example.com/" + "a" * 2049 + ")",
        ],
        ids=["link-text-over-limit", "link-url-over-limit"],
    )
    def test_oversized_markdown_link_kept_whole(self, link: str) -> None:
        """超出链接正则限长后要么完整保留，要么按平台上限切开且不丢正文。"""
        text = f"看这个{link}吧，里面有说明"
        segments = split_text_locally(text, max_segments=5)
        if len(text) <= 1500:
            assert segments == [text]
        else:
            assert all(len(segment) <= 1500 for segment in segments)
            assert "".join(segments) == text

    def test_url_atom_stops_at_ascii_close_paren(self) -> None:
        """URL 原子尾部不排除 ")" 时会吞掉闭括号与紧邻中文，括号 stage 就看不到那对括号。"""
        assert split_text_locally("看 https://a.com/x)(笑)", max_segments=5) == [
            "看 https://a.com/x)",
            "(笑)",
        ]

    def test_oversized_link_kept_whole_next_to_fence(self) -> None:
        """降级链接扫描与围栏遮罩串行：链接紧贴围栏时两边都整块保留，占位符不嵌套。"""
        link = "[" + "文" * 201 + "](https://example.com/a)"
        fence = "```py\nprint(1)\n```"
        text = f"看{link}然后\n{fence}\n结束"
        segments = split_text_locally(text, max_segments=5)
        joined = "".join(segments)
        assert "\ue000" not in joined and "\ue001" not in joined, segments
        assert any(link in segment for segment in segments), segments
        assert any(fence in segment for segment in segments), segments
        assert same_content(segments, text), segments

    def test_link_inside_fence_not_rescanned(self) -> None:
        """围栏先遮罩：围栏里的 ]( 不会被降级链接扫描当成新链接再切一遍。"""
        text = "```md\n[a](https://e.test/x)\n```"
        assert split_text_locally(text, max_segments=5) == [text]

    @pytest.mark.parametrize(
        "text",
        ["]" * 10000, "[" * 20000 + "](" * 10000],
        ids=["close-10k", "open-then-link-pairs"],
    )
    def test_many_brackets_keep_content(self, text: str) -> None:
        """满屏方括号是链接正则的平方回溯诱因：降级扫描必须保住内容，不能靠限长兜。"""
        segments = split_text_locally(text, max_segments=5)
        assert "".join(segments) == text

    def test_malformed_nested_link_keeps_prefix(self) -> None:
        text = "[[[a]( )END"
        segments = split_text_locally(text, max_segments=20)
        assert "".join(segments) == text

    def test_nested_link_text_kept_whole(self) -> None:
        """链接文本里再套链接时外层要整块保留。

        _MD_LINK_RE 在第一个 "](...)" 处收口，外层链接只剩裸 "]" 与 "(地址)"，
        括号 stage 会把一条链接拆成「正文 + 裸括号 + 地址」——内容还在，语义已坏。
        """
        text = "x [text [nested](https://x.test/a) tail](https://y.test/b) z"
        assert split_text_locally(text, max_segments=5) == [text]

    def test_adjacent_links_kept_whole(self) -> None:
        text = "看[a](https://a.test/1)和[b](https://b.test/2)结束"
        assert split_text_locally(text, max_segments=5) == [text]

    def test_link_text_with_balanced_brackets_kept_whole(self) -> None:
        """链接文本含配对方括号（markdown 合法写法）时不能拆成裸括号加地址。"""
        text = "看[a [b] c](https://x.test/y)结束"
        assert split_text_locally(text, max_segments=5) == [text]

    def test_link_url_with_parens_kept_whole(self) -> None:
        """URL 带配平括号（维基常见）时既有行为不能退化。"""
        text = "看[文档](https://en.wikipedia.org/wiki/Foo_(bar))结束"
        assert split_text_locally(text, max_segments=5) == [text]

    @pytest.mark.parametrize(
        "text",
        [
            "看[文档](https://x.test/Foo_(bar)?a=1)结束",
            "看 https://en.wikipedia.org/wiki/Foo_(bar)?a=1 结束",
        ],
        ids=["markdown", "bare"],
    )
    def test_paren_url_with_query_keeps_content(self, text: str) -> None:
        """括号型 URL 带 query 时按句读符切分，但内容必须逐字节无损。"""
        segments = split_text_locally(text, max_segments=5)
        assert "".join(segments) == text, segments

    def test_random_mixed_corpus_keeps_content(self) -> None:
        """拼接后逐字节等于原文：链接遮罩/还原对任意括号组合都不能吞字或重排。

        段数上限给足，避开「超限合并时补回行分隔符」这条会改变字节的既有取舍。
        """
        import random

        alphabet = [
            "[",
            "]",
            "(",
            ")",
            "text",
            "中文",
            "https://x.test/a",
            "](https://y.test/b)",
            "！",
            "a",
        ]
        rng = random.Random(20261114)
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
            segments = split_text_locally(text, max_segments=MAX_SEGMENTS)
            assert len(segments) <= MAX_SEGMENTS, (text, segments)
            assert "".join(segments) == text, (text, segments)
            assert "\ue000" not in "".join(segments), (text, segments)

    def test_fence_inside_fence_kept_whole(self) -> None:
        """```` 块里的内层 ``` 不能被当成闭合：否则整块仍被拆碎。

        文档里用四个反引号包住含围栏的示例是常见写法，模型也会这么给。
        """
        text = "````md\n```py\nx=1\n```\n````"
        segments = split_text_locally(text, max_segments=5)
        assert segments == [text], segments

    def test_atom_placeholders_do_not_nest(self) -> None:
        """后一趟遮罩不得把前一趟的占位符吞进原子。

        三趟遮罩（围栏→链接→URL）串行；URL 正则若不排除私用区，URL 紧贴围栏时
        会把占位符收进原子，还原后内层占位符当普通字符发出、那段原文丢失。
        """
        for text in (
            "https://a.test/x?y=1```py\nprint(1)\n```\n后面",
            "？[文档](https://e.test/a?b=1)。。https://example.com/a?b=1&c=2````md\n```\ninner\n```\n````。",
            "见 https://f.test/?q=1[说明](https://g.test/a?b=1)结尾",
        ):
            segments = split_text_locally(text, max_segments=1)
            joined = "".join(segments)
            assert "\ue000" not in joined and "\ue001" not in joined, (text, joined)
            assert same_content(segments, text), (text, segments)

    @pytest.mark.parametrize(
        "text",
        ["`x```x```!`", "| ```x``` |"],
        ids=["inline-code", "table"],
    )
    def test_fence_placeholder_is_not_swallowed_by_other_masks(self, text: str) -> None:
        segments = split_text_locally(text, max_segments=10)
        assert "".join(segments) == text
        assert "\ue000" not in "".join(segments)

    def test_local_cap_keeps_line_separator(self) -> None:
        """兜底合并尾部段必须保留行分隔符。

        最后一级 stage 是 ``_stage_sentences``，行边界在更早的 ``_stage_lines``
        就已经被吃掉；合并用空串会把列表项黏成「2. 再改配置文件3. 重启」。
        跨句注入的换行是这条取舍的已知代价（README 的本地规则说明）。
        """
        text = "可以，步骤如下：\n1. 先安装依赖\n2. 再改配置文件\n3. 重启 AstrBot"
        segments = split_text_locally(text, max_segments=2)
        assert segments[-1].count("\n") >= 1, segments


class TestActionAndBrackets:
    def test_not_action_when_trailing_text(self) -> None:
        assert is_action_only_text("（笑）哈") is False

    def test_not_action_when_close_bracket_precedes_text(self) -> None:
        """闭括号在段中（后文还有文字或另一对括号）不是纯动作段。"""
        assert is_action_only_text("你好（笑）") is False
        assert is_action_only_text("（a）b（c）") is False
        assert is_action_only_text("a）") is False

    def test_merge_unbalanced(self) -> None:
        assert merge_segments_balancing_brackets(["（前", "后）", "完"]) == [
            "（前后）",
            "完",
        ]

    def test_merge_brackets_nested(self) -> None:
        assert merge_segments_balancing_brackets(["（外（内）外）", "后"]) == [
            "（外（内）外）",
            "后",
        ]

    def test_merge_flushes_trailing_unbalanced_buffer(self) -> None:
        """末段左括号未闭合时，buffer 必须作为最后一段发出（不能丢内容）。"""
        assert merge_segments_balancing_brackets(["（a", "b"]) == ["（ab"]
        assert merge_segments_balancing_brackets(["完", "（未闭合"]) == [
            "完",
            "（未闭合",
        ]

    def test_merge_balances_all_bracket_pairs(self) -> None:
        """四对括号都要参与配平，不能只查第一对。"""
        for open_b, close_b in ("【", "】"), ("[", "]"), ("(", ")"):
            assert merge_segments_balancing_brackets(
                [f"{open_b}前", f"后{close_b}", "完"]
            ) == [f"{open_b}前后{close_b}", "完"]

    def test_split_text_at_brackets_all_pairs(self) -> None:
        """四对括号都要能在拆分阶段独立成段，不能只识别第一对。"""
        for open_b, close_b in (("（", "）"), ("【", "】"), ("[", "]"), ("(", ")")):
            assert split_text_at_brackets(f"正文{open_b}动作{close_b}继续") == [
                "正文",
                f"{open_b}动作{close_b}",
                "继续",
            ]


class TestCapSegments:
    def test_joiner_restores_dropped_separator(self) -> None:
        assert cap_segments(["1. 甲", "2. 乙"], max_segments=1, joiner="\n") == [
            "1. 甲\n2. 乙"
        ]

    def test_joiner_survives_when_capped_groups_split(self) -> None:
        segments = cap_segments(["a" * 800, "b" * 800], max_segments=1, joiner="\n")
        assert max(map(len, segments)) <= 1500
        assert "".join(segments) == "a" * 800 + "\n" + "b" * 800


class TestPromptAndDelay:
    def test_prompt_contains_text_and_style(self) -> None:
        prompt = build_segmentation_prompt("原文内容", "active", 4)
        assert "原文内容" in prompt
        assert "最多分成 4 条" in prompt
        assert "活泼" in prompt

    def test_prompt_requires_preserving_internal_punctuation(self) -> None:
        """模型漏掉原文逗号会被 same_content 拒绝，提示词必须把这条说清楚。"""
        prompt = build_segmentation_prompt("原文内容", "natural", 4)
        assert "逗号、分号" in prompt
        assert "必须保留" in prompt

    def test_delay_clamped_when_max_below_base(self) -> None:
        delay = calculate_send_delay("abc", 5.0, 0.0, 1.0)
        assert delay <= 1.0

    def test_delay_jitter_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """抖动必须真的加到延迟上（去掉 uniform 调用后本测试变红）。"""
        monkeypatch.setattr(
            "astrbot_plugin_smart_segmentation.segmentation.random.uniform",
            lambda _a, b: b,
        )
        assert calculate_send_delay("abcd", 1.0, 0.1, 9.0) == pytest.approx(
            1.4 + DELAY_JITTER_SECONDS
        )
        monkeypatch.setattr(
            "astrbot_plugin_smart_segmentation.segmentation.random.uniform",
            lambda a, _b: a,
        )
        assert calculate_send_delay("abcd", 1.0, 0.1, 9.0) == pytest.approx(1.4)

    def test_delay_per_char_weight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """每字权重：长度线性影响延迟。"""
        monkeypatch.setattr(
            "astrbot_plugin_smart_segmentation.segmentation.random.uniform",
            lambda *_args, **_kwargs: 0.0,
        )
        short = calculate_send_delay("a", 1.0, 0.1, 9.0)
        long = calculate_send_delay("a" * 11, 1.0, 0.1, 9.0)
        assert long - short == pytest.approx(1.0)


class TestBoundsAlignSchema:
    def test_schema_slider_max_matches_bounds(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        assert schema["min_length"]["slider"]["max"] == MAX_MIN_LENGTH
        assert schema["max_segments"]["slider"]["max"] == MAX_SEGMENTS
        assert schema["timeout_seconds"]["slider"]["max"] == MAX_TIMEOUT_SECONDS
        assert schema["delay_base"]["slider"]["max"] == MAX_DELAY_BASE
        assert schema["delay_max"]["slider"]["max"] == MAX_DELAY_MAX
        assert schema["delay_per_char"]["slider"]["max"] == MAX_DELAY_PER_CHAR
        assert schema["streaming_min_chars"]["slider"]["max"] == MAX_STREAMING_MIN_CHARS
        assert schema["streaming_max_chars"]["slider"]["max"] == MAX_STREAMING_MAX_CHARS

    def test_schema_slider_min_is_accepted(self) -> None:
        """UI 能选到的最小值必须原样生效：代码下界被抬到 slider min 之上时，
        用户按界面选值会被悄悄改写。"""
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        for key in (
            "min_length",
            "max_segments",
            "timeout_seconds",
            "delay_base",
            "delay_per_char",
            "delay_max",
            "streaming_min_chars",
            "streaming_max_chars",
        ):
            value = schema[key]["slider"]["min"]
            settings = load_settings({"enabled": True, key: value})
            assert settings is not None, key
            assert getattr(settings, key) == value, key

    def test_schema_style_options_match_allowed_styles(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        assert set(schema["style"]["options"]) == set(ALLOWED_STYLES)

    def test_schema_defaults_match_dataclass(self) -> None:
        """Schema default values must match SegmentationSettings defaults."""
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        defaults = SegmentationSettings()
        for field in fields(SegmentationSettings):
            expected = getattr(defaults, field.name)
            if field.name == "tail_clean_marks":
                expected = list(expected)
            assert schema[field.name]["default"] == expected, field.name


class TestPlaceholderEscape:
    def test_local_placeholder_lookalike_kept(self) -> None:
        """正文自带「U+E000 数字 U+E001」时，编号落在 atoms 下标内就被当成占位符。

        还原阶段分不清哪段是遮罩函数写入的，正文片段被原子内容顶替、原子再出现一次。
        """
        text = "看\ue0000\ue001然后 http://x.com/a"
        segments = split_text_locally(text, max_segments=5)
        joined = "".join(segments)
        assert joined == text, segments
        assert joined.count("http://x.com/a") == 1, segments

    def test_local_lookalike_next_to_url_link_fence(self) -> None:
        """同形字符与 URL/围栏/链接同时出现：转义与三趟遮罩不能互相吞并。"""
        text = (
            "看\ue0000\ue001说 https://x.test/a?b=1 "
            "见[文档](https://e.test/p?q=1)和```py x=1```然后\ue003\ue002 结束"
        )
        segments = split_text_locally(text, max_segments=5)
        assert "".join(segments) == text, segments

    def test_candidate_exhaustion_keeps_text(self) -> None:
        text = "看\ue0000\ue001 \x00\ue002\ue003\ue004 https://example.com/a 结束"

        local = split_text_locally(text, max_segments=5)
        assert "".join(local) == text

        raw = json.dumps([text], ensure_ascii=False)
        model = parse_segments_from_model_output(raw, max_segments=5)
        assert "".join(model) == text

    def test_model_output_placeholder_lookalike_kept(self) -> None:
        """模型路径共用同一套遮罩，同形字符在那里同样是内容损坏。"""
        text = "看\ue0000\ue001然后 http://x.com/a"
        raw = json.dumps([text], ensure_ascii=False)
        segments = parse_segments_from_model_output(raw, max_segments=5)
        joined = "".join(segments)
        assert joined == text, segments
        assert joined.count("http://x.com/a") == 1, segments


def test_table_does_not_swallow_the_paragraph_after_it() -> None:
    """表格是原子，但表格后面的换行不是。

    此前表格正则把行尾换行一起吞进原子，紧跟其后的正文被黏成同一段，
    一条消息里塞进表格与正文两段无关内容。
    """
    text = "| 项 | 值 |\n| a | 1 |\n紧跟表格后面的正文。另一个句子。"
    segments = split_text_locally(text, max_segments=10)
    assert segments[0].strip() == "| 项 | 值 |\n| a | 1 |", segments
    assert any("紧跟表格" in seg for seg in segments[1:]), segments


def test_crlf_table_leaves_no_stray_carriage_return() -> None:
    """CRLF 文本里 `\\r` 属于行尾，不能留在表格原子里；公开 API 输出不得含回车。"""
    segments = split_text_locally("| a | b |\r\n后续正文。", max_segments=10)
    assert "\r" not in "".join(segments)


def test_markdown_title_link_stays_whole() -> None:
    """`[文档](url "标题")` 是合法的 CommonMark 链接，整块保留。

    URL 后的空白让它命中不了普通链接正则，漏遮之后括号 stage 会把链接
    拆成「文档」和「(url "标题")」两条，用户收到的是坏掉的链接。
    """
    text = '点这个 [文档](https://example.com/a "标题") 看看。然后还有一句。'
    segments = split_text_locally(text, max_segments=10)
    # 链接必须整块落在某一段里：拼接后再查会掩盖被切开的事实。
    assert any('[文档](https://example.com/a "标题")' in seg for seg in segments), (
        segments
    )


def test_nested_link_stays_whole_and_long_title_link_is_bounded() -> None:
    """嵌套短链接保持完整；超长标题链接保住内容和平台长度上限。

    `_mask_bracket_links`（嵌套方括号）与 `_mask_long_links`（超长 URL）都靠
    「URL 到第一个空白为止」判终点，标题链接在 URL 后有空白，于是它们把链接
    判成不合法、漏遮，括号 stage 随后把整条链接拆成两条消息。
    """
    nested = '看[外[内]层](https://a.com/x "T")文字结束'
    segments = split_text_locally(nested, max_segments=10)
    assert any('[外[内]层](https://a.com/x "T")' in seg for seg in segments), segments

    long_url = "看[文档](https://a.com/" + "x" * 3000 + ' "T")之后结束'
    segments = split_text_locally(long_url, max_segments=10)
    assert all(len(segment) <= 1500 for segment in segments)
    assert "".join(segments) == long_url
