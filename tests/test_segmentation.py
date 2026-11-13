from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from astrbot_plugin_smart_segmentation.bounds import (
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
from astrbot_plugin_smart_segmentation.bounds import DELAY_JITTER_SECONDS
from astrbot_plugin_smart_segmentation.segmentation import (
    SegmentationParseError,
    same_content,
    build_segmentation_prompt,
    calculate_send_delay,
    cap_segments,
    extract_json_array_text,
    is_action_only_text,
    merge_segments_balancing_brackets,
    parse_segments_from_model_output,
    split_text_at_brackets,
    split_text_locally,
    strip_thinking_content,
)
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings, load_settings


ROOT = Path(__file__).resolve().parents[1]


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

    def test_unclosed_think_prefix(self) -> None:
        text = "<think>还在想\n\n可见正文"
        assert "可见正文" in strip_thinking_content(text)
        assert "还在想" not in strip_thinking_content(text)

    def test_unclosed_think_without_blank_line_keeps_reply(self) -> None:
        """推理与正文之间没有空行时，正文不能被当成推理整段删掉。"""
        assert strip_thinking_content("<think>推理\n答案") == "推理\n答案"
        assert strip_thinking_content("<think>想想答案") == "想想答案"

    def test_think_lookalike_in_body_is_kept(self) -> None:
        """正文里出现「<think 然后写推理 >」这种写法时不能把中间内容当标签删掉。"""
        text = "格式是 <think 然后写推理 > 结束"
        assert strip_thinking_content(text) == text

    def test_boundary_tags_removed(self) -> None:
        assert strip_thinking_content("前</think>后") == "前后"

    @pytest.mark.parametrize(
        ("label", "pathological", "limit", "expected", "absent"),
        [
            ("标签名错配", "<think>a</thinking>" * 4000, 1.0, None, "<think"),
            ("未闭合开标签", "<think>" * 30000, 3.0, "", None),
            ("闭合标签之后再来大量开标签", "正文</think>" + "<think>" * 20000, 3.0, "正文", None),
            ("开标签缺少 >", "<think" * 40000, 3.0, "<think" * 40000, None),
            ("闭合标签落在文末", "x" + "<thinking " * 16000 + "</thinking>", 0.5, None, None),
        ],
        ids=["mismatch", "unclosed", "after-close", "no-gt", "close-at-end"],
    )
    def test_pathological_think_input_stays_fast(
        self,
        label: str,
        pathological: str,
        limit: float,
        expected: str | None,
        absent: str | None,
    ) -> None:
        """病态输入必须毫秒级返回:限界回溯是这套正则唯一的护栏。"""
        import time

        start = time.perf_counter()
        result = strip_thinking_content(pathological)
        elapsed = time.perf_counter() - start
        if expected is not None:
            assert result == expected
        if absent is not None:
            assert absent not in result
        assert elapsed < limit, f"{label} 退化:{elapsed:.1f}s"


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
        assert parse_segments_from_model_output(raw, max_segments=1) == ["（哈\n哈）\n继续"]

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
        with pytest.raises(SegmentationParseError):
            parse_segments_from_model_output("[" * 100000, max_segments=5)


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
        assert split_text_locally("。你好。世界。", max_segments=5) == ["。你好", "世界"]

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

    def test_long_word_run_does_not_blow_up(self) -> None:
        """无空格长串(长 base64、哈希、单行英文)不能让 URL 匹配退化成平方:
        scheme 段用 `*` 时每个起始位置都回溯到串尾(实测 8 万字符 9.8s)。"""
        import time

        start = time.perf_counter()
        segments = split_text_locally("a" * 80000, max_segments=5)
        elapsed = time.perf_counter() - start
        assert "".join(segments) == "a" * 80000
        assert elapsed < 2.0, f"split_text_locally 长串联时退化:{elapsed:.1f}s"

    def test_same_content_only_tolerates_whitespace_and_periods(self) -> None:
        """模型 segmentation 允许去掉每条消息末尾的「。」、允许重排空白。

        这两条容差是 LLM 路径的命门：去掉任何一条，带句号或换行的回复都会静默
        永久退回本地兜底。
        """
        assert same_content(["你好", "世界"], "你好。世界。")
        assert same_content(["第一行", "第二行"], "第一行\n第二行")
        assert not same_content(["你好", "世界"], "你好。世界。还有一句。")
        assert not same_content(["你好", "世界"], "你好。世界！")

    def test_pua_placeholder_lookalike_does_not_raise(self) -> None:
        """正文里恰好出现同形的私用区字符时不能炸：越界的占位符原样保留。"""
        text = "看这个 https://example.com/a?b=1 然后9 结束。"
        segments = split_text_locally(text, max_segments=5)
        assert "9" in "".join(segments)
        assert "https://example.com/a?b=1" in "".join(segments)

    def test_local_no_empty_segment(self) -> None:
        """段本身就是「。」时去掉句末句号会得到空串，补发会为它单发一条空消息。"""
        assert split_text_locally("他笑了（笑）。", max_segments=5) == ["他笑了", "（笑）"]
        assert split_text_locally("（笑）。", max_segments=5) == ["（笑）"]

    def test_local_action_only_with_marks_kept_whole(self) -> None:
        assert split_text_locally("（笑死我了！）", max_segments=5) == ["（笑死我了！）"]

    def test_local_action_only_with_newline_kept_whole(self) -> None:
        assert split_text_locally("（哈\n哈）", max_segments=5) == ["（哈\n哈）"]

class TestActionAndBrackets:
    def test_not_action_when_trailing_text(self) -> None:
        assert is_action_only_text("（笑）哈") is False

    def test_not_action_when_close_bracket_precedes_text(self) -> None:
        """闭括号在段中（后文还有文字或另一对括号）不是纯动作段。"""
        assert is_action_only_text("你好（笑）") is False
        assert is_action_only_text("（a）b（c）") is False
        assert is_action_only_text("a）") is False

    def test_merge_unbalanced(self) -> None:
        assert merge_segments_balancing_brackets(["（前", "后）", "完"]) == ["（前后）", "完"]

    def test_merge_brackets_nested(self) -> None:
        assert merge_segments_balancing_brackets(["（外（内）外）", "后"]) == [
            "（外（内）外）",
            "后",
        ]

    def test_merge_flushes_trailing_unbalanced_buffer(self) -> None:
        """末段左括号未闭合时，buffer 必须作为最后一段发出（不能丢内容）。"""
        assert merge_segments_balancing_brackets(["（a", "b"]) == ["（ab"]
        assert merge_segments_balancing_brackets(["完", "（未闭合"]) == ["完", "（未闭合"]

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
    def test_zero_max_still_returns_single_segment(self) -> None:
        """max_segments<=0 在正常配置下不可达；真传进来也不能原样返回多段。"""
        assert cap_segments(["a", "b", "c"], max_segments=0, joiner="") == ["abc"]

    def test_joiner_restores_dropped_separator(self) -> None:
        assert cap_segments(["1. 甲", "2. 乙"], max_segments=1, joiner="\n") == [
            "1. 甲\n2. 乙"
        ]


class TestPromptAndDelay:
    def test_prompt_contains_text_and_style(self) -> None:
        prompt = build_segmentation_prompt("原文内容", "active", 4)
        assert "原文内容" in prompt
        assert "最多分成 4 条" in prompt
        assert "活泼" in prompt

    def test_delay_clamped_when_max_below_base(self) -> None:
        delay = calculate_send_delay("abc", 5.0, 0.0, 1.0)
        assert delay <= 1.0

    def test_delay_exact_value_with_zero_rng(self) -> None:
        """delay = base + len*per_char（rng 注入 0 抖动时的精确值）。"""

        class ZeroRng:
            def uniform(self, a: float, b: float) -> float:
                return 0.0

        assert calculate_send_delay("abcd", 1.0, 0.1, 9.0, rng=ZeroRng()) == 1.4  # type: ignore[arg-type]

    def test_delay_jitter_is_applied(self) -> None:
        """抖动必须真的加到延迟上（去掉 uniform 调用后本测试变红）。"""

        class MaxJitterRng:
            def uniform(self, a: float, b: float) -> float:
                return b

        class MinJitterRng:
            def uniform(self, a: float, b: float) -> float:
                return a

        assert calculate_send_delay(
            "abcd", 1.0, 0.1, 9.0, rng=MaxJitterRng()  # type: ignore[arg-type]
        ) == pytest.approx(1.4 + DELAY_JITTER_SECONDS)
        assert calculate_send_delay(
            "abcd", 1.0, 0.1, 9.0, rng=MinJitterRng()  # type: ignore[arg-type]
        ) == pytest.approx(1.4)

    def test_delay_per_char_weight(self) -> None:
        """每字权重：长度线性影响延迟。"""

        class ZeroRng:
            def uniform(self, a: float, b: float) -> float:
                return 0.0

        short = calculate_send_delay("a", 1.0, 0.1, 9.0, rng=ZeroRng())  # type: ignore[arg-type]
        long = calculate_send_delay("a" * 11, 1.0, 0.1, 9.0, rng=ZeroRng())  # type: ignore[arg-type]
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
            assert schema[field.name]["default"] == getattr(defaults, field.name), field.name
