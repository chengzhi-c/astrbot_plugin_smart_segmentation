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
    MAX_TEMPERATURE,
    MAX_TIMEOUT_SECONDS,
    MAX_TOKENS,
)
from astrbot_plugin_smart_segmentation.bounds import DELAY_JITTER_SECONDS
from astrbot_plugin_smart_segmentation.segmentation import (
    SegmentationParseError,
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

    def test_boundary_tags_removed(self) -> None:
        assert strip_thinking_content("前</think>后") == "前后"

    def test_guards_against_backtracking(self) -> None:
        """未闭合开标签的病态输入必须毫秒级返回（回归守卫：`_sub_before` 的
        三个 limit 让匹配不可能回溯到文末）。"""
        import time

        pathological = "<think>" * 30000
        start = time.perf_counter()
        assert strip_thinking_content(pathological) == ""
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"strip_thinking_content 回溯爆炸：{elapsed:.1f}s"

    def test_guards_against_backtracking_after_close_tag(self) -> None:
        """闭合标签之后的大量未闭合开标签同样会回溯爆炸：只探测"全文有无
        闭合标签"覆盖不到这种形状。"""
        import time

        pathological = "正文</think>" + "<think>" * 20000
        start = time.perf_counter()
        assert strip_thinking_content(pathological) == "正文"
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"strip_thinking_content 回溯爆炸：{elapsed:.1f}s"

    def test_guards_against_backtracking_without_closing_angle(self) -> None:
        """开标签缺少 ">" 时 THINKING_BOUNDARY_RE 也会逐位回溯到文末。"""
        import time

        pathological = "<think" * 40000
        start = time.perf_counter()
        assert strip_thinking_content(pathological) == pathological
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"strip_thinking_content 回溯爆炸：{elapsed:.1f}s"

    def test_guards_against_backtracking_with_trailing_close_tag(self) -> None:
        """大量未闭合开标签 + 末尾一个闭合标签：闭合标签在文末时 `_sub_before`
        的 limit 形同虚设，开标签若允许吃 ">" 会逐位回溯（实测 1.3s vs 2ms）。"""
        import time

        pathological = "x" + "<thinking " * 16000 + "</thinking>"
        start = time.perf_counter()
        strip_thinking_content(pathological)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"strip_thinking_content 回溯爆炸：{elapsed:.1f}s"


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
        """段首句读符曾因正则匹配规则被静默丢弃（内容丢失），现拼回首句。"""
        assert split_text_locally("。你好。世界。", max_segments=5) == ["。你好", "世界"]
        assert split_text_locally("！！大事不好了。", max_segments=5) == ["！！大事不好了"]
        assert split_text_locally("（看）！快走。", max_segments=5) == ["（看）", "！快走"]

    def test_local_action_only_with_marks_kept_whole(self) -> None:
        assert split_text_locally("（笑死我了！）", max_segments=5) == ["（笑死我了！）"]

    def test_local_action_only_with_newline_kept_whole(self) -> None:
        assert split_text_locally("（哈\n哈）", max_segments=5) == ["（哈\n哈）"]

    def test_local_respects_max_segments(self) -> None:
        text = "一。二。三。四。五。"
        segs = split_text_locally(text, max_segments=2)
        assert len(segs) == 2
        assert "".join("".join(segs).split()) == "一二三四五"


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
    def test_join_tail(self) -> None:
        assert cap_segments(["a", "b", "c"], max_segments=2, joiner="") == ["a", "bc"]

    def test_max_one_merges_all(self) -> None:
        assert cap_segments(["a", "b", "c"], max_segments=1, joiner="") == ["abc"]

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
        assert schema["temperature"]["slider"]["max"] == MAX_TEMPERATURE
        assert schema["max_tokens"]["slider"]["max"] == MAX_TOKENS
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
            "max_tokens",
            "timeout_seconds",
            "temperature",
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
