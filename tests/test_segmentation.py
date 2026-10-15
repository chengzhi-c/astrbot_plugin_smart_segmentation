from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path

import pytest

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
from astrbot_plugin_smart_segmentation.bounds import DELAY_JITTER_SECONDS
from astrbot_plugin_smart_segmentation.segmentation import (
    SegmentationParseError,
    _stage_lines,
    _stage_sentences,
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
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings


ROOT = Path(__file__).resolve().parents[1]


class TestStripThinking:
    def test_closed_think_tag(self) -> None:
        text = "<think>secret</think>你好"
        assert strip_thinking_content(text) == "你好"

    def test_thinking_tag_case_insensitive(self) -> None:
        text = "<THINKING>x</THINKING>正文"
        assert strip_thinking_content(text) == "正文"

    def test_unclosed_think_prefix(self) -> None:
        text = "<think>还在想\n\n可见正文"
        assert "可见正文" in strip_thinking_content(text)
        assert "还在想" not in strip_thinking_content(text)

    def test_boundary_tags_removed(self) -> None:
        assert strip_thinking_content("前</think>后") == "前后"

    def test_empty(self) -> None:
        assert strip_thinking_content("") == ""

    def test_guards_against_backtracking(self) -> None:
        """大量未闭合开标签会让 THINKING_TAG_RE 回溯爆炸（无守卫时耗时随输入
        长度指数级放大，阻塞事件循环）；守卫后应毫秒级返回。"""
        import time

        pathological = "<think>" * 30000
        start = time.perf_counter()
        assert strip_thinking_content(pathological) == ""
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"strip_thinking_content 回溯爆炸：{elapsed:.1f}s"


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

    def test_parse_strips_whitespace(self) -> None:
        raw = '[" 一 ", " 二 "]'
        assert parse_segments_from_model_output(raw, max_segments=5) == ["一", "二"]

    def test_parse_success_cap_joins_tail(self) -> None:
        raw = '["a", "b", "c", "d"]'
        assert parse_segments_from_model_output(raw, max_segments=2) == ["a", "bcd"]

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

    def test_parse_splits_action_brackets(self) -> None:
        raw = '["你好（挥手）继续"]'
        assert parse_segments_from_model_output(raw, max_segments=5) == [
            "你好",
            "（挥手）",
            "继续",
        ]

    def test_parse_keeps_newline_inside_bracket_part(self) -> None:
        raw = '["（哈\\n哈）", "继续"]'
        assert parse_segments_from_model_output(raw, max_segments=1) == ["（哈\n哈）继续"]

    def test_extract_no_bracket_returns_raw(self) -> None:
        raw = "模型说分段如下"
        assert extract_json_array_text(raw) == raw

    def test_extract_prefers_array_candidate(self) -> None:
        """模型把数组包在对象里时，应取到可用的数组而不是整个对象。"""
        raw = '```json\n{"segments": ["消息1", "消息2"]}\n```'
        assert json.loads(extract_json_array_text(raw)) == ["消息1", "消息2"]

    def test_parse_deeply_nested_raises_parse_error(self) -> None:
        """深层嵌套 JSON 的 RecursionError 必须收敛为 SegmentationParseError。"""
        with pytest.raises(SegmentationParseError):
            parse_segments_from_model_output("[" * 100000, max_segments=5)


class TestTrailingPeriodAndLocalSplit:
    def test_local_strips_single_period(self) -> None:
        segs = split_text_locally("你好。世界。", max_segments=5)
        assert segs == ["你好", "世界"]

    @pytest.mark.parametrize("mark", ["。", "！", "？", "~", "…", "!", "?"])
    def test_local_splits_on_every_strong_mark(self, mark: str) -> None:
        """每个强句读符都要能作为句中边界（不只测全角句号）。

        句末的单个「。」会被有意剥掉（更像真人聊天），其余情绪标点保留。
        """
        text = f"真的吗{mark}好吧。"
        expected_head = "真的吗" if mark == "。" else f"真的吗{mark}"
        assert split_text_locally(text, max_segments=5) == [expected_head, "好吧"]

    def test_local_keeps_leading_marks(self) -> None:
        """段首句读符曾因正则匹配规则被静默丢弃（内容丢失），现拼回首句。"""
        assert split_text_locally("。你好。世界。", max_segments=5) == ["。你好", "世界"]
        assert split_text_locally("！！大事不好了。", max_segments=5) == ["！！大事不好了"]
        assert split_text_locally("（看）！快走。", max_segments=5) == ["（看）", "！快走"]

    def test_local_keeps_ellipsis_style_periods(self) -> None:
        segs = split_text_locally("等等。。。好的。", max_segments=5)
        assert segs[0] == "等等。。。"
        assert segs[1] == "好的"

    def test_local_action_only_kept_whole(self) -> None:
        assert split_text_locally("（轻轻笑了）", max_segments=5) == ["（轻轻笑了）"]

    def test_local_action_only_with_marks_kept_whole(self) -> None:
        assert split_text_locally("（笑死我了！）", max_segments=5) == ["（笑死我了！）"]

    def test_local_action_only_with_newline_kept_whole(self) -> None:
        assert split_text_locally("（哈\n哈）", max_segments=5) == ["（哈\n哈）"]

    def test_local_respects_max_segments(self) -> None:
        text = "一。二。三。四。五。"
        segs = split_text_locally(text, max_segments=2)
        assert len(segs) == 2
        assert "".join(segs).replace("。", "") == "一二三四五"


class TestActionAndBrackets:
    def test_action_only_cn(self) -> None:
        assert is_action_only_text("（笑）") is True

    def test_action_only_en(self) -> None:
        assert is_action_only_text("(nod)") is True

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

    def test_merge_ignores_surplus_close_bracket(self) -> None:
        """右括号过剩（孤立右括号、`:)` 类表情）不得把后续所有段粘成一条。"""
        assert merge_segments_balancing_brackets(["a）", "b", "c"]) == ["a）", "b", "c"]
        assert merge_segments_balancing_brackets(["你好）", "今天天气不错"]) == [
            "你好）",
            "今天天气不错",
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

    @pytest.mark.parametrize(
        ("open_b", "close_b"),
        [("（", "）"), ("【", "】"), ("[", "]"), ("(", ")")],
    )
    def test_split_text_at_brackets_all_pairs(
        self, open_b: str, close_b: str
    ) -> None:
        """四对括号都要能在拆分阶段独立成段，不能只识别第一对。"""
        assert split_text_at_brackets(f"正文{open_b}动作{close_b}继续") == [
            "正文",
            f"{open_b}动作{close_b}",
            "继续",
        ]


class TestCapSegments:
    def test_join_tail(self) -> None:
        assert cap_segments(["a", "b", "c"], max_segments=2) == ["a", "bc"]

    def test_non_positive_max(self) -> None:
        assert cap_segments(["a", "b"], max_segments=0) == ["a", "b"]

    def test_max_one_merges_all(self) -> None:
        assert cap_segments(["a", "b", "c"], max_segments=1) == ["abc"]


class TestStageInternals:
    def test_stage_lines_no_fallback_needed(self) -> None:
        """_stage_lines 不需要兜底：对 strip 后的非空串按换行拆分，首块必非空。"""
        assert _stage_lines(["hello"]) == ["hello"]
        assert _stage_lines(["a\nb"]) == ["a", "b"]
        assert _stage_lines(["x\r\ny"]) == ["x", "y"]
        assert _stage_lines(["a\n\n\nb"]) == ["a", "b"]

    def test_stage_sentences_fallback_reachable(self) -> None:
        """纯标点段拆不出非空句，靠兜底原样保留（对比 `_stage_lines`：换行拆分
        对 strip 后的非空串总产出非空首块，不需要兜底）。"""
        assert _stage_sentences(["。。。"]) == ["。。。"]
        assert _stage_sentences(["？！"]) == ["？！"]


class TestPromptAndDelay:
    def test_prompt_contains_text_and_style(self) -> None:
        prompt = build_segmentation_prompt("原文内容", "active", 4)
        assert "原文内容" in prompt
        assert "最多分成 4 条" in prompt
        assert "活泼" in prompt

    def test_delay_bounds_with_rng(self) -> None:
        rng = random.Random(0)
        delay = calculate_send_delay("abcd", 1.0, 0.1, 1.2, rng=rng)
        assert 0.0 <= delay <= 1.2

    def test_delay_monotonic_without_jitter_ceiling(self) -> None:
        class ZeroRng:
            def uniform(self, a: float, b: float) -> float:
                return 0.0

        short = calculate_send_delay("a", 0.5, 0.1, 10.0, rng=ZeroRng())  # type: ignore[arg-type]
        long = calculate_send_delay("aaaaa", 0.5, 0.1, 10.0, rng=ZeroRng())  # type: ignore[arg-type]
        assert long > short

    def test_delay_clamped_when_max_below_base(self) -> None:
        delay = calculate_send_delay("abc", 5.0, 0.0, 1.0)
        assert delay <= 1.0

    def test_delay_exact_value_with_zero_rng(self) -> None:
        """delay = base + len*per_char（rng 注入 0 抖动时的精确值）。"""

        class ZeroRng:
            def uniform(self, a: float, b: float) -> float:
                return 0.0

        assert calculate_send_delay("abcd", 1.0, 0.1, 9.0, rng=ZeroRng()) == 1.4  # type: ignore[arg-type]

    def test_delay_jitter_stays_within_bound(self) -> None:
        """抖动必须落在 [0, DELAY_JITTER_SECONDS] 内。"""
        base, per_char, delay_max = 1.0, 0.1, 9.0
        for seed in range(50):
            rng = random.Random(seed)
            delay = calculate_send_delay("abcd", base, per_char, delay_max, rng=rng)
            assert base + 4 * per_char <= delay <= base + 4 * per_char + DELAY_JITTER_SECONDS

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

    def test_schema_defaults_match_dataclass(self) -> None:
        """Schema default values must match SegmentationSettings defaults."""
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        defaults = SegmentationSettings()
        for field in fields(SegmentationSettings):
            assert schema[field.name]["default"] == getattr(defaults, field.name), field.name
