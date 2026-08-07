from __future__ import annotations

import json
import random
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
from astrbot_plugin_smart_segmentation.segmentation import (
    SegmentationParseError,
    build_segmentation_prompt,
    calculate_send_delay,
    cap_segments,
    extract_json_array_text,
    is_action_only_text,
    merge_segments_balancing_brackets,
    parse_segments_from_model_output,
    split_text_locally,
    strip_thinking_content,
)


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


class TestJsonExtractAndParse:
    def test_plain_array(self) -> None:
        assert extract_json_array_text('["a", "b"]') == '["a", "b"]'

    def test_fenced_json(self) -> None:
        raw = '废话\n```json\n["一", "二"]\n```\n尾'
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


class TestTrailingPeriodAndLocalSplit:
    def test_local_strips_single_period(self) -> None:
        segs = split_text_locally("你好。世界。", max_segments=5)
        assert segs == ["你好", "世界"]

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

    def test_merge_unbalanced(self) -> None:
        assert merge_segments_balancing_brackets(["（前", "后）", "完"]) == ["（前后）", "完"]


class TestCapSegments:
    def test_no_op(self) -> None:
        assert cap_segments(["a", "b"], max_segments=5) == ["a", "b"]

    def test_join_tail(self) -> None:
        assert cap_segments(["a", "b", "c"], max_segments=2) == ["a", "bc"]

    def test_non_positive_max(self) -> None:
        assert cap_segments(["a", "b"], max_segments=0) == ["a", "b"]


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
