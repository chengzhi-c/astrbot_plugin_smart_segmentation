from __future__ import annotations

import json
import random
import re
from typing import Any

from .bounds import DELAY_JITTER_SECONDS

THINKING_TAG_RE = re.compile(
    r"<(?P<tag>think|thinking)\b[^>]*>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
UNCLOSED_THINKING_TAG_RE = re.compile(
    r"^\s*<(?:think|thinking)\b[^>]*>.*?(?:$|(?=\n{2,}|\r\n\r\n))",
    flags=re.IGNORECASE | re.DOTALL,
)
THINKING_BOUNDARY_RE = re.compile(
    r"</?(?:think|thinking)\b[^>]*>",
    flags=re.IGNORECASE,
)


class SegmentationParseError(ValueError):
    """Raised when model output cannot be parsed into usable segments."""

BRACKET_PAIRS: tuple[tuple[str, str], ...] = (
    ("（", "）"),
    ("(", ")"),
    ("【", "】"),
    ("[", "]"),
)
STRONG_BOUNDARY_MARKS = "。？！~…!?"
WEAK_BOUNDARY_MARKS = "，,；;、 "
SENTENCE_BOUNDARY_RE = re.compile(
    rf"[^{re.escape(STRONG_BOUNDARY_MARKS)}\n]+[{re.escape(STRONG_BOUNDARY_MARKS)}]*",
)


def _strip_single_trailing_period(text: str) -> str:
    """Only strip exactly one trailing '。', preserving intentional repetition like '。。。'."""
    if text.endswith("。") and not text.endswith("。。"):
        return text[:-1]
    return text

STYLE_GUIDES = {
    "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
    "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
    "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。",
}


def cap_segments(segments: list[str], *, max_segments: int) -> list[str]:
    if max_segments <= 0 or len(segments) <= max_segments:
        return list(segments)
    head = segments[: max_segments - 1]
    tail = "".join(segments[max_segments - 1 :])
    return head + [tail]


def strip_thinking_content(text: str) -> str:
    if not text:
        return ""
    cleaned_text = THINKING_TAG_RE.sub("", str(text))
    cleaned_text = UNCLOSED_THINKING_TAG_RE.sub("", cleaned_text)
    cleaned_text = THINKING_BOUNDARY_RE.sub("", cleaned_text)
    return cleaned_text.strip()


def extract_json_array_text(raw_text: str) -> str:
    result_text = str(raw_text or "").strip()
    candidates: list[str] = []
    fenced_json = re.search(
        r"```\s*json\s*(.*?)```",
        result_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_json is not None:
        candidates.append(fenced_json.group(1).strip())
    start = result_text.find("[")
    end = result_text.rfind("]")
    if start != -1 and end != -1 and start < end:
        candidates.append(result_text[start : end + 1])
    if "```" in result_text:
        candidates.append(result_text.split("```", 1)[1].split("```", 1)[0].strip())

    for candidate in candidates:
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        return candidate
    return result_text


def is_action_only_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False

    for open_bracket, close_bracket in BRACKET_PAIRS:
        if not stripped.startswith(open_bracket) or not stripped.endswith(close_bracket):
            continue
        depth = 0
        for index, char in enumerate(stripped):
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:
                    return index == len(stripped) - 1
    return False


def has_unbalanced_brackets(text: str) -> bool:
    for open_bracket, close_bracket in BRACKET_PAIRS:
        if text.count(open_bracket) != text.count(close_bracket):
            return True
    return False


def merge_segments_balancing_brackets(segments: list[str]) -> list[str]:
    if not segments:
        return list(segments)

    merged: list[str] = []
    buffer = ""
    for segment in segments:
        buffer = buffer + segment if buffer else segment
        if not has_unbalanced_brackets(buffer):
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    return merged


def split_text_at_brackets(text: str) -> list[str]:
    if not text:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        matched_pair: tuple[str, str] | None = None
        for open_bracket, close_bracket in BRACKET_PAIRS:
            if char == open_bracket:
                matched_pair = (open_bracket, close_bracket)
                break

        if matched_pair is None:
            buffer.append(char)
            index += 1
            continue

        open_bracket, close_bracket = matched_pair
        depth = 1
        scan_index = index + 1
        while scan_index < len(text) and depth > 0:
            if text[scan_index] == open_bracket:
                depth += 1
            elif text[scan_index] == close_bracket:
                depth -= 1
            scan_index += 1

        if depth != 0:
            buffer.append(text[index:])
            index = len(text)
            break

        if buffer:
            parts.append("".join(buffer))
            buffer = []
        parts.append(text[index:scan_index])
        index = scan_index

    if buffer:
        parts.append("".join(buffer))

    return parts


# ── 内部管道 stage（不对外暴露）──────────────────────────────


def _stage_bracket(segments: list[str]) -> list[str]:
    """对每段在括号边界拆分；括号整体不拆。"""
    result: list[str] = []
    for seg in segments:
        for part in split_text_at_brackets(seg):
            p = part.strip()
            if p:
                result.append(p)
    return result


def _stage_lines(segments: list[str]) -> list[str]:
    """对每段按换行拆分；action-only 段保持完整。"""
    result: list[str] = []
    for seg in segments:
        s = str(seg or "").strip()
        if not s:
            continue
        if is_action_only_text(s):
            result.append(s)
            continue
        lines = [line.strip() for line in re.split(r"\r?\n+", s) if line.strip()]
        result.extend(lines)
    return result


def _stage_sentences(segments: list[str]) -> list[str]:
    """对每段按句读符拆分；action-only 段保持完整。"""
    result: list[str] = []
    for seg in segments:
        if is_action_only_text(seg):
            result.append(seg)
            continue
        sentences = [
            _strip_single_trailing_period(m.group(0).strip())
            for m in SENTENCE_BOUNDARY_RE.finditer(seg)
            if _strip_single_trailing_period(m.group(0).strip())
        ]
        result.extend(sentences or [seg])
    return result


def split_text_locally(text: str, *, max_segments: int) -> list[str]:
    cleaned = strip_thinking_content(text).strip()
    if not cleaned:
        return []

    pipeline = _stage_sentences(_stage_lines(_stage_bracket([cleaned])))
    return cap_segments(pipeline, max_segments=max_segments)


def normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = []
    for segment in segments:
        if not isinstance(segment, str):
            continue
        normalized_segment = segment.strip()
        if normalized_segment:
            normalized_segments.append(normalized_segment)
    if not normalized_segments:
        raise ValueError("模型返回的分段结果为空")

    return cap_segments(normalized_segments, max_segments=max_segments)


def build_segmentation_prompt(text: str, style: str, max_segments: int) -> str:
    style_guide = STYLE_GUIDES.get(style, STYLE_GUIDES["natural"])
    return f"""你正在模拟一个人用手机聊天。下面是 ta 想说的内容，请把它分成几条消息，就像真人会怎么一条一条发出来那样。

{style_guide}

规则：
- 不要改写原意，不要补充新信息
- 去掉每条消息末尾的句号「。」
- 保留感叹号、问号、省略号、波浪号等有情绪的标点
- 不要每个逗号都拆开，相关的内容放在一条里
- 消息长短可以不均匀
- 不要在单个数组元素内部使用换行；如果你想换行，就拆成数组中的下一项
- 括号（中文「（）」「【】」或英文「()」「[]」）内的内容（动作、神态、旁白等描述）必须作为独立的一条消息单独发送，不要和括号外的正文合在同一条
- 括号内的内容本身不能再拆开，需保持完整
- 如果整段内容就是被括号包裹的动作/神态描述，直接整段返回不再切分
- 最多分成 {max_segments} 条
- 如果不适合切分，就返回只包含原文的一项数组

原文：{text}

只返回 JSON 数组，如 ["消息1", "消息2"]"""


def parse_segments_from_model_output(
    raw_text: str,
    *,
    max_segments: int,
) -> list[str]:
    try:
        json_text = extract_json_array_text(raw_text)
        segments = json.loads(json_text)
        normalized = normalize_segments(segments, max_segments=max_segments)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise SegmentationParseError("模型返回的分段结果无法解析") from exc

    balanced = merge_segments_balancing_brackets(normalized)
    pipeline = _stage_lines(_stage_bracket(balanced))
    return cap_segments(pipeline, max_segments=max_segments)


def calculate_send_delay(
    segment: str,
    delay_base: float,
    delay_per_char: float,
    delay_max: float,
    *,
    rng: random.Random | None = None,
) -> float:
    """Compute follow-up delay; rng injects jitter for tests."""
    roller = rng if rng is not None else random
    normalized_delay = delay_base + len(segment) * delay_per_char
    normalized_delay += roller.uniform(0.0, DELAY_JITTER_SECONDS)
    return max(0.0, min(delay_max, normalized_delay))
