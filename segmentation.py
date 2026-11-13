from __future__ import annotations

import json
import random
import re
from typing import Any

from .bounds import DELAY_JITTER_SECONDS

# 标签的属性段必须限字符集与长度：[^>]* 能一路吃到下一个 ">"（跨行、跨整段
# 正文），把可见内容当属性删掉。
# 属性只收窄到「不会出现在正文散文里」的字符集：放开后正文里的
# 「格式是 <think 然后写推理 > 结束」这类写法会被当成标签，把中间的真实内容删掉。
# 代价是属性含中文/换行等字符的 think 标签识别不了（开标签与推理原文会留在正文里），
# 这里宁可漏标签也不删可见内容。
_THINK_ATTR = r"[A-Za-z0-9_=:\"'/. -]{0,64}"
# 每个标签名一条正则。用反向引用 </(?P=tag)> 时，名字不匹配（模型常混用
# <think> 与 </thinking>）会让 .*? 在 limit 内逐位回溯，仍是 O(n^2)。
THINKING_PAIR_RES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"<think\b{_THINK_ATTR}>.*?</think\s*>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(
        rf"<thinking\b{_THINK_ATTR}>.*?</thinking\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    ),
)
THINKING_CLOSE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"</think\s*>", flags=re.IGNORECASE),
    re.compile(r"</thinking\s*>", flags=re.IGNORECASE),
)
UNCLOSED_THINKING_OPEN_RE = re.compile(
    rf"^\s*<(?:think|thinking)\b{_THINK_ATTR}>",
    flags=re.IGNORECASE,
)
BLANK_LINE_RE = re.compile(r"\r?\n[ \t]*\r?\n")
THINKING_BOUNDARY_RE = re.compile(
    rf"</?(?:think|thinking)\b{_THINK_ATTR}>",
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


def _strip_trailing_periods(segments: list[str]) -> list[str]:
    """最终每条消息只去一个句末「。」。

    必须在 cap_segments 合并之后做：合并进来的句子在消息中间，那里的句号是内容。
    """
    stripped = [_strip_single_trailing_period(segment) for segment in segments]
    # 段本身就是「。」时去句号会得到空串：补发会对空串各发一条消息，直接滤掉。
    return [segment for segment in stripped if segment]


# 尾部不能用 \S+：中文不加空格，URL 后面的句读符和整段正文会被当成 URL 一起
# 遮掉，那一行就再也切不动。中文若紧贴 URL（无句读/空白）则一起被遮，不切也不丢字。
_URL_RE = re.compile(
    # scheme 部分必须限长:无空格长串(长 base64、哈希、单行英文)下 `*` 会在
    # 每个起始位置回溯到串尾,退化成平方(实测 8 万字符 9.8s)。
    r"[A-Za-z][A-Za-z0-9+.\-]{0,31}://[^\s<>\"'\u3000-\u303f\uff00-\uffef]+",
)
_URL_MASK_RE = re.compile("\ue000(\\d+)\ue001")


def _mask_urls(text: str) -> tuple[str, list[str]]:
    """把 URL 换成不含标点的占位符。

    半角 ? 与 ! 在 STRONG_BOUNDARY_MARKS 里，不保护就会把链接拆到两条消息。
    """
    urls: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f"\ue000{len(urls) - 1}\ue001"

    return _URL_RE.sub(_replace, text), urls


def _unmask_urls(text: str, urls: list[str]) -> str:
    if not urls:
        return text
    def _restore(match: re.Match[str]) -> str:
        index = int(match.group(1))
        # 正文里恰好出现同形的私用区字符时不能炸：原样保留那一段
        return urls[index] if 0 <= index < len(urls) else match.group(0)

    return _URL_MASK_RE.sub(_restore, text)


STYLE_GUIDES = {
    "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
    "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
    "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。",
}


def cap_segments(
    segments: list[str],
    *,
    max_segments: int,
    joiner: str,
) -> list[str]:
    """合并超出上限的尾部段。

    `joiner` 必须是上游 stage 拆掉的分隔符：`_stage_lines` 按换行拆分并丢弃
    分隔符，直接空串回并会把 markdown 列表/分步说明黏成一行。
    """
    max_segments = max(1, max_segments)
    if len(segments) <= max_segments:
        return list(segments)
    head = segments[: max_segments - 1]
    tail = joiner.join(segments[max_segments - 1 :])
    return head + [tail]


def _sub_before(pattern: re.Pattern[str], text: str, limit: int) -> str:
    """pattern 的匹配不可能越过 limit：限制替换范围可避免未闭合开标签让每个
    起始位置都回溯到文末（O(n^2)，阻塞事件循环）。"""
    return pattern.sub("", text[:limit]) + text[limit:]


def strip_thinking_content(text: str) -> str:
    if not text:
        return ""
    cleaned_text = str(text)
    # 成对标签只可能出现在最后一个闭合标签之前：把成对替换限制在该前缀内即可，
    # 闭合标签之后的大量未闭合开标签不会再让每个起始位置都回溯一遍。
    for pair_pattern, close_pattern in zip(THINKING_PAIR_RES, THINKING_CLOSE_RES):
        last_close_end = 0
        for match in close_pattern.finditer(cleaned_text):
            last_close_end = match.end()
        if last_close_end:
            cleaned_text = _sub_before(pair_pattern, cleaned_text, last_close_end)

    # 段首的未闭合开标签：推理区延伸到第一个闭合标签 / 空行 / 文末。既没有空行
    # 也没有闭合标签时不能把后文整段当成推理删掉（那等于把可见回复删空），只去掉
    # 标签本身。
    unclosed = UNCLOSED_THINKING_OPEN_RE.match(cleaned_text)
    if unclosed is not None:
        body = cleaned_text[unclosed.end():]
        cut = len(body)
        for close_pattern in THINKING_CLOSE_RES:
            first_close = close_pattern.search(body)
            if first_close is not None:
                cut = min(cut, first_close.start())
        blank_line = BLANK_LINE_RE.search(body)
        if blank_line is not None:
            cut = min(cut, blank_line.start())
        # 没有闭合标签也没有空行 = 拿不到推理边界，只去掉标签本身，别把正文删空。
        cleaned_text = body[cut:] if cut < len(body) else body

    cleaned_text = _sub_before(
        THINKING_BOUNDARY_RE,
        cleaned_text,
        cleaned_text.rfind(">") + 1,
    )
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
    # 首个 "[" 之前若有别的 "["（对象包装、前缀引用），first..last 会拼成非法 JSON。
    # 兜底再试一次最后一个 "["：放在 fence 之后，保留 fenced 数组的优先级。
    last_start = result_text.rfind("[")
    if last_start != -1 and end != -1 and last_start < end:
        candidates.append(result_text[last_start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, RecursionError):
            continue
        if isinstance(parsed, list):
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
    """仅当左括号多于右括号（含未闭合）时判定为不均衡。

    右括号过剩（模型多输出的孤立右括号、`:)` 类表情）不需要向后合并段来配平，
    判为不均衡会把后续所有段粘成一条。
    """
    for open_bracket, close_bracket in BRACKET_PAIRS:
        depth = 0
        for char in text:
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
        if depth > 0:
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
        lines = [line.strip() for line in re.split(r"(?:\r\n|\r|\n)+", s) if line.strip()]
        result.extend(lines)
    return result


def _stage_sentences(segments: list[str]) -> list[str]:
    """对每段按句读符拆分；action-only 段保持完整。"""
    result: list[str] = []
    for seg in segments:
        if is_action_only_text(seg):
            result.append(seg)
            continue
        # 段首句读符无法被 SENTENCE_BOUNDARY_RE 匹配（匹配必须以非句读符开头），
        # 先剥离再拼回首句，否则这些字符会被静默丢弃。
        rest = seg.lstrip(STRONG_BOUNDARY_MARKS)
        leading_marks = seg[: len(seg) - len(rest)]
        sentences: list[str] = []
        for match in SENTENCE_BOUNDARY_RE.finditer(rest):
            sentence = match.group(0).strip()
            if sentence:
                sentences.append(sentence)
        if not sentences:
            if seg:
                result.append(seg)
            continue
        sentences[0] = leading_marks + sentences[0]
        result.extend(sentences)
    return result


def split_text_locally(text: str, *, max_segments: int) -> list[str]:
    cleaned = strip_thinking_content(text).strip()
    if not cleaned:
        return []

    # URL 要先遮罩再做所有 stage：括号 stage 会把 URL 里的 () 当成结构拆开
    masked, urls = _mask_urls(cleaned)
    pipeline = cap_segments(
        _stage_sentences(_stage_lines(_stage_bracket([masked]))),
        max_segments=max_segments,
        joiner="\n",
    )
    return [_unmask_urls(segment, urls) for segment in _strip_trailing_periods(pipeline)]


_CONTENT_IGNORED = ("。",)


def _content_signature(text: str) -> str:
    return "".join(
        char for char in text if not char.isspace() and char not in _CONTENT_IGNORED
    )


def same_content(segments: list[str], text: str) -> bool:
    """分段只能切不能改:拼回去(忽略空白与「。」)必须与原文一致。

    提示词要求「不要改写」，但模型漏字/合并是真实失败模式；不校验的话用户会收到
    一条被模型改残的回复。忽略「。」(含句中的)是因为每条消息末尾的句号都会被剥掉。
    """
    return _content_signature("".join(segments)) == _content_signature(text)


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

    return cap_segments(normalized_segments, max_segments=max_segments, joiner="\n")


def build_segmentation_prompt(text: str, style: str, max_segments: int) -> str:
    style_guide = STYLE_GUIDES[style]
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
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise SegmentationParseError("模型返回的分段结果无法解析") from exc

    balanced = merge_segments_balancing_brackets(normalized)
    pipeline = _stage_lines(_stage_bracket(balanced))
    return _strip_trailing_periods(
        cap_segments(pipeline, max_segments=max_segments, joiner="\n"),
    )


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
