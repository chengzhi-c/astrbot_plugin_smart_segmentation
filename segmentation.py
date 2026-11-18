from __future__ import annotations

import json
import random
import re
from itertools import pairwise
from typing import Any

from .bounds import DELAY_JITTER_SECONDS

# 标签的属性段必须限字符集与长度：[^>]* 能一路吃到下一个 ">"（跨行、跨整段
# 正文），把可见内容当属性删掉。
# 属性只收窄到「不会出现在正文散文里」的字符集：放开后正文里的
# 「格式是 <think 然后写推理 > 结束」这类写法会被当成标签，把中间的真实内容删掉。
# 代价是属性含中文/换行等字符的 think 标签识别不了（开标签与推理原文会留在正文里），
# 这里宁可漏标签也不删可见内容。
_THINK_ATTR = r"[A-Za-z0-9_=:\"'/. -]{0,64}"
_THINKING_TAG_RE = re.compile(
    rf"<(?P<close>/)?(?P<name>think(?:ing)?)\b{_THINK_ATTR}[ \t]*>",
    flags=re.IGNORECASE,
)


class SegmentationParseError(ValueError):
    """Raised when model output cannot be parsed into usable segments."""


class _ThinkingFilter:
    def __init__(self) -> None:
        self.depth = 0
        self.tag = ""
        self.fence: tuple[str, int] | None = None
        self.fence_header = False
        self.inline_ticks = 0
        self.inline_pending: list[str] | None = None
        self.marker = ""
        self.marker_length = 0
        self.line_indent: int | None = 0
        self.marker_at_line_start = False
        self.closing_fence = False

    def _release_inline(self, output: list[str], *, literal: bool) -> None:
        if self.inline_pending is None:
            return
        pending = "".join(self.inline_pending)
        self.inline_pending = None
        if literal:
            output.append(pending)
        else:
            # 尾文已完整，只保留确实闭合的行内代码，避免回放再次建立待确认缓冲。
            output.append(self.feed(pending, complete_inline=True))
            if self.tag:
                candidate, self.tag = self.tag, ""
                self._write_plain(candidate, output)

    def _finish_marker(self, output: list[str]) -> None:
        if not self.marker:
            return
        if self.fence is not None and self.fence_header:
            if self.fence[0] == "`" and self.marker == "`":
                opened_length = self.fence[1]
                self.fence = None
                self.fence_header = False
                self.inline_ticks = (
                    0 if self.marker_length == opened_length else opened_length
                )
                if not self.inline_ticks:
                    self._release_inline(output, literal=True)
        elif self.fence is not None:
            self.closing_fence = (
                self.marker_at_line_start
                and self.marker == self.fence[0]
                and self.marker_length >= self.fence[1]
            )
        elif self.inline_ticks:
            if self.marker == "`" and self.marker_length == self.inline_ticks:
                self.inline_ticks = 0
                self._release_inline(output, literal=True)
        elif self.marker_at_line_start and self.marker_length >= 3:
            self.fence = (self.marker, self.marker_length)
            self.fence_header = True
        elif self.marker == "`":
            self.inline_ticks = self.marker_length
        self.marker = ""
        self.marker_length = 0

    def _write_plain(self, text: str, output: list[str]) -> None:
        for char in text:
            if self.depth:
                continue
            if char in "`~":
                if self.marker == char:
                    self.marker_length += 1
                else:
                    self._finish_marker(output)
                    self.marker = char
                    self.marker_length = 1
                    self.marker_at_line_start = (
                        self.line_indent is not None and self.line_indent <= 3
                    )
            else:
                self._finish_marker(output)
                if char == "\n":
                    if self.fence_header:
                        self.fence_header = False
                        self._release_inline(output, literal=True)
                    if self.closing_fence:
                        self.fence = None
                    self.inline_ticks = 0
                    self._release_inline(output, literal=False)
                elif not char.isspace():
                    self.closing_fence = False
            (self.inline_pending if self.inline_pending is not None else output).append(
                char
            )
            if char == "\n":
                self.line_indent = 0
                self.closing_fence = False
            elif char == " " and self.line_indent is not None:
                self.line_indent += 1
            else:
                self.line_indent = None

    @staticmethod
    def _possible_tag(value: str) -> bool:
        value = value.casefold()
        return any(
            prefix.startswith(value) or value.startswith(prefix)
            for prefix in ("<think", "</think")
        )

    def _finish_tag(self, output: list[str], *, complete_inline: bool = False) -> None:
        candidate, self.tag = self.tag, ""
        match = _THINKING_TAG_RE.fullmatch(candidate)
        if match is None:
            self._write_plain(candidate, output)
            return

        closing = bool(match.group("close"))
        if self.depth:
            self.depth += -1 if closing else 1
        elif self.fence is not None and not self.fence_header:
            self._write_plain(candidate, output)
        elif self.inline_ticks or self.fence_header:
            if self.inline_pending is None and not complete_inline:
                self.inline_pending = []
            self._write_plain(candidate, output)
        elif not closing:
            self.depth = 1

    def feed(self, text: str, *, complete_inline: bool = False) -> str:
        output: list[str] = []
        code_spans = (
            iter(_INLINE_CODE_RE.finditer(text)) if complete_inline else iter(())
        )
        code_span = next(code_spans, None)
        for index, char in enumerate(text):
            while code_span is not None and index >= code_span.end():
                code_span = next(code_spans, None)
            if self.tag:
                if char == "<":
                    candidate, self.tag = self.tag, ""
                    self._write_plain(candidate, output)
                    self.tag = char
                else:
                    self.tag += char
                    if char == ">":
                        if complete_inline and (
                            code_span is None or index < code_span.start()
                        ):
                            self.inline_ticks = 0
                        self._finish_tag(output, complete_inline=complete_inline)
                    elif len(self.tag) > 80 or not self._possible_tag(self.tag):
                        candidate, self.tag = self.tag, ""
                        self._write_plain(candidate, output)
            elif char == "<":
                self._finish_marker(output)
                self.closing_fence = False
                self.tag = char
            else:
                self._write_plain(char, output)
        return "".join(output)

    def flush_pending(self) -> str:
        output: list[str] = []
        if self.tag:
            candidate, self.tag = self.tag, ""
            self._write_plain(candidate, output)
        self._finish_marker(output)
        if self.fence_header:
            self.fence = None
            self.fence_header = False
        self.inline_ticks = 0
        if self.inline_pending is not None:
            self._release_inline(output, literal=False)
        return "".join(output)

    def finish(self) -> str:
        output = [self.flush_pending()]
        self.depth = 0
        return "".join(output)


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


def has_visible_text(text: str) -> bool:
    """str.strip() 会保留零宽格式字符，它们单独成段时没有可见正文。"""
    return any(char.isprintable() and not char.isspace() for char in text)


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
    # 私用区占位符必须排除：三趟遮罩串行，后一趟把前一趟的占位符吞进原子后，
    # 还原时内层占位符会被当普通字符发出去，那段原文随之丢失。
    # ASCII 右括号也必须排除：不排除时 URL 原子会把链接/句子的闭括号连同紧邻中文
    # 一起吞掉，括号 stage 就看不到那对括号。代价是 URL 里合法的 ")" 被截断，
    # 闭括号留在正文里（内容不丢，只是可能少切一刀），这个取舍可以接受。
    r"[A-Za-z][A-Za-z0-9+.\-]{0,31}://"
    r"[^\s<>\"'\u3000-\u303f\uff00-\uffef\ue000-\ue001)]+",
)
_URL_MASK_RE = re.compile("\ue000(\\d+)\ue001")
# 围栏代码块整块不可切：行 stage 会逐行拆开并 strip 掉缩进，括号 stage 会把代码里的
# () 当成结构拆开。未闭合的围栏按「到文本末尾」处理。
# 闭围栏长度不得短于开围栏：否则 ```` 块里的内层 ``` 会被当成闭合，整块仍被拆碎。
_FENCE_RE = re.compile(
    r"^[ ]{0,3}(?P<fence>(?P<ticks>`{3,})|~{3,})"
    r"(?(ticks)(?!`)(?![^\n]*`)|(?!~))"
    r".*?(?:^[ ]{0,3}(?P=fence)(?(ticks)`*|~*)[ \t]*(?=\r?$)|\Z)",
    flags=re.DOTALL | re.MULTILINE,
)
# markdown 表格：连续的「行首到行尾都是一个竖线对」的行算一张表。逐行拆开会把
# 一张表拆成 N 条消息，行内的「。」还会把它切成两半、留下一条只有 `|` 的碎片。
# 只认首尾都有竖线的形态：省掉尾竖线的写法与正文里的 `a | b` 无法区分。
# 尾部那一支吃「缓冲区最后一行还没收到行尾竖线」的形态——流式下 chunk 边界必然
# 产生它，不并进同一张表的话那行的句号就是切点。`\r` 要单独放进字符类：MULTILINE
# 的 `$` 停在 `\n` 之前，CRLF 文本的行尾是 `\r`，不认就整张表漏遮。
_TABLE_RE = re.compile(
    r"(?:(?:^[ \t]*\|[^\n\ue000-\ue001]*\|[ \t\r]*$\n)+"
    r"(?:^[ \t]*\|[^\n\ue000-\ue001]*\Z)?"
    r"|^[ \t]*\|[^\n\ue000-\ue001]*\Z)",
    flags=re.MULTILINE,
)

# 行内代码以相同长度的反引号包裹；更短的反引号 run 可以出现在内容里。
_INLINE_CODE_RE = re.compile(
    r"(?<!`)(?P<fence>`+)(?!`)"
    r"(?:(?!(?P=fence)(?!`))[^\n\ue000-\ue001])+?(?P=fence)(?!`)"
)

# 流式专用：缓冲区末尾那行还没收到闭合反引号，_INLINE_CODE_RE 匹配不到它，
# 代码里的 ! ? 就成了切点。遮到缓冲区末尾当作未闭合单元——与围栏、表格尾部
# 的 \Z 兜底同一思路。整段末尾没有闭合反引号时也会命中，这是刻意的：漏遮的
# 代价是一条坏消息，误遮的代价只是这一次不切。
_INLINE_CODE_OPEN_TAIL_RE = re.compile(
    r"(?<!`)(?P<fence>`+)(?!`)"
    r"(?:(?!(?P=fence)(?!`))[^\n\ue000-\ue001])*\Z"
)

# 流式专用：缓冲区分两次到达的链接，两支分别对应「还没收到 `]`」与「还没收到 `)`」。
# 只认这两种未闭合形态：正文里成对的 `[1]`、`[x]` 不匹配，否则它们后面的句号全成死区，
# 切点丢光、退化成按字数硬切。
_MD_LINK_OPEN_TAIL_RE = re.compile(
    r"(?:\[[^\]\n]{0,200}\Z|\[[^\]\n]{0,200}\]\([^)\n]{0,300}\Z)"
)
# markdown 链接连 [] () 一起遮：只遮 URL 时括号 stage 仍会把链接拆成三截。
# 两个字符类限长：满屏 "[" 时无界回溯会在每个起始位置吃到串尾（平方）。
# 同样排除私用区占位符，理由见 _URL_RE。
# 尾部那一支是 CommonMark 的标题链接：`[文档](url "标题")`。URL 后面的空白
# 让它无法命中前面的分支，漏遮之后括号 stage 会把链接拆成两截。
_MD_LINK_RE = re.compile(
    r"\[[^\]\n\ue000-\ue001]{0,200}\]"
    r"\([^)\s\ue000-\ue001]{0,2048}(?:\s+[^)\n\ue000-\ue001]{0,200})?\)"
)

# 降级链接扫描的终止符：链接文本里不能有 "]"、换行或占位符；URL 部分到 ")"、
# 空白或占位符为止（与 _MD_LINK_RE 的字符类一致）。交给正则是因为逐字符循环
# 扫 8 万字符要几毫秒，C 层扫描没有这个成本。
# 链接从 "(" 之后到收尾括号之间允许一段标题（CommonMark）：URL 不含空白，
# 之后可以有 ` "标题"` 或 ` '标题'`，再是 ")"。两支共用同一判定，
# 否则正则主分支与这两处兜底扫描会对同一个链接给出不同边界。
# 限长取「无上界」会退化成平方回溯，所以用一个远大于正文的常数；兜底扫描
# 本来就是给超过 _MD_LINK_RE 限长的链接用的，这里不能比它更窄。
_LINK_TAIL_RE = re.compile(
    r"[^)\s\ue000\ue001]{1,20000}(?:\s+[^)\n\ue000\ue001]{0,200})?\)"
)
_LINK_TEXT_STOP_RE = re.compile(r"[\]\n\ue000\ue001]")


def _long_link_spans(text: str) -> list[tuple[int, int]]:
    """返回限长链接正则（``_MD_LINK_RE``）漏掉的超长 Markdown 链接区间。

    提高 ``_MD_LINK_RE`` 的限长不可行（满屏 "[" 时无界回溯是平方），这里只用
    单向游标：pos 只前进，每段文字至多被扫一次，整体线性。
    """
    if "](" not in text or "[" not in text:
        return []

    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        close = text.find("](", pos)
        if close < 0:
            break
        open_bracket = text.rfind("[", pos, close)
        blocked = open_bracket < 0 or (
            _LINK_TEXT_STOP_RE.search(text, open_bracket + 1, close) is not None
        )
        if blocked:
            pos = close + 2
            continue
        # 与 _MD_LINK_RE 同义：URL 到空白或 ")" 为止，可带一段标题。
        # 没有任何收尾括号时，正则的 20000 字符上限会让每个 `](` 重扫同一
        # 个尾部；此时后面不可能再有可闭合链接，直接结束扫描。
        closing = text.find(")", close + 2)
        if closing < 0:
            break
        tail_match = _LINK_TAIL_RE.match(text, close + 2)
        if tail_match is None:
            # 当前候选已经覆盖到最近的收尾括号；跳过它，避免 malformed `](`
            # 在同一长尾上反复回溯。
            pos = closing + 1
            continue
        end = tail_match.end() - 1
        spans.append((open_bracket, end + 1))
        pos = end + 1
    return spans


def _mask_long_links(text: str, atoms: list[str]) -> str:
    """给限长链接正则漏掉的超长链接整段遮罩。

    退回「只遮 URL」不行：[] 与 () 仍是结构字符，链接会被括号 stage 拆成多条。
    """
    spans = _long_link_spans(text)
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        atoms.append(text[start:end])
        parts.append(text[cursor:start])
        parts.append(f"\ue000{len(atoms) - 1}\ue001")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def mask_uncuttable_spans(
    text: str, *, open_tails: bool = False
) -> list[tuple[int, int]]:
    """按与 ``_mask_atoms`` 相同的顺序返回不可切原子区间。

    ``open_tails`` 供流式路径使用：缓冲区末尾未闭合的行内代码与 Markdown 链接
    也按整块处理，否则分块到达时切点会落进链接或反引号内部。两份调用共用同一
    规则，避免本地分段与流式分段对同一段文本给出不同边界。
    """
    spans: list[tuple[int, int]] = []
    marks = list(text)

    def quote(start: int, end: int) -> None:
        if end < len(text):
            # 尾随换行留着当切点：遮进来会把表格/围栏后面的正文黏成同一段。
            # 原子一直延伸到文本末尾时不修剪——那个换行还在块内，是代码/表格的
            # 一部分（未闭合围栏尤其如此），剪掉会让流式在块中间切一刀。
            body = text[start:end].rstrip("\r\n")
            end = start + len(body)
        spans.append((start, end))
        marks[start:end] = '"' * (end - start)

    patterns = [_FENCE_RE, _TABLE_RE, _INLINE_CODE_RE]
    if open_tails:
        patterns.append(_INLINE_CODE_OPEN_TAIL_RE)
    for pattern in patterns:
        for match in pattern.finditer("".join(marks)):
            quote(*match.span())
    if _NESTED_OPEN_RE.search("".join(marks)):
        for start, end in _bracket_link_spans("".join(marks)):
            quote(start, end)
    else:
        for match in _MD_LINK_RE.finditer("".join(marks)):
            quote(*match.span())
        if open_tails:
            for match in _MD_LINK_OPEN_TAIL_RE.finditer("".join(marks)):
                quote(*match.span())
        for start, end in _long_link_spans("".join(marks)):
            quote(start, end)
    for match in _URL_RE.finditer("".join(marks)):
        quote(*match.span())
    return sorted(spans)


# 链接文本里又出现 "["：外层链接的真实边界只有方括号平衡扫描判得出来，_MD_LINK_RE
# 会在内层链接的 "](...)" 处收口。判定用 C 层正则，满屏 "[" 上逐字循环是平方。
_NESTED_OPEN_RE = re.compile(r"\[[^\]\n\ue000\ue001]*\[")
_BRACKET_RE = re.compile(r"[\[\]]")


def _bracket_link_spans(text: str) -> list[tuple[int, int]]:
    """按方括号平衡返回正则收不住的 Markdown 链接区间。"""
    opens: list[int] = []
    close_of: dict[int, int] = {}
    stack: list[int] = []
    for match in _BRACKET_RE.finditer(text):
        index = match.start()
        if text[index] == "[":
            stack.append(index)
            opens.append(index)
        elif stack:
            close_of[stack.pop()] = index
    if not close_of:
        return []
    spans: list[tuple[int, int]] = []
    skip_until = 0
    # opens 天然升序：外层先判，被吞掉的内层由 cursor 跳过
    for start in opens:
        if start < skip_until:
            continue
        close = close_of.get(start)
        if close is None or close + 1 >= len(text) or text[close + 1] != "(":
            continue
        # 收尾必须与 _MD_LINK_RE 同义：URL 到空白或 ")" 为止，可带一段标题。
        # 只按「第一个终止符是 )」判会把标题链接判成不合法，于是漏遮。
        closing = text.find(")", close + 2)
        if closing < 0:
            break
        if closing - (close + 2) > 20000:
            skip_until = closing + 1
            continue
        tail_match = _LINK_TAIL_RE.match(text, close + 2)
        if tail_match is None:
            skip_until = closing + 1
            continue
        if (
            text.find("\n", start, close) >= 0
            or text.find("\ue000", start, close) >= 0
            or text.find("\ue001", start, close) >= 0
        ):
            continue
        end = tail_match.end() - 1
        spans.append((start, end + 1))
        skip_until = end + 1
    return spans


def _mask_bracket_links(text: str, atoms: list[str]) -> str:
    """按方括号平衡判定链接边界，遮罩正则收不住的外层链接。"""
    spans = _bracket_link_spans(text)
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        atoms.append(text[start:end])
        parts.append(text[cursor:start])
        parts.append(f"\ue000{len(atoms) - 1}\ue001")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _mask_atoms(text: str, atoms: list[str]) -> str:
    """把不可切的原子换成不含标点的占位符，顺序为「围栏 → 表格 → 行内代码 → 链接 → 裸 URL」。

    半角 ? 与 ! 在 STRONG_BOUNDARY_MARKS 里，不保护就会把链接拆到两条消息。
    行内代码同理：``a!=b`` 或 ``foo()`` 被切断后两条消息的反引号都不配对，
    用户看到的是坏掉的 markdown。
    atoms 由调用方持有：模型路径逐段遮罩时要共用同一份索引。
    """

    def _replace(match: re.Match[str]) -> str:
        # 尾随换行不进原子：它是分隔符。把它一起遮掉，表格/围栏后面的正文就会
        # 和它们黏成一段，分段时同一条消息里塞进两段无关内容。
        chunk = match.group(0)
        body = chunk.rstrip("\r\n")
        trailing = chunk[len(body) :]
        atoms.append(body)
        return f"\ue000{len(atoms) - 1}\ue001{trailing}"

    text = _FENCE_RE.sub(_replace, text)
    text = _TABLE_RE.sub(_replace, text)
    # 围栏已整块换掉，剩下的反引号都是行内的。
    text = _INLINE_CODE_RE.sub(_replace, text)
    if _NESTED_OPEN_RE.search(text):
        text = _mask_bracket_links(text, atoms)
    else:
        text = _MD_LINK_RE.sub(_replace, text)
        text = _mask_long_links(text, atoms)
    return _URL_RE.sub(_replace, text)


def _unmask_atoms(text: str, atoms: list[str]) -> str:
    if not atoms:
        return text

    def _restore(match: re.Match[str]) -> str:
        index = int(match.group(1))
        # 正文里恰好出现同形的私用区字符时不能炸：原样保留那一段
        return atoms[index] if 0 <= index < len(atoms) else match.group(0)

    return _URL_MASK_RE.sub(_restore, text)


def _pick_escape_char(texts: list[str]) -> str | None:
    """Pick a character absent from the text so mask escapes stay unambiguous."""
    used = set("".join(texts))
    for char in ("\x00", "\ue002", "\ue003", "\ue004"):
        if char not in used:
            return char
    return None


def _escape_placeholder_chars(text: str, escape: str) -> str:
    """遮罩前先把正文自带的占位符同形字符转义掉。

    U+E000/U+E001 是占位符定界符，不转义的话还原阶段分不清「本函数写入的
    占位符」与正文：正文的「U+E000 0 U+E001」会被替换成 atoms[0]。
    """
    return text.replace("\ue000", escape + "0").replace("\ue001", escape + "1")


def _unescape_placeholder_chars(text: str, escape: str) -> str:
    return text.replace(escape + "0", "\ue000").replace(escape + "1", "\ue001")


def _pick_segment_boundary_marker(texts: list[str], escape: str) -> str | None:
    """Choose a private separator for model-array boundaries.

    The separator lets masking see fenced blocks that span array elements while still
    allowing bracket balancing to remove only the artificial boundary newline.
    """
    used = set("".join(texts))
    for marker in ("\ufdd0", "\ufdd1", "\x01", "\x02"):
        if marker not in used and marker != escape:
            return marker
    return None


def _join_model_segments(segments: list[str], marker: str) -> str:
    """Join model array elements while preserving meaningful block boundaries."""
    if not segments:
        return ""
    pieces = [segments[0]]
    scanner = _ThinkingFilter()
    scanner._write_plain(segments[0], [])
    scanner._finish_marker([])
    for previous, current in pairwise(segments):
        previous_is_table = previous.lstrip().startswith("|")
        link_boundary = previous.rstrip().endswith("]") and current.lstrip().startswith(
            "("
        )
        if scanner.fence is not None or previous_is_table:
            separator = "\n"
        elif scanner.inline_ticks or has_unbalanced_brackets(previous) or link_boundary:
            separator = ""
        else:
            separator = f"{marker}\n"
        pieces.append(separator + current)
        scanner._write_plain(separator + current, [])
        scanner._finish_marker([])
    return "".join(pieces)


STYLE_GUIDES = {
    "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
    "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
    "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。",
}


# 普通消息段的字符上限。平台对单条消息的上限差异很大（Discord 超过 2000
# 字直接静默截断），把整条尾巴并进一段会让插件自己产出一条超限消息——那正是用户
# 开分段要避免的。能装进上限的链接和代码块保持完整。
_MAX_SEGMENT_CHARS = 1500


def cap_segments(
    segments: list[str],
    *,
    max_segments: int,
    joiner: str,
) -> list[str]:
    """合并超出上限的尾部段。

    `joiner` 必须是上游 stage 拆掉的分隔符：`_stage_lines` 按换行拆分并丢弃
    分隔符，直接空串回并会把 markdown 列表/分步说明黏成一行。

    尾部合并受单段长度上限约束：内容确实装不进 `max_segments` 段时，允许段数
    超出配置值，而不是合并出一条超过平台上限的长消息。
    """
    if len(segments) <= max_segments:
        return list(segments)
    head = segments[: max_segments - 1]
    tails: list[str] = []
    group: list[str] = []
    group_len = 0
    for segment in segments[max_segments - 1 :]:
        extra = len(segment) + (len(joiner) if group else 0)
        if group and group_len + extra > _MAX_SEGMENT_CHARS:
            tails.append(joiner.join(group))
            group = [joiner + segment]
            group_len = extra
        else:
            group.append(segment)
            group_len += extra
    if group:
        tails.append(joiner.join(group))
    return head + tails


def split_oversized_segments(segments: list[str]) -> list[str]:
    """硬切超长消息；能装进上限的链接、代码等原子保持完整。"""
    result: list[str] = []
    for segment in segments:
        if len(segment) <= _MAX_SEGMENT_CHARS:
            result.append(segment)
            continue

        atoms: list[str] = []
        escape = _pick_escape_char([segment])
        if escape is None:
            result.extend(
                segment[start : start + _MAX_SEGMENT_CHARS]
                for start in range(0, len(segment), _MAX_SEGMENT_CHARS)
            )
            continue

        masked = _mask_atoms(_escape_placeholder_chars(segment, escape), atoms)
        # 超长原子本身无法保持完整又满足平台上限：只展开这些占位符，让后续
        # 硬切保住长度上限与全部正文；其余原子仍按完整占位符保护。
        for match in reversed(list(_URL_MASK_RE.finditer(masked))):
            atom_index = int(match.group(1))
            if len(atoms[atom_index]) > _MAX_SEGMENT_CHARS:
                masked = (
                    masked[: match.start()] + atoms[atom_index] + masked[match.end() :]
                )

        protected = sorted(
            [
                *_URL_MASK_RE.finditer(masked),
                *re.finditer(re.escape(escape) + r"[01]", masked),
            ],
            key=lambda match: match.start(),
        )
        chunk: list[str] = []
        chunk_length = 0

        def flush_chunk(
            chunk: list[str] = chunk,
            atoms: list[str] = atoms,
            escape: str = escape,
        ) -> None:
            nonlocal chunk_length
            if chunk:
                masked_chunk = "".join(chunk)
                restored = _unmask_atoms(masked_chunk, atoms)
                result.append(_unescape_placeholder_chars(restored, escape))
                chunk.clear()
                chunk_length = 0

        def append_plain(text: str, chunk: list[str] = chunk) -> None:
            nonlocal chunk_length
            while text:
                available = _MAX_SEGMENT_CHARS - chunk_length
                if available == 0:
                    flush_chunk()
                    available = _MAX_SEGMENT_CHARS
                part = text[:available]
                chunk.append(part)
                chunk_length += len(part)
                text = text[len(part) :]

        cursor = 0
        for match in protected:
            append_plain(masked[cursor : match.start()])
            token = match.group(0)
            atom_match = _URL_MASK_RE.fullmatch(token)
            token_length = len(atoms[int(atom_match.group(1))]) if atom_match else 1
            if chunk_length and chunk_length + token_length > _MAX_SEGMENT_CHARS:
                flush_chunk()
            chunk.append(token)
            chunk_length += token_length
            cursor = match.end()
        append_plain(masked[cursor:])
        flush_chunk()
    return result


def strip_thinking_content(text: str) -> str:
    if not text:
        return ""
    thinking_filter = _ThinkingFilter()
    return (thinking_filter.feed(str(text)) + thinking_filter.finish()).strip()


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
        if not stripped.startswith(open_bracket) or not stripped.endswith(
            close_bracket
        ):
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
        lines = [
            line.strip() for line in re.split(r"(?:\r\n|\r|\n)+", s) if line.strip()
        ]
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

    # URL 与代码块要先遮罩再做所有 stage：括号 stage 会把 URL 里的 () 当成结构拆开，
    # 行 stage 会把代码块逐行拆碎并抹掉缩进
    atoms: list[str] = []
    escape = _pick_escape_char([cleaned])
    if escape is None:
        return split_oversized_segments([cleaned])
    masked = _mask_atoms(_escape_placeholder_chars(cleaned, escape), atoms)
    pipeline = cap_segments(
        _stage_sentences(_stage_lines(_stage_bracket([masked]))),
        max_segments=max_segments,
        joiner="\n",
    )
    restored = [
        _unescape_placeholder_chars(_unmask_atoms(segment, atoms), escape)
        for segment in _strip_trailing_periods(pipeline)
    ]
    return split_oversized_segments(restored)


def same_content(segments: list[str], text: str) -> bool:
    """分段只能切不能改，允许每段末尾省略一个「。」。

    模型提示词要求去掉每条消息末尾的句号；普通文本允许空白差异，代码的内容和
    空白必须完全一致，句中句号仍是正文。
    """
    for pattern in (_FENCE_RE, _INLINE_CODE_RE):
        if [
            match.group(0)
            for segment in segments
            for match in pattern.finditer(segment)
        ] != [match.group(0) for match in pattern.finditer(text)]:
            return False

    source = [char for char in text if not char.isspace()]
    cursor = 0
    for index, segment in enumerate(segments):
        for char in segment:
            if char.isspace():
                continue
            if cursor >= len(source) or source[cursor] != char:
                return False
            cursor += 1
        next_char = (
            next(
                (char for char in segments[index + 1] if not char.isspace()),
                None,
            )
            if index + 1 < len(segments)
            else None
        )
        if cursor < len(source) and source[cursor] == "。" and next_char != "。":
            cursor += 1
    return cursor == len(source)


def normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = []
    for segment in segments:
        if not isinstance(segment, str):
            continue
        # 保留行首缩进：模型可能把围栏代码逐行放进数组，统一遮罩前需要这些空格。
        # 普通消息的边缘空白仍会在后续行 stage 清掉。
        normalized_segment = segment.rstrip()
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

    # 与本地路径同因：代码块里的 () 会被括号配平拆碎、换行会被行 stage 拆碎
    atoms: list[str] = []
    escape = _pick_escape_char(normalized)
    if escape is None:
        return split_oversized_segments(normalized)
    # 模型有时会把围栏/表格逐行放进数组；先拼回一份带行分隔的文本，才能跨元素
    # 识别这些不可拆原子。私有 marker 保留数组边界，括号配平时只移除人工换行。
    marker = _pick_segment_boundary_marker(normalized, escape)
    if marker is None:
        masked_segments = [
            _mask_atoms(_escape_placeholder_chars(segment, escape), atoms)
            for segment in normalized
        ]
    else:
        combined = _join_model_segments(normalized, marker)
        masked_combined = _mask_atoms(
            _escape_placeholder_chars(combined, escape),
            atoms,
        )
        # marker 落进代码块/表格等原子时会随原子保存；它只是数组边界，恢复原子
        # 前应去掉，原子内部真实换行仍保留。
        atoms[:] = [atom.replace(marker, "") for atom in atoms]
        masked_segments = masked_combined.split(f"{marker}\n")
    balanced = merge_segments_balancing_brackets(masked_segments)
    pipeline = _stage_lines(_stage_bracket(balanced))
    restored = [
        _unescape_placeholder_chars(_unmask_atoms(segment, atoms), escape)
        for segment in _strip_trailing_periods(
            cap_segments(pipeline, max_segments=max_segments, joiner="\n"),
        )
    ]
    return split_oversized_segments(restored)


def calculate_send_delay(
    segment: str,
    delay_base: float,
    delay_per_char: float,
    delay_max: float,
) -> float:
    """基础延迟 + 字数权重 + 0~DELAY_JITTER_SECONDS 抖动，夹在 [0, delay_max]。"""
    normalized_delay = delay_base + len(segment) * delay_per_char
    normalized_delay += random.uniform(0.0, DELAY_JITTER_SECONDS)
    return max(0.0, min(delay_max, normalized_delay))
