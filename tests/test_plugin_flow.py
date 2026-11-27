from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from astrbot.api.event import MessageChain, MessageEventResult, ResultContentType
from astrbot.api.message_components import At, ComponentType, Image, Node, Plain, Record
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin


class FakeResult(MessageEventResult):
    def __init__(self, text: str) -> None:
        super().__init__(
            chain=[Plain(text)],
            result_content_type=ResultContentType.LLM_RESULT,
        )


class FakeEvent:
    def __init__(self, result: FakeResult) -> None:
        self._result = result
        self.extra: dict = {}
        self.unified_msg_origin = "sess:1"
        self.sent = []

    def get_result(self):
        return self._result

    def get_extra(self, key: str, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value

    def get_platform_name(self) -> str:
        return "aiocqhttp"

    async def send(self, message):
        self.sent.append(message)


class FakeContext:
    # 桩输出默认与下文文本一致；分段模型路径若失效（校验拒绝/调用失败）会退回本地
    # 切分，而这些文本没有本地边界，因此断言能区分「模型分段生效」与「全程兜底」。
    def __init__(self, completion: str = '["今天天气不错", "我们出去玩吧"]') -> None:
        self.sent: list[tuple[str, str]] = []
        self.generate_calls: list[dict] = []
        self.completion = completion

    def get_config(self, umo=None):
        return {}

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        return "p1"

    async def llm_generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(completion_text=self.completion)

    async def send_message(self, session, chain):
        text = "".join(getattr(c, "text", "") for c in chain.chain)
        self.sent.append((session, text))
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("use_model", [False, True])
@pytest.mark.parametrize(
    "config,expected",
    [
        ({"head_clean_enabled": False}, ["，开头，句中", "。第二句", "；·最后一句！"]),
        ({}, ["开头，句中", "第二句", "·最后一句！"]),
        (
            {"head_clean_marks": ["，", "；", "·"]},
            ["开头，句中", "。第二句", "最后一句！"],
        ),
        ({"head_clean_marks": ["，"]}, ["开头，句中", "。第二句", "；·最后一句！"]),
        ({"head_clean_marks": []}, ["，开头，句中", "。第二句", "；·最后一句！"]),
    ],
)
async def test_head_clean_reaches_first_and_followups(
    config, expected, use_model: bool, caplog
) -> None:
    segments = ["，开头，句中", "。第二句", "；·最后一句！"]
    ctx = FakeContext(json.dumps(segments) if use_model else "invalid-json")
    plugin = SmartSegmentationPlugin(
        ctx,
        {"min_length": 2, "delay_max": 0, "head_clean_enabled": True, **config},
    )
    event = FakeEvent(FakeResult("\n".join(segments)))
    try:
        await plugin.on_decorating_result(event)
        await event.send(event.get_result())
        await plugin.after_message_sent(event)
        await asyncio.gather(*list(plugin._follow_ups._tasks))
        assert [
            event.sent[0].get_plain_text(),
            *[text for _, text in ctx.sent],
        ] == expected
        assert ("本次按标点拆分" not in caplog.text) is use_model
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", [["，", "；正文"], ["，", "；"]])
async def test_head_clean_drops_empty_segments(parts) -> None:
    plugin = SmartSegmentationPlugin(
        FakeContext(json.dumps(parts)), {"min_length": 0, "head_clean_enabled": True}
    )
    result = FakeResult("".join(parts))
    result.chain.insert(0, At(qq="123"))
    try:
        await plugin.on_decorating_result(FakeEvent(result))
        if "；正文" in parts:
            assert result.get_plain_text() == "正文"
            assert isinstance(result.chain[0], At)
        else:
            assert result.chain == []
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_model", [False, True])
@pytest.mark.parametrize(
    "config,expected",
    [
        ({"tail_clean_enabled": False}, ["开头，句中，，", "第二句", "末尾？！"]),
        ({}, ["开头，句中", "第二句", "末尾？！"]),
        ({"tail_clean_marks": ["？", "！"]}, ["开头，句中，，", "第二句。", "末尾"]),
        ({"tail_clean_marks": ["，"]}, ["开头，句中", "第二句。", "末尾？！"]),
        ({"tail_clean_marks": []}, ["开头，句中，，", "第二句。", "末尾？！"]),
    ],
)
async def test_tail_clean_reaches_first_and_followups(
    config, expected, use_model, caplog
):
    segments = ["开头，句中，，", "第二句。", "末尾？！"]
    ctx = FakeContext(json.dumps(segments) if use_model else "not-json")
    plugin = SmartSegmentationPlugin(
        ctx,
        {
            "min_length": 2,
            "delay_max": 0,
            "tail_clean_enabled": True,
            **config,
        },
    )
    event = FakeEvent(FakeResult("\n".join(segments)))
    try:
        await plugin.on_decorating_result(event)
        await event.send(event.get_result())
        await plugin.after_message_sent(event)
        await asyncio.gather(*list(plugin._follow_ups._tasks))
        assert [
            event.sent[0].get_plain_text(),
            *[text for _, text in ctx.sent],
        ] == expected
        assert ("本次按标点拆分" not in caplog.text) is use_model
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("segments", [["，", "正文，"], ["，", "；"]])
async def test_tail_clean_drops_empty_segments(segments: list[str]) -> None:
    plugin = SmartSegmentationPlugin(
        FakeContext(json.dumps(segments)),
        {"min_length": 0, "tail_clean_enabled": True},
    )
    result = FakeResult("".join(segments))
    result.chain.insert(0, At(qq="123"))
    event = FakeEvent(result)
    try:
        await plugin.on_decorating_result(event)
        if "正文，" in segments:
            assert result.get_plain_text() == "正文"
            assert isinstance(result.chain[0], At)
        else:
            assert result.chain == []
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "```text\n" + "x" * 2991 + ";\nend;\n```",
        "https://example.com/" + "x" * 2980 + ";tail;",
        "x" * 1498 + "3.14，",
        "x" * 1500 + "，正文",
    ],
    ids=["fenced-code", "url", "decimal", "forced-leading-comma"],
)
async def test_tail_clean_preserves_content_at_forced_message_limit(text: str) -> None:
    ctx = FakeContext("not-json")
    plugin = SmartSegmentationPlugin(
        ctx,
        {
            "min_length": 2,
            "delay_max": 0,
            "head_clean_enabled": True,
            "tail_clean_enabled": True,
        },
    )
    event = FakeEvent(FakeResult(text))
    try:
        await plugin.on_decorating_result(event)
        await event.send(event.get_result())
        await plugin.after_message_sent(event)
        await asyncio.gather(*list(plugin._follow_ups._tasks))
        parts = [event.sent[0].get_plain_text(), *[part for _, part in ctx.sent]]
        assert "".join(parts) == text.removesuffix("，")
        assert all(len(part) <= 1500 for part in parts)
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_wrapper", [False, True])
async def test_plugin_end_to_end_reuses_event_after_reload(
    monkeypatch: pytest.MonkeyPatch,
    legacy_wrapper: bool,
) -> None:
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )

    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(
        ctx,  # type: ignore[arg-type]
        {
            "enabled": True,
            "min_length": 2,
            "max_segments": 5,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    if legacy_wrapper:
        dispatcher = plugin._follow_ups
        original_send = event.send

        async def legacy_send(message):
            sent = await original_send(message)
            pending_id = dispatcher.pending_id_of(event)
            if (
                pending_id
                and dispatcher.is_first_segment_send(event, pending_id, message)
                and pending_id in dispatcher._pending
            ):
                if sent is False:
                    dispatcher._pending[pending_id].first_send_failed = True
                else:
                    dispatcher._pending[pending_id].first_send_delivered = True
            return sent

        event.send = legacy_send
        event.set_extra("smart_segmentation_first_send_tracked", True)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    # first segment replaced in place, follow-up registered
    assert [c.text for c in result.chain] == ["今天天气不错"]
    assert plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]
    send_wrapper = event.send

    # re-decorate same result is a no-op
    calls_before = len(ctx.generate_calls)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert len(ctx.generate_calls) == calls_before

    await event.send(result)
    await plugin.after_message_sent(event)  # type: ignore[arg-type]
    tasks = [t for t in list(plugin._follow_ups._tasks) if not t.done()]  # type: ignore[attr-defined]
    if tasks:
        await asyncio.gather(*tasks)
    assert ctx.sent == [("sess:1", "我们出去玩吧")]

    await plugin.terminate()

    ctx.sent.clear()
    event._result = FakeResult("今天天气不错我们出去玩吧")
    reloaded = SmartSegmentationPlugin(ctx, plugin.config)  # type: ignore[arg-type]
    try:
        await reloaded.on_decorating_result(event)  # type: ignore[arg-type]
        assert event.send is send_wrapper
        await event.send(event.get_result())
        await reloaded.after_message_sent(event)  # type: ignore[arg-type]
        await asyncio.gather(*list(reloaded._follow_ups._tasks))
        assert ctx.sent == [("sess:1", "我们出去玩吧")]
    finally:
        await reloaded.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversion", "send_result", "derive"),
    [
        ("plain", None, True),
        ("tts", None, True),
        ("tts", False, True),
        ("tts-dual", None, True),
        ("t2i", None, True),
        ("forward", None, True),
        ("tts-prefix", None, True),
        ("tts", None, False),
        ("tts-prefix", None, False),
        ("tts-legacy", None, False),
        ("tts-dual-legacy", None, False),
        ("tts-dual-legacy-plain-rejected", None, False),
        ("t2i", None, False),
        ("forward", None, False),
    ],
)
async def test_respond_stage_derived_chain_keeps_followups_after_conversion(
    conversion: str,
    send_result: bool | None,
    derive: bool,
) -> None:
    """宿主转换首段后仍须认领其发送；Record 会先从原链抽走。"""
    if derive and not hasattr(MessageEventResult, "derive"):
        pytest.skip("当前 AstrBot 版本没有 MessageChain.derive")
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    result.chain.insert(0, At(qq="123"))
    event = FakeEvent(result)

    async def send(message):
        event.sent.append(message)
        if conversion == "tts-dual-legacy-plain-rejected" and any(
            isinstance(component, Plain) for component in message.chain
        ):
            return False
        return send_result

    event.send = send
    try:
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        await event.send(MessageChain([result.chain[0]]))
        pending_id = plugin._follow_ups.pending_id_of(event)
        pending = plugin._follow_ups._pending[pending_id]
        assert not pending.first_send_delivered and not pending.first_send_failed

        first_plain = result.chain[-1]
        if conversion == "tts-prefix":
            first_plain.text = "回复前缀：" + first_plain.text
        if conversion.startswith("tts"):
            result.chain = [
                Record(
                    file="file:///speech.wav",
                    text=first_plain.text,
                )
            ]
            if "legacy" in conversion and hasattr(result.chain[0], "text"):
                delattr(result.chain[0], "text")
            if "dual" in conversion:
                result.chain.append(first_plain)
        elif conversion == "t2i":
            result.chain = [Image.fromURL("https://example.com/rendered.png")]
        elif conversion == "forward":
            result.chain = [Node(uin="bot", name="AstrBot", content=[first_plain])]

        await event.send(MessageChain([Plain("独立发送的其他内容")]))
        assert not pending.first_send_delivered and not pending.first_send_failed

        make_chain = result.derive if derive else MessageChain
        records = RespondStage()._extract_comp(result.chain, {ComponentType.Record})
        for record in records:
            await event.send(make_chain([record]))
        if result.chain:
            await event.send(make_chain(result.chain))
        await plugin.after_message_sent(event)  # type: ignore[arg-type]
        await asyncio.gather(*list(plugin._follow_ups._tasks))

        expected = [("sess:1", "我们出去玩吧")]
        if send_result is False:
            expected.insert(0, ("sess:1", "今天天气不错"))
        assert ctx.sent == expected
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_respond_stage_derived_chain_orphan_only_sends_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod

    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    await event.send(MessageChain(result.chain))
    plugin._follow_ups.reap_expired()  # type: ignore[attr-defined]
    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task

    assert ctx.sent == [("sess:1", "我们出去玩吧")]
    await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send_first", "send_result", "expected_follow_ups"),
    [
        (False, None, []),
        (True, None, ["我们出去玩吧"]),
        (True, False, ["今天天气不错\n我们出去玩吧"]),
    ],
    ids=["suppressed-before-send", "sent-without-after-hook", "rejected-before-send"],
)
async def test_expired_pending_only_sends_remainder_after_first_send(
    monkeypatch: pytest.MonkeyPatch,
    send_first: bool,
    send_result: bool | None,
    expected_follow_ups: list[str],
) -> None:
    """孤儿回收不能复活已抑制的回复，也不能重复发送已送达的首段。"""
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod

    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    if send_result is False:

        async def rejected_send(_message):
            return False

        event.send = rejected_send

    try:
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        if send_first:
            await event.send(result)

        plugin._follow_ups.reap_expired()  # type: ignore[attr-defined]
        await asyncio.gather(*list(plugin._follow_ups._tasks))  # type: ignore[attr-defined]

        assert [text for _session, text in ctx.sent] == expected_follow_ups
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_model", [True, False], ids=["model", "local"])
async def test_plugin_does_not_send_invisible_segments_with_mention(
    use_model: bool,
) -> None:
    parts = ["\u200b \u200b", "今天天气不错", "\u2060", "我们出去玩吧", "\ufeff"]
    ctx = FakeContext(json.dumps(parts) if use_model else "invalid JSON")
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {"enabled": True, "min_length": 2, "delay_max": 0},
    )
    result = FakeResult("\n".join(parts))
    mention = At(qq="123", name="\u2060")
    result.chain.insert(0, mention)
    event = FakeEvent(result)
    try:
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        assert result.get_plain_text() == "今天天气不错"
        assert result.chain[0] is mention

        await event.send(result)
        await plugin.after_message_sent(event)  # type: ignore[arg-type]
        await asyncio.gather(*list(plugin._follow_ups._tasks))
        assert ctx.sent == [("sess:1", "我们出去玩吧")]
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send_outcome", "expected_follow_ups"),
    [
        ("unknown", []),
        ("rejected", [("sess:1", "今天天气不错"), ("sess:1", "我们出去玩吧")]),
    ],
)
async def test_first_send_retries_only_when_rejected(
    monkeypatch: pytest.MonkeyPatch,
    send_outcome: str,
    expected_follow_ups: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(
        "astrbot_plugin_smart_segmentation.follow_up.calculate_send_delay",
        lambda *_a, **_k: 0.0,
    )

    class FailingEvent(FakeEvent):
        async def send(self, message):
            if send_outcome == "unknown":
                raise RuntimeError("platform send failed")
            return False

    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FailingEvent(result)
    try:
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        if send_outcome == "unknown":
            with pytest.raises(RuntimeError, match="platform send failed"):
                await event.send(result)
        else:
            assert await event.send(result) is False

        await plugin.after_message_sent(event)  # type: ignore[arg-type]
        await asyncio.gather(*list(plugin._follow_ups._tasks))  # type: ignore[attr-defined]
        assert ctx.sent == expected_follow_ups
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded", [False, True], ids=["idle", "follow-up-in-flight"])
async def test_plugin_clears_result_without_visible_body(guarded: bool) -> None:
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True})  # type: ignore[arg-type]
    result = FakeResult("\u200b \u2060\ufeff")
    result.chain.insert(0, At(qq="123", name="\u2060"))
    event = FakeEvent(result)
    try:
        guard = (
            plugin._follow_ups._guard_session(event.unified_msg_origin)
            if guarded
            else nullcontext()
        )
        with guard:
            await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        assert result.chain == []
        assert ctx.generate_calls == []
        assert plugin._follow_ups.pending_id_of(event) == ""
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_skips_segmentation_when_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """terminate 落在分段调用期间时不得替换结果链。

    否则链已被换成首段、补发又被 after_message_sent 丢弃，用户只看到首段。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)

    async def fake_segment(event, text, settings):
        plugin._stopping = True  # 模拟 terminate 在分段调用期间发生
        return ["你好", "世界", "继续说一点"]

    monkeypatch.setattr(plugin._segmenter, "segment", fake_segment)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert plugin._follow_ups.pending_id_of(event) == ""

    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_keeps_whole_reply_when_disabled_during_generation() -> None:
    ctx = FakeContext()
    config = {"enabled": True, "min_length": 2}
    plugin = SmartSegmentationPlugin(ctx, config)
    source = "今天天气不错我们出去玩吧"
    event = FakeEvent(FakeResult(source))
    generate = ctx.llm_generate

    async def disable_during_generate(**kwargs):
        config["enabled"] = False
        return await generate(**kwargs)

    ctx.llm_generate = disable_during_generate
    try:
        await plugin.on_decorating_result(event)
        assert event.get_result().get_plain_text() == source
        assert plugin._follow_ups.pending_id_of(event) == ""
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_skips_text_already_sent_by_send_message_to_user() -> None:
    """宿主 RespondStage 按这份 extra 抑制重复回复。

    换链后 result.get_plain_text() 只剩首段，宿主的比对永远不命中，同一条回复会被
    发两遍（agent 用 send_message_to_user 直发全文 + 最终回复又是全文）。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    event.set_extra(
        "_send_message_to_user_current_session_plain_texts",
        ["你好。世界。继续说一点"],
    )

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert ctx.generate_calls == []
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_split_plain_thinking_tag_is_removed_from_short_reply() -> None:
    plugin = SmartSegmentationPlugin(FakeContext(), {"min_length": 20})
    result = FakeResult("")
    result.chain = [Plain("<thi"), Plain("nk>private-thought</think>你好")]
    try:
        await plugin.on_decorating_result(FakeEvent(result))
        assert result.get_plain_text() == "你好"
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_normalizes_thinking_tags_when_only_body_was_sent() -> None:
    """已发的是剥离后的正文、结果链还带 think：宿主按原文比对抑制不了，
    至少不能把 think 原文发出去。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    body = "今天天气不错我们出去玩吧"
    result = FakeResult("<think>内部推理</think>" + body)
    event = FakeEvent(result)
    event.set_extra("_send_message_to_user_current_session_plain_texts", [body])

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert ctx.generate_calls == []
    assert [c.text for c in result.chain] == [body]
    await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded", [False, True], ids=["idle", "guarded"])
async def test_plugin_strips_thinking_from_short_reply(guarded: bool) -> None:
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 12})  # type: ignore[arg-type]
    result = FakeResult("<think>secret</think>Hi")
    event = FakeEvent(result)
    guard = (
        plugin._follow_ups._guard_session(event.unified_msg_origin)
        if guarded
        else nullcontext()
    )

    try:
        with guard:
            await plugin.on_decorating_result(event)  # type: ignore[arg-type]
        assert [component.text for component in result.chain] == ["Hi"]
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_but_still_strips_thinking_tags() -> None:
    """让位给宿主内置分段时也要摘掉 think 原文：宿主会把链直接切分发给用户。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    body = "今天天气不错我们出去玩吧"
    result = FakeResult("<think>内部推理</think>" + body)
    event = FakeEvent(result)
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == [body]
    assert ctx.generate_calls == []
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_without_touching_multi_plain_chain() -> None:
    """让位时只有带 think 的链才归一：多 Plain 且无 think 的链不能并成一段。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("今天天气不错")
    result.chain = [Plain("今天天气不错"), Plain("我们出去玩吧")]
    event = FakeEvent(result)
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["今天天气不错", "我们出去玩吧"]
    assert ctx.generate_calls == []
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_yields_to_host_builtin_segmented_reply() -> None:
    """宿主内置分段会在装饰钩子之后再切一次首段：同时开启时必须让位。

    否则段数膨胀、插件边界被销毁，且补发被宿主的逐条间隔拖后。
    """

    class SegmentedContext(FakeContext):
        def get_config(self, umo=None):
            return {"platform_settings": {"segmented_reply": {"enable": True}}}

    ctx = SegmentedContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    text = "你好。世界。继续说一点"
    result = FakeResult(text)
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == [text]
    assert ctx.generate_calls == []
    assert plugin._warned_builtin_segmentation is True  # type: ignore[attr-defined]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_builtin_segmentation_warning_logged_once(monkeypatch) -> None:
    """让位 warning 只记一次：每轮都刷会在长会话里把日志刷满。"""
    from astrbot_plugin_smart_segmentation import main as main_mod

    warnings: list[str] = []
    monkeypatch.setattr(
        main_mod.logger, "warning", lambda *args, **_k: warnings.append(str(args))
    )

    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    host_config = {"platform_settings": {"segmented_reply": {"enable": True}}}
    plugin.context.get_config = lambda _umo=None: host_config  # type: ignore[method-assign,union-attr]
    for text in ("今天天气不错我们出去玩吧", "再来一条同样被让位的回复"):
        await plugin.on_decorating_result(FakeEvent(FakeResult(text)))  # type: ignore[arg-type]
    assert len(warnings) == 1
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_keeps_segmenting_where_host_skips_builtin_segmentation() -> None:
    """宿主在 qq_official_webhook 等平台跳过内置分段：这里不该跟着让位。"""

    class SegmentedContext(FakeContext):
        def get_config(self, umo=None):
            return {"platform_settings": {"segmented_reply": {"enable": True}}}

    ctx = SegmentedContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    event.get_platform_name = lambda: "qq_official_webhook"  # type: ignore[attr-defined]

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["今天天气不错"]
    await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_during_provider_lookup", [False, True])
async def test_plugin_skips_model_call_when_stopping(
    stop_during_provider_lookup: bool,
) -> None:
    """停止中不要再打一次分段模型：既浪费额度，也会拖住卸载。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    if stop_during_provider_lookup:

        async def stop_during_lookup(umo):
            await plugin.terminate()
            return "p1"

        ctx.get_current_chat_provider_id = stop_during_lookup  # type: ignore[method-assign]
    else:
        plugin._stopping = True
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert ctx.generate_calls == []
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_disabled_passthrough() -> None:
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": False})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。"]
    assert ctx.generate_calls == []
    await plugin.terminate()


@pytest.mark.asyncio
async def test_sync_loop_installs_patch_without_message_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流式路径无 hook，补丁装卸必须能不依赖消息事件完成（D-1）。"""
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
    from astrbot_plugin_smart_segmentation import main as main_mod

    original = ResultDecorateStage.process
    monkeypatch.setattr(main_mod, "STREAMING_SYNC_INTERVAL_SECONDS", 0.01)

    config = {"enabled": True, "streaming_compat_enabled": False}
    plugin = main_mod.SmartSegmentationPlugin(FakeContext(), config)  # type: ignore[arg-type]
    try:
        assert plugin._streaming is None

        synced = asyncio.Event()
        sync = plugin._sync_streaming

        def wait_for_install(settings):
            sync(settings)
            if plugin._streaming is not None and plugin._streaming._installed:
                synced.set()

        plugin._sync_streaming = wait_for_install  # type: ignore[method-assign]
        config["streaming_compat_enabled"] = True
        await asyncio.wait_for(synced.wait(), timeout=1)

        assert plugin._streaming is not None
        assert plugin._streaming._desired_enabled is True
        # 只做决策不够：补丁必须真的装上，否则流式兼容不生效
        assert plugin._streaming._installed is True
        assert ResultDecorateStage.process is not original
    finally:
        await plugin.terminate()
        assert ResultDecorateStage.process is original


@pytest.mark.asyncio
async def test_sync_loop_reaps_expired_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """交接钩子一直不触发时，只有周期同步能兜底交付：循环里必须 reap。"""
    from astrbot_plugin_smart_segmentation import follow_up as follow_up_mod
    from astrbot_plugin_smart_segmentation import main as main_mod

    monkeypatch.setattr(main_mod, "STREAMING_SYNC_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(follow_up_mod, "PENDING_FOLLOW_UP_TTL_SECONDS", 0.0)

    class SignalContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.delivered = asyncio.Event()

        async def send_message(self, session, chain):
            sent = await super().send_message(session, chain)
            self.delivered.set()
            return sent

    ctx = SignalContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    settings = plugin._get_settings()  # type: ignore[attr-defined]
    assert settings is not None
    plugin._follow_ups.register(  # type: ignore[attr-defined]
        session="sess:1",
        segments=["一", "二", "三"],
        settings=settings,
    )
    next(iter(plugin._follow_ups._pending.values())).first_send_delivered = True  # type: ignore[attr-defined]

    await asyncio.wait_for(ctx.delivered.wait(), timeout=1)
    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ctx.sent == [("sess:1", "二\n三")]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_after_message_sent_drops_pending_when_stopping() -> None:
    """停止中必须丢弃 pending 并清簿记：否则补发会在插件关闭后飞出。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]

    plugin._stopping = True
    await plugin.after_message_sent(event)  # type: ignore[arg-type]
    # 让出事件循环：若这里走了 schedule，任务会真的发出去
    await asyncio.sleep(0)
    assert plugin._follow_ups._pending == {}  # type: ignore[attr-defined]
    assert plugin._follow_ups._tasks == set()  # type: ignore[attr-defined]
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    assert ctx.sent == []

    await plugin.terminate()


@pytest.mark.asyncio
async def test_terminate_unpatches_even_if_shutdown_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutdown 抛异常时补丁仍必须卸掉：否则它永久留在 ResultDecorateStage 上，
    插件实例也随闭包一起无法回收。"""
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {"enabled": True, "streaming_compat_enabled": True},
    )
    assert plugin._streaming is not None
    assert plugin._streaming._installed is True

    async def broken_shutdown() -> None:
        raise RuntimeError("shutdown exploded")

    monkeypatch.setattr(plugin._follow_ups, "shutdown", broken_shutdown)
    with pytest.raises(RuntimeError):
        await plugin.terminate()

    assert ResultDecorateStage.process is original


@pytest.mark.asyncio
async def test_guarded_decorate_clears_stale_bookkeeping() -> None:
    """会话守卫分支必须清掉旧簿记：否则上一轮残留的 pending 会被本轮
    after_message_sent 当成新登记的发出去（重复补发），或在 _pending 里滞留。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(result)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    stale_id = plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]
    assert stale_id
    await event.send(MessageChain(result.chain))
    assert stale_id in plugin._follow_ups._pending  # type: ignore[attr-defined]

    # 模拟：上一轮登记了 pending，但 RespondStage 被跳过；本轮同 event 被守卫挡住
    with plugin._follow_ups._guard_session(event.unified_msg_origin):
        await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    assert stale_id not in plugin._follow_ups._pending  # type: ignore[attr-defined]

    # 清簿记不能等于丢内容：剩余段要合并补发出去
    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ctx.sent == [("sess:1", "我们出去玩吧")]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_stale_pending_from_previous_round_is_delivered() -> None:
    """上一轮首段已发送但没有 after hook 时，下一轮先交付余段。"""
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(  # type: ignore[arg-type]
        ctx,
        {
            "enabled": True,
            "min_length": 2,
            "delay_base": 0,
            "delay_per_char": 0,
            "delay_max": 0,
        },
    )
    first = FakeResult("今天天气不错我们出去玩吧")
    event = FakeEvent(first)
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    stale_id = plugin._follow_ups.pending_id_of(event)  # type: ignore[attr-defined]
    assert stale_id
    await event.send(MessageChain(first.chain))

    # 宿主再次进入装饰钩子（同一 event，新 result），而上一轮的 after 钩子没触发
    event._result = FakeResult("新的回复。第二段内容")  # type: ignore[attr-defined]
    await plugin.on_decorating_result(event)  # type: ignore[arg-type]

    for task in list(plugin._follow_ups._tasks):  # type: ignore[attr-defined]
        await task
    assert ("sess:1", "我们出去玩吧") in ctx.sent
    assert plugin._follow_ups.pending_id_of(event) != stale_id  # type: ignore[attr-defined]

    await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_skips_round_after_streaming_finished() -> None:
    """流式回复发完后宿主对后续结果一律不发送。

    第三方插件把结果类型改回 LLM_RESULT 时，这里若照常分段并登记补发，剩余段会在
    兜底超时后被当成孤儿重发一遍（用户看到重复残段）。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("你好。世界。继续说一点")
    event = FakeEvent(result)
    event.set_extra("_streaming_finished", True)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]
    assert [c.text for c in result.chain] == ["你好。世界。继续说一点"]
    assert ctx.generate_calls == []
    assert plugin._follow_ups.pending_id_of(event) == ""  # type: ignore[attr-defined]
    await plugin.terminate()


@pytest.mark.asyncio
async def test_sync_streaming_does_not_install_after_terminate() -> None:
    """terminate 之后在途 hook 不得装上补丁：同步任务已取消，没人再卸它。

    关键场景是加载时流式兼容没开（`_streaming is None`）——补丁管理器自己的
    `_stopping` 保护不到它，只有 main 的停止检查能拦住。
    """
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage

    original = ResultDecorateStage.process
    config = {"enabled": True, "streaming_compat_enabled": False}
    plugin = SmartSegmentationPlugin(FakeContext(), config)  # type: ignore[arg-type]
    assert plugin._streaming is None
    await plugin.terminate()

    config["streaming_compat_enabled"] = True
    plugin._sync_streaming(plugin._get_settings())  # 在途 hook 的迟到调用
    assert plugin._streaming is None
    assert ResultDecorateStage.process is original


@pytest.mark.asyncio
async def test_sync_loop_stops_on_terminate() -> None:
    """terminate 必须回收周期同步任务，不留悬挂协程。"""
    from astrbot_plugin_smart_segmentation import main as main_mod

    plugin = main_mod.SmartSegmentationPlugin(FakeContext(), {"enabled": True})  # type: ignore[arg-type]
    task = plugin._sync_task
    assert task is not None
    await plugin.terminate()
    assert task.done()


@pytest.mark.asyncio
async def test_image_reply_is_left_to_the_host_instead_of_being_deleted() -> None:
    """带图片的回复必须原样留给宿主，不能被当成空回复吞掉。

    非纯文本链在 is_model_text_result 就被挡回（链不动、直接返回），这里钉住的是
    这个顺序：任何把「取不出正文」与「内容是空的」并成一个出口的改动，都会让
    模型每一条「文字 + 图片」回复在群里消失。
    """
    ctx = FakeContext()
    plugin = SmartSegmentationPlugin(ctx, {"enabled": True, "min_length": 2})  # type: ignore[arg-type]
    result = FakeResult("配图说明")
    result.chain = [Plain("配图说明"), Image(file="pic.jpg")]
    event = FakeEvent(result)

    await plugin.on_decorating_result(event)  # type: ignore[arg-type]

    assert result.chain, "含图片的回复被整条删掉了"
    await plugin.terminate()
