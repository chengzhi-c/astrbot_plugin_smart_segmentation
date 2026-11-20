from __future__ import annotations

import asyncio
import json
import time
from functools import wraps
from types import SimpleNamespace

import pytest
from astrbot.api.event import (
    AstrMessageEvent,
    MessageChain,
    MessageEventResult,
    ResultContentType,
)
from astrbot.api.message_components import Plain
from astrbot.builtin_stars.builtin_commands.commands.conversation import (
    ConversationCommands,
)
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.platform.astrbot_message import MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.platform.sources.webchat.webchat_event import (
    WebChatMessageEvent,
    webchat_queue_mgr,
)
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot_plugin_smart_segmentation.main import SmartSegmentationPlugin
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings
from astrbot_plugin_smart_segmentation.streaming.wrapper import stream_realtime_segments

pytestmark = pytest.mark.asyncio
name = "astrbot_plugin_smart_segmentation"


@pytest.fixture(autouse=True)
def restore_streaming_patch(monkeypatch):
    original_process = ResultDecorateStage.process
    patch_attributes = {
        key: value
        for key, value in vars(ResultDecorateStage).items()
        if key.startswith("_smart_segmentation_")
    }

    yield
    monkeypatch.undo()
    ResultDecorateStage.process = original_process
    for key in list(vars(ResultDecorateStage)):
        if key.startswith("_smart_segmentation_"):
            delattr(ResultDecorateStage, key)
    for key, value in patch_attributes.items():
        setattr(ResultDecorateStage, key, value)


class ContextStub:
    def __init__(self, output=None):
        self.output = output
        self.sent = []

    def get_config(self, umo=None):
        return {"agent_runner": {"runner_type": "astrbot"}}

    async def get_current_chat_provider_id(self, umo=None):
        return "test-provider" if self.output else ""

    async def llm_generate(self, **_):
        return SimpleNamespace(completion_text=json.dumps(self.output))

    async def send_message(self, session, chain):
        self.sent.append(chain.get_plain_text())
        return True


class MemoryEvent(AstrMessageEvent):
    async def send(self, message):
        self.sent.append(message.get_plain_text())


def event_with(text=None, event_type=MemoryEvent):
    message = SimpleNamespace(
        type=MessageType.FRIEND_MESSAGE,
        self_id="bot",
        sender=MessageMember("1", "User"),
        message=[],
        message_id="review-msg",
        message_str="request",
        group=None,
    )
    event = event_type(
        "request",
        message,
        PlatformMetadata("webchat", "review", "webchat"),
        "review-session",
    )
    event.sent = []
    if text is not None:
        event.set_result(
            MessageEventResult(
                [Plain(text)], result_content_type=ResultContentType.LLM_RESULT
            )
        )
    return event


def plugin_with(context, **config):
    return SmartSegmentationPlugin(context, {"min_length": 1, "delay_max": 0, **config})


def respond_stage():
    stage = RespondStage()
    stage.platform_settings = {}
    stage.enable_seg = False
    stage.config = {}
    return stage


def bind_after(plugin, monkeypatch):
    module_name = type(plugin).__module__
    monkeypatch.setitem(
        star_map,
        module_name,
        SimpleNamespace(activated=True, name=name, reserved=False),
    )
    (handler,) = star_handlers_registry.get_handlers_by_event_type(
        EventType.OnAfterMessageSentEvent
    )
    monkeypatch.setattr(handler, "handler", plugin.after_message_sent)


async def drain(plugin):
    await asyncio.gather(*list(plugin._follow_ups._tasks))


async def source(*chunks):
    for chunk in chunks:
        yield MessageChain([Plain(chunk)])


async def test_host_dedupe_collides_with_rewritten_first_segment(monkeypatch):
    context = ContextStub()
    plugin = plugin_with(context)
    bind_after(plugin, monkeypatch)
    first = "第一部分"
    second = "还要记得带钥匙"
    event = event_with(f"{first}。{second}。")
    event.set_extra("_send_message_to_user_current_session_plain_texts", [first])
    try:
        await plugin.on_decorating_result(event)
        await respond_stage().process(event)
        await drain(plugin)
        assert any(second in reply for reply in event.sent + context.sent)
    finally:
        await plugin.terminate()


async def test_first_send_still_running_at_pending_ttl(monkeypatch):
    context = ContextStub()
    plugin = plugin_with(context)
    bind_after(plugin, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowEvent(MemoryEvent):
        async def send(self, message):
            entered.set()
            await release.wait()
            await super().send(message)

    event = event_with("第一部分。第二部分。", SlowEvent)
    try:
        await plugin.on_decorating_result(event)
        job = asyncio.create_task(respond_stage().process(event))
        await asyncio.wait_for(entered.wait(), timeout=2)
        pending_id = plugin._follow_ups.pending_id_of(event)
        pending = plugin._follow_ups._pending[pending_id]
        pending.expires_at = time.monotonic() - 1
        plugin._follow_ups.reap_expired()
        release.set()
        await job
        await drain(plugin)
        assert context.sent == ["第二部分"]
    finally:
        release.set()
        await plugin.terminate()


async def test_first_send_can_finish_after_pre_send_processing_exceeds_ttl(monkeypatch):
    context = ContextStub()
    plugin = plugin_with(context)
    bind_after(plugin, monkeypatch)
    event = event_with("第一部分。第二部分。")
    try:
        await plugin.on_decorating_result(event)
        pending_id = plugin._follow_ups.pending_id_of(event)
        plugin._follow_ups._pending[pending_id].expires_at = time.monotonic() - 1
        plugin._follow_ups.reap_expired()
        await respond_stage().process(event)
        await drain(plugin)
        assert event.sent == ["第一部分"]
        assert context.sent == ["第二部分"]
    finally:
        await plugin.terminate()


async def test_host_stop_after_foreground_releases_event(monkeypatch):
    context = ContextStub()
    plugin = plugin_with(context, delay_base=0.05, delay_max=0.05)
    bind_after(plugin, monkeypatch)
    event = event_with("第一部分。第二部分。")
    command = event_with()
    active_event_registry.register(event)
    try:
        await plugin.on_decorating_result(event)
        await respond_stage().process(event)
        active_event_registry.unregister(event)
        await asyncio.sleep(0)
        tasks = list(plugin._follow_ups._tasks)
        await ConversationCommands(context).stop(command)
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(task.cancelled() for task in tasks)
        assert context.sent == []
    finally:
        await plugin.terminate()


async def test_late_first_send_does_not_restore_records_after_terminate():
    plugin = plugin_with(ContextStub())
    event = event_with("第一部分。第二部分。")
    await plugin.on_decorating_result(event)
    await plugin.terminate()
    await event.send(event.get_result())
    assert plugin._follow_ups._pending == {}


async def test_stream_patch_honors_session_plugin_list():
    context = ContextStub()
    plugin = plugin_with(context, streaming_compat_enabled=True, streaming_min_chars=1)
    event = event_with()
    event.plugins_name = []
    original_stream = source("First! Second!")
    event.set_result(
        MessageEventResult(
            result_content_type=ResultContentType.STREAMING_RESULT,
            async_stream=original_stream,
        )
    )
    try:
        async for _ in ResultDecorateStage().process(event):
            pass
        output = [
            item.get_plain_text() async for item in event.get_result().async_stream
        ]
        assert output == ["First! Second!"]
    finally:
        await plugin.terminate()


async def test_wrapped_patch_is_not_owned_by_plugin(monkeypatch):
    stage_type = ResultDecorateStage
    host_process = stage_type.process
    plugin = plugin_with(ContextStub(), streaming_compat_enabled=True)
    plugin_process = stage_type.process

    @wraps(plugin_process)
    async def foreign(*args, **kwargs):
        async for item in plugin_process(*args, **kwargs):
            yield item
        yield "foreign"

    monkeypatch.setattr(stage_type, "process", foreign)
    await plugin.terminate()
    current = stage_type.process
    stage_type.process = host_process
    assert current is foreign


async def test_validated_model_segmentation_keeps_url_whole(monkeypatch):
    url = "https://example.com/guide"
    text = f"访问 {url}。请确认。"
    context = ContextStub(["访问 https://example.com/", "guide。请确认。"])
    plugin = plugin_with(context)
    bind_after(plugin, monkeypatch)
    event = event_with(text)
    try:
        await plugin.on_decorating_result(event)
        await respond_stage().process(event)
        await drain(plugin)
        output = event.sent + context.sent
        assert any(url in reply for reply in output)
    finally:
        await plugin.terminate()


async def test_webchat_hard_length_cut_keeps_word_separator(monkeypatch):
    async def put(_request, payload):
        payloads.append(payload)
        return True

    monkeypatch.setattr(webchat_queue_mgr, "put_back_queue", put)
    payloads = []
    event = event_with(event_type=WebChatMessageEvent)
    text = "one two three four five six seven eight"
    settings = SegmentationSettings(streaming_min_chars=10, streaming_max_chars=20)
    await event.send_streaming(stream_realtime_segments(source(text), settings))
    complete = [
        payload["data"] for payload in payloads if payload["type"] == "complete"
    ]
    assert complete == [text]


async def test_long_split_does_not_send_format_only_first_chunk(monkeypatch):
    context = ContextStub()
    plugin = plugin_with(context)
    bind_after(plugin, monkeypatch)
    event = event_with("\u200b" * 1500 + "可见正文")
    try:
        await plugin.on_decorating_result(event)
        await respond_stage().process(event)
        await drain(plugin)
        got_visible = [
            bool(reply.strip("\u200b")) for reply in event.sent + context.sent
        ]
        assert all(got_visible)
    finally:
        await plugin.terminate()


async def test_model_cannot_erase_time_separator(monkeypatch):
    text = "请你下午12:30到办公室开会。"
    context = ContextStub(["请你下午12", "30到办公室开会。"])
    plugin = plugin_with(context, tail_clean_enabled=True)
    bind_after(plugin, monkeypatch)
    event = event_with(text)
    try:
        await plugin.on_decorating_result(event)
        await respond_stage().process(event)
        await drain(plugin)
        output = event.sent + context.sent
        assert any("12:30" in reply for reply in output)
    finally:
        await plugin.terminate()


async def test_stream_cleaning_does_not_treat_word_after_whitespace_as_url():
    text = "https://example.com/a separate,"
    settings = SegmentationSettings(
        tail_clean_enabled=True, streaming_min_chars=3, streaming_max_chars=21
    )
    output = [
        item.get_plain_text()
        async for item in stream_realtime_segments(source(text), settings)
    ]
    assert "".join(output) == "https://example.com/a separate"


async def test_previously_blocked_old_patch_does_not_take_back_superseded_slot(
    monkeypatch,
):
    stage_type = ResultDecorateStage
    host_process = stage_type.process
    old = plugin_with(ContextStub(), streaming_compat_enabled=True)
    fresh = None
    try:

        async def foreign(*args):
            if False:
                yield

        monkeypatch.setattr(stage_type, "process", foreign)
        old._streaming.sync(old._get_settings())
        stage_type.process = host_process
        fresh = plugin_with(ContextStub(), streaming_compat_enabled=True)
        await fresh.terminate()
        assert stage_type.process is host_process
        old._streaming.sync(old._get_settings())
        assert stage_type.process is host_process
    finally:
        if fresh is not None and not fresh._stopping:
            await fresh.terminate()
        await old.terminate()
        stage_type.process = host_process
