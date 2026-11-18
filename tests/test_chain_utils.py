from __future__ import annotations

from types import SimpleNamespace

from astrbot.api.event import MessageChain, MessageEventResult, ResultContentType
from astrbot.api.message_components import At, Plain
from astrbot_plugin_smart_segmentation.chain_utils import (
    extract_plain_text_chain,
    is_model_text_result,
    is_plain_chain,
    replace_plain_text_chain,
    should_segment_text,
)
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings


def test_plain_chain_with_at() -> None:
    assert is_plain_chain(MessageChain([At(qq="123"), Plain("x")])) is True
    assert is_plain_chain(MessageChain([Plain("x")])) is True
    assert is_plain_chain(MessageChain([At(qq="1"), At(qq="2"), Plain("x")])) is True


def test_plain_chain_accepts_multiple_plains() -> None:
    """多段 Plain（gemini 多 part、agent runner 直接交链）仍要能分段：全文 join 后
    首段进第一条 Plain，其余走补发。"""
    chain = MessageChain([At(qq="123"), Plain("前"), Plain("后")])
    assert extract_plain_text_chain(chain) == "前后"
    assert replace_plain_text_chain(chain, "首段") is True
    assert [type(item) for item in chain.chain] == [At, Plain]
    assert chain.chain[-1].text == "首段"


def test_plain_chain_rejects_displaceable_at() -> None:
    """At 出现在链中间或链尾时不分段：分段会把正文写进第一个 Plain、丢掉其余
    Plain，At 就落到首段末尾且后面没有正文（读 at.name 的端渲染成裸 @）。"""
    assert is_plain_chain(MessageChain([Plain("x"), At(qq="123")])) is False
    assert (
        is_plain_chain(MessageChain([Plain("前"), At(qq="123"), Plain("后")])) is False
    )
    chain = MessageChain([Plain("前"), At(qq="123"), Plain("后")])
    assert replace_plain_text_chain(chain, "新") is False
    assert [type(item) for item in chain.chain] == [Plain, At, Plain]


def test_plain_chain_with_foreign_component() -> None:
    assert is_plain_chain(MessageChain([Plain("x"), SimpleNamespace()])) is False  # type: ignore[arg-type]


def test_is_model_text_result_rejects_displaceable_at() -> None:
    """钩子入口的第一道闸就要拒绝：否则会先替换链、后插的 At 落到首条消息末尾。"""
    for component_chain in (
        [Plain("x"), At(qq="1")],
        [Plain("前"), At(qq="1"), Plain("后")],
        [At(qq="1")],
    ):
        result = MessageEventResult(
            chain=component_chain,
            result_content_type=ResultContentType.LLM_RESULT,
        )
        assert is_model_text_result(result) is False


def test_extract_strips_thinking() -> None:
    chain = MessageChain([Plain("<think>t</think>你好"), Plain("世界")])
    assert extract_plain_text_chain(chain) == "你好世界"


def test_replace_keeps_at_and_replaces_plain() -> None:
    chain = MessageChain([At(qq="123"), Plain("旧")])
    assert replace_plain_text_chain(chain, "新") is True
    assert isinstance(chain.chain[0], At)
    assert chain.chain[1].text == "新"
    assert len(chain.chain) == 2


def test_replace_rejects_chain_without_plain() -> None:
    chain = MessageChain([At(qq="123")])
    assert replace_plain_text_chain(chain, "新") is False


def test_replace_rejects_non_plain_chain() -> None:
    chain = MessageChain([Plain("x"), SimpleNamespace()])  # type: ignore[arg-type]
    assert replace_plain_text_chain(chain, "新") is False


def test_is_model_text_result() -> None:
    model = MessageEventResult(
        chain=[Plain("模型回复")],
        result_content_type=ResultContentType.LLM_RESULT,
    )
    runner_error = SimpleNamespace(
        chain=[Plain("agent runner error")],
        is_llm_result=lambda: False,
    )
    assert is_model_text_result(model) is True
    assert is_model_text_result(runner_error) is False


def test_should_segment_boundaries() -> None:
    settings = SegmentationSettings(min_length=12)
    assert should_segment_text("x" * 11, settings) is False
    assert should_segment_text("x" * 12, settings) is True
    # 动作-only 输入必须长于 min_length，否则被长度闸门挡住，测不到动作分支
    assert should_segment_text("（只做动作只做动作只做动作）", settings) is False
