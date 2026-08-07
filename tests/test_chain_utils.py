from __future__ import annotations

from types import SimpleNamespace

from astrbot.api.event import MessageChain
from astrbot.api.message_components import At, Plain

from astrbot_plugin_smart_segmentation.chain_utils import (
    _is_plain_chain,
    extract_plain_text_chain,
    is_model_text_result,
    replace_plain_text_chain,
    should_segment_text,
)
from astrbot_plugin_smart_segmentation.settings import SegmentationSettings


def test_plain_chain_pure() -> None:
    assert _is_plain_chain(MessageChain([Plain("a"), Plain("b")])) is True


def test_plain_chain_with_at() -> None:
    assert _is_plain_chain(MessageChain([At(qq="123"), Plain("x")])) is True


def test_plain_chain_with_foreign_component() -> None:
    assert _is_plain_chain(MessageChain([Plain("x"), SimpleNamespace()])) is False  # type: ignore[arg-type]


def test_extract_joins_and_strips_thinking() -> None:
    chain = MessageChain([Plain("<think>t</think>你好"), Plain("世界")])
    assert extract_plain_text_chain(chain) == "你好世界"


def test_replace_keeps_at_and_replaces_first_plain() -> None:
    chain = MessageChain([At(qq="123"), Plain("旧"), Plain("旧2")])
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
    ok = SimpleNamespace(is_model_result=lambda: True, chain=[Plain("x")])
    assert is_model_text_result(ok) is True  # type: ignore[arg-type]
    no = SimpleNamespace(is_model_result=lambda: False, chain=[Plain("x")])
    assert is_model_text_result(no) is False  # type: ignore[arg-type]
    no_callable = SimpleNamespace(chain=[Plain("x")])
    assert is_model_text_result(no_callable) is True  # type: ignore[arg-type]


def test_should_segment_boundaries() -> None:
    settings = SegmentationSettings(min_length=12)
    assert should_segment_text("x" * 11, settings) is False
    assert should_segment_text("x" * 12, settings) is True
    assert should_segment_text("（只做动作）", settings) is False
