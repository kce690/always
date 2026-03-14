"""Tests for weak-input detection and short reply shaping."""

from nanobot.agent.loop import AgentLoop


def test_detects_qq_face_payload_as_weak_input() -> None:
    assert AgentLoop._is_weak_input('<faceType=6,faceId="0",ext="xxx">')


def test_detects_empty_and_symbol_only_as_weak_input() -> None:
    assert AgentLoop._is_weak_input("   ")
    assert AgentLoop._is_weak_input("。。。")


def test_detects_single_char_as_weak_input() -> None:
    assert AgentLoop._is_weak_input("啊")


def test_non_weak_input_keeps_normal_path() -> None:
    assert not AgentLoop._is_weak_input("你在干什么")
    assert not AgentLoop._is_weak_input("我今天好累")


def test_weak_input_reply_is_short_and_no_terminal_punct() -> None:
    out = AgentLoop._shape_weak_input_reply('<faceType=6,faceId="0",ext="xxx">')
    assert out in {"咋啦", "干嘛", "在呢", "怎么了", "嗯"}
    assert out[-1] not in "。！？!?"
