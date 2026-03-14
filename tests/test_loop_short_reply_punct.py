"""Tests for short-reply punctuation stripping."""

from nanobot.agent.loop import AgentLoop


def test_forced_status_two_part_reply_drops_sentence_punct() -> None:
    out = AgentLoop._strip_short_reply_terminal_punct(
        "你在干什么",
        "在家躺着。你呢。",
    )
    assert out == "在家躺着 你呢"


def test_forced_knowledge_reply_drops_terminal_punct() -> None:
    out = AgentLoop._strip_short_reply_terminal_punct(
        "你知道行列式吗",
        "知道啊。",
    )
    assert out == "知道啊"


def test_weak_input_short_reply_drops_terminal_punct() -> None:
    out = AgentLoop._strip_short_reply_terminal_punct(
        "<faceType=6,faceId=\"0\">",
        "在呢。",
    )
    assert out == "在呢"


def test_non_forced_long_reply_keeps_punct() -> None:
    text = "这个问题我得先看下上下文再回答。"
    out = AgentLoop._strip_short_reply_terminal_punct("帮我分析一下", text)
    assert out == text
