"""Tests for knowledge-probe short-reply shaping."""

from nanobot.agent.loop import AgentLoop


def test_knowledge_probe_defaults_to_short_ack() -> None:
    user = "你知道行列式的定义吗"
    reply = "知道。行列式是一个从方阵到标量的映射，满足多线性和交替性。"

    out = AgentLoop._shape_knowledge_probe_reply(user, reply)

    assert out == "知道啊。"


def test_knowledge_probe_for_hui_defaults_to_hui_yi_dian() -> None:
    user = "这个你会吗"
    reply = "会，我给你详细讲一下。"

    out = AgentLoop._shape_knowledge_probe_reply(user, reply)

    assert out == "会一点。"


def test_explicit_explain_request_not_forced_short() -> None:
    user = "讲讲三极管"
    reply = "三极管分为NPN和PNP，核心是电流放大。"

    out = AgentLoop._shape_knowledge_probe_reply(user, reply)

    assert out == reply
