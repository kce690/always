"""Tests for internal fallback suppression and social short-reply guards."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


def _make_loop(tmp_path: Path, llm_response: LLMResponse) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.chat_with_retry = AsyncMock(return_value=llm_response)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=65_536,
    )
    loop.provider.chat_with_retry = provider.chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


@pytest.mark.asyncio
async def test_no_response_is_suppressed_from_outbound_and_history(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content=None, tool_calls=[]))
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你好")

    out = await loop._process_message(msg)

    assert out is None
    session = loop.sessions.get_or_create("qq:c1")
    assistant_msgs = [m for m in session.messages if m.get("role") == "assistant"]
    assert assistant_msgs == []


@pytest.mark.asyncio
async def test_internal_fallback_text_not_persisted(tmp_path: Path) -> None:
    loop = _make_loop(
        tmp_path,
        LLMResponse(content="I've completed processing but have no response to give.", tool_calls=[]),
    )
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="测试一下")

    out = await loop._process_message(msg)

    assert out is None
    session = loop.sessions.get_or_create("qq:c1")
    assert all(
        "i've completed processing but have no response to give"
        not in str(m.get("content", "")).lower()
        for m in session.messages
    )


@pytest.mark.asyncio
async def test_user_mentions_internal_fallback_text_is_treated_as_user_text(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="知道 这是占位句", tool_calls=[]))
    msg = InboundMessage(
        channel="qq",
        sender_id="u1",
        chat_id="c1",
        content="I've completed processing but have no response to give 这句是什么意思",
    )

    out = await loop._process_message(msg)

    assert out is not None
    assert "占位句" in out.content


@pytest.mark.asyncio
async def test_social_ping_gets_short_reply_without_llm_roundtrip(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="这条不该被用到", tool_calls=[]))
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="想我了吗")

    out = await loop._process_message(msg)

    assert out is not None
    assert out.content in {"想呀", "在呢", "歇着呢"}
    assert out.content[-1] not in "。！？!?"
    assert loop.provider.chat_with_retry.await_count == 0


def test_save_turn_filters_tool_and_internal_fallback_entries(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    session = Session(key="qq:c1")

    messages = [
        {"role": "assistant", "content": "I've completed processing but have no response to give."},
        {"role": "tool", "name": "web_search", "content": "debug trace"},
        {"role": "user", "content": "普通输入"},
    ]
    loop._save_turn(session, messages, 0)

    assert [m["role"] for m in session.messages] == ["user"]


def test_memory_formatter_skips_internal_and_tool_text(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    formatted = store._format_messages(
        [
            {"role": "tool", "content": "debug trace", "timestamp": "2026-03-14T14:00:00"},
            {
                "role": "assistant",
                "content": "I've completed processing but have no response to give.",
                "timestamp": "2026-03-14T14:00:01",
            },
            {"role": "user", "content": "想我了吗", "timestamp": "2026-03-14T14:00:02"},
        ]
    )

    assert "debug trace" not in formatted
    assert "completed processing" not in formatted.lower()
    assert "想我了吗" in formatted
