"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _SHORT_REPLY_MAX_CHARS = 12
    _SHORT_REPLY_END_PUNCT = "。！？!?!.~～…"
    _WEAK_INPUT_FILLERS = (
        "嗯", "嗯嗯", "哦", "噢", "啊", "哈", "哈哈", "诶", "欸", "哎",
        "emm", "emmm", "hh", "hhh", "...", "。。。", "…", "?", "？", "!", "！", ".", "。",
    )
    _WEAK_INPUT_REPLIES = ("咋啦", "干嘛", "在呢", "怎么了", "嗯")
    _STATUS_QUERY_HINTS = (
        "你在干什么", "你现在在干嘛", "你现在在干什么", "干嘛呢",
        "在吗", "你在吗", "你现在在吗", "你现在在干嘛呢", "你是不是刚忙完", "你现在方便聊吗",
    )
    _KNOWLEDGE_PROBE_HINTS = (
        "你知道", "你懂", "你会", "这个你会吗", "这个你知道吗",
        "这个你懂吗", "什么意思吗", "这个什么意思", "这个你知道什么意思吗",
    )
    _EXPLAIN_REQUEST_HINTS = ("讲", "讲讲", "详细讲", "展开说", "解释一下")
    _SOCIAL_PING_HINTS = ("想我了吗", "在吗", "在不在", "干嘛呢", "干嘛", "咋不说话")
    _INTERNAL_FALLBACK_MARKERS = (
        "i've completed processing but have no response to give",
        "no response",
        "empty response",
        "internal fallback",
        "background task completed",
    )

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @classmethod
    def _is_status_query(cls, text: str) -> bool:
        msg = (text or "").strip()
        return any(hint in msg for hint in cls._STATUS_QUERY_HINTS)

    @classmethod
    def _is_explain_request(cls, text: str) -> bool:
        msg = (text or "").strip()
        return any(hint in msg for hint in cls._EXPLAIN_REQUEST_HINTS)

    @classmethod
    def _is_knowledge_probe(cls, text: str) -> bool:
        msg = (text or "").strip()
        if not msg:
            return False
        if cls._is_explain_request(msg):
            return False
        if "吗" not in msg and "？" not in msg and "?" not in msg:
            return False
        return any(hint in msg for hint in cls._KNOWLEDGE_PROBE_HINTS)

    @classmethod
    def _shape_knowledge_probe_reply(cls, user_text: str, reply: str | None) -> str | None:
        """For knowledge probes, default to a very short acknowledgement."""
        if not reply or not cls._is_knowledge_probe(user_text):
            return reply

        msg = (user_text or "").strip()
        if "会" in msg or "懂" in msg:
            return "会一点"
        if "什么意思" in msg:
            return "知道"
        return "知道啊"

    @staticmethod
    def _strip_weak_input_markup(text: str) -> str:
        """Remove channel markup/tag payload from weak-input candidates."""
        clean = text or ""
        clean = re.sub(r"<[^>\n]{1,256}>", " ", clean)
        clean = re.sub(r"\[CQ:[^\]]+\]", " ", clean)
        clean = re.sub(
            r"\[(?:表情|图片|动画表情|qq表情|emoji|face|sticker|image)\]",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        clean = clean.replace("\u200b", " ")
        return re.sub(r"\s+", " ", clean).strip()

    @classmethod
    def _is_weak_input(cls, text: str) -> bool:
        """Detect emoji/placeholder/low-semantic input that should not trigger full generation."""
        raw = (text or "").strip()
        if not raw:
            return True
        if re.fullmatch(r"(?:<[^>\n]{1,256}>|\[CQ:[^\]]+\]|\s)+", raw):
            return True

        stripped = cls._strip_weak_input_markup(raw)
        if not stripped:
            return True

        flat = re.sub(r"\s+", "", stripped)
        flat_lower = flat.lower()
        if not flat:
            return True
        if flat in cls._WEAK_INPUT_FILLERS or flat_lower in cls._WEAK_INPUT_FILLERS:
            return True
        if flat_lower in {"emoji", "face", "sticker", "image"}:
            return True
        if re.fullmatch(r"[^\w\u4e00-\u9fff]+", flat, flags=re.UNICODE):
            return True
        if len(flat) == 1 and re.fullmatch(r"[A-Za-z0-9\u4e00-\u9fff]", flat):
            return True
        if len(flat) <= 2 and flat in {"哈", "啊", "哦", "嗯", "噢", "诶", "欸", "哎"}:
            return True
        return False

    @classmethod
    def _shape_weak_input_reply(cls, user_text: str) -> str | None:
        """Short-circuit weak input with very short casual replies."""
        if not cls._is_weak_input(user_text):
            return None

        raw = (user_text or "").strip()
        if not raw:
            return "在呢"

        stripped = cls._strip_weak_input_markup(raw)
        flat = re.sub(r"\s+", "", stripped)
        if flat in {"嗯", "嗯嗯", "哦", "噢", "啊", "哈", "哈哈", "诶", "欸", "哎"}:
            return "嗯"
        if re.search(r"[?？]", raw):
            return "怎么了"

        seed = sum(ord(ch) for ch in raw) % len(cls._WEAK_INPUT_REPLIES)
        return cls._WEAK_INPUT_REPLIES[seed]

    @classmethod
    def _is_social_ping(cls, text: str) -> bool:
        """Detect short social ping/chitchat turns that should get short replies."""
        msg = re.sub(r"[\s，,。！？!?~～]+", "", (text or ""))
        if not msg or len(msg) > 12:
            return False
        return any(hint in msg for hint in cls._SOCIAL_PING_HINTS)

    @classmethod
    def _shape_social_ping_reply(cls, user_text: str) -> str | None:
        """Return short casual reply for social pings."""
        if not cls._is_social_ping(user_text):
            return None

        msg = re.sub(r"[\s，,。！？!?~～]+", "", (user_text or ""))
        if "想我" in msg:
            return "想呀"
        if "干嘛" in msg:
            return "歇着呢"
        if "咋不说话" in msg:
            return "在呢"
        if "在" in msg:
            return "在呢"
        return "在呢"

    @classmethod
    def _is_internal_fallback_output(cls, text: str | None) -> bool:
        """Detect internal placeholders/fallbacks that must not be sent or persisted."""
        if text is None:
            return True
        msg = text.strip()
        if not msg:
            return True
        lowered = msg.lower()
        if any(marker in lowered for marker in cls._INTERNAL_FALLBACK_MARKERS):
            return True
        if lowered in {"n/a", "(empty)", "none"}:
            return True
        return False

    @classmethod
    def _strip_short_reply_terminal_punct(cls, user_text: str, reply: str | None) -> str | None:
        """For short casual replies, remove sentence-final punctuation."""
        if not reply:
            return reply
        text = reply.strip()
        if not text:
            return reply

        forced = (
            cls._is_status_query(user_text)
            or cls._is_knowledge_probe(user_text)
            or cls._is_weak_input(user_text)
            or cls._is_social_ping(user_text)
        )

        if forced and re.search(r"[。！？!?]", text):
            pieces = [p.strip() for p in re.split(r"[。！？!?]+", text) if p.strip()]
            if 1 <= len(pieces) <= 2 and all(len(re.sub(r"\s+", "", p)) <= 10 for p in pieces):
                return " ".join(pieces)

        core = text.rstrip(cls._SHORT_REPLY_END_PUNCT).strip()
        if core == text:
            return reply
        if not core or "\n" in core:
            return core or reply
        if re.search(r"[。！？!?]", core):
            return reply

        compact_len = len(re.sub(r"\s+", "", core))
        if forced or compact_len <= cls._SHORT_REPLY_MAX_CHARS:
            return core
        return reply

    @staticmethod
    def _shape_status_reply(
        user_text: str,
        reply: str | None,
        *,
        has_recent_event: bool,
    ) -> str | None:
        """Constrain casual self-status replies to short, spoken, non-report style."""
        if not reply or not AgentLoop._is_status_query(user_text):
            return reply

        text = re.sub(r"\s+", " ", reply).strip()
        if not text:
            return reply

        parts = [p.strip(" ，,。！？!?") for p in re.split(r"[。！？!?]", text) if p.strip()]
        if not parts:
            return reply

        first = parts[0]
        if not has_recent_event:
            first = re.sub(r"(，|,)?刚[^，。！？!?]{0,24}", "", first).strip(" ，,")
            first = re.sub(r"(，|,)?刚刚[^，。！？!?]{0,24}", "", first).strip(" ，,")
            first = re.sub(r"(，|,)?刚才[^，。！？!?]{0,24}", "", first).strip(" ，,")

        first = re.sub(r"^我(现在|正在)", "", first).strip()
        first = re.sub(r"(在这里)?等着?你", "", first).strip(" ，,")
        first = re.sub(r"待命", "", first).strip(" ，,")
        if len(first) > 18:
            first = first[:18].rstrip(" ，,")
        if not first:
            first = "在呢"

        second = ""
        if len(parts) > 1:
            candidate = parts[1]
            if not has_recent_event:
                candidate = re.sub(r"(，|,)?刚[^，。！？!?]{0,24}", "", candidate).strip(" ，,")
            if not re.search(r"(安排|帮助|帮忙|请问|需要|服务|计划|问题)", candidate):
                if len(candidate) <= 8:
                    second = candidate
                else:
                    m = re.search(r"(你呢|咋啦|怎么了|怎么啦|干嘛|在吗)", candidate)
                    if m:
                        second = m.group(1)

        return f"{first}。{second + '。' if second else ''}"

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _persist_short_reply(
        self,
        session: Session,
        *,
        user_text: str,
        reply_text: str,
    ) -> None:
        """Persist a direct short-reply turn without calling the LLM loop."""
        from datetime import datetime

        session.messages.append(
            {
                "role": "user",
                "content": user_text,
                "timestamp": datetime.now().isoformat(),
            }
        )
        session.messages.append(
            {
                "role": "assistant",
                "content": reply_text,
                "timestamp": datetime.now().isoformat(),
            }
        )
        session.updated_at = datetime.now()
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0, include_assistant_text=False)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            if self._is_internal_fallback_output(final_content):
                return None
            return OutboundMessage(channel=channel, chat_id=chat_id, content=final_content)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        weak_reply = self._shape_weak_input_reply(msg.content)
        if weak_reply:
            await self._persist_short_reply(session, user_text=msg.content, reply_text=weak_reply)
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, weak_reply)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=weak_reply,
                metadata=msg.metadata or {},
            )

        social_reply = self._shape_social_ping_reply(msg.content)
        if social_reply:
            await self._persist_short_reply(session, user_text=msg.content, reply_text=social_reply)
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, social_reply)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=social_reply,
                metadata=msg.metadata or {},
            )

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0, include_assistant_text=False)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        final_content = self._shape_knowledge_probe_reply(
            msg.content,
            final_content,
        ) or final_content

        final_content = self._shape_status_reply(
            msg.content,
            final_content,
            has_recent_event=self.context.has_recent_life_event(),
        ) or final_content

        final_content = self._strip_short_reply_terminal_punct(
            msg.content,
            final_content,
        ) or final_content

        if self._is_internal_fallback_output(final_content):
            logger.warning("Suppressing internal fallback output for {}:{}", msg.channel, msg.sender_id)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return None

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                continue  # never persist tool traces to session history/memory
            if role == "assistant":
                if entry.get("tool_calls"):
                    continue  # skip assistant tool-call scaffolding
                if self._is_internal_fallback_output(content if isinstance(content, str) else None):
                    continue  # skip internal fallback/status placeholders
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
