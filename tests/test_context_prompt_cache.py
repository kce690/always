"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder
from nanobot.utils.helpers import sync_workspace_templates


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("nanobot") / "templates"

    for filename in ContextBuilder.BOOTSTRAP_FILES:
        assert (template_dir / filename).is_file(), f"missing bootstrap template: {filename}"


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    # Runtime context is now merged with user message into a single message
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in user_content
    assert "Current Time:" in user_content
    assert "Channel: cli" in user_content
    assert "Chat ID: direct" in user_content
    assert "Return exactly: OK" in user_content


def test_sync_workspace_templates_creates_life_state_files(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)

    added = sync_workspace_templates(workspace, silent=True)

    assert "LIFESTATE.json" in added
    assert "RELATIONSHIP.json" in added
    assert "STYLE_PROFILE.json" in added
    assert "LIFELOG.md" in added


def test_system_prompt_includes_state_sections(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    sync_workspace_templates(workspace, silent=True)
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "# Companion Behavior (High Priority)" in prompt
    assert "1-2 short spoken sentences" in prompt
    assert "Ask at most one short follow-up question" in prompt
    assert "knowledge-probe questions" in prompt
    assert "Do not start explanation/teaching mode" in prompt
    assert "# Current Life State" in prompt
    assert "- Hidden state cues: location=home; activity=resting; mood=calm; energy=72." in prompt
    assert "- Grounded recent event (from LIFELOG): none" in prompt
    assert "do not claim you just finished a specific task" in prompt
    assert "# Relationship State" in prompt
    assert "- Hidden relationship cues: stage=warming_up, intimacy=0.35, trust=0.4, conflict_last7d=0." in prompt
    assert "# Style Profile" in prompt
    assert "- Hidden style cues: tone=gentle, verbosity=0.6, emoji=light, reply_delay_s=8." in prompt


def test_sync_workspace_templates_repairs_legacy_defaults(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "AGENTS.md").write_text(
        """# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

When the user asks for a recurring/periodic task, update `HEARTBEAT.md` instead of creating a one-time cron reminder.
""",
        encoding="utf-8",
    )
    (workspace / "SOUL.md").write_text(
        """# Soul

I am nanobot, a personal AI assistant.

- Helpful and friendly
""",
        encoding="utf-8",
    )
    (workspace / "LIFESTATE.json").write_text('{"location":"\ufffd\ufffd"}', encoding="utf-8")

    changed = sync_workspace_templates(workspace, silent=True)

    assert "AGENTS.md (updated)" in changed
    assert "SOUL.md (updated)" in changed
    assert "LIFESTATE.json (updated)" in changed
    assert "helpful AI assistant" not in (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert '"location": "home"' in (workspace / "LIFESTATE.json").read_text(encoding="utf-8")


def test_invalid_state_json_does_not_break_system_prompt(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "LIFESTATE.json").write_text("{", encoding="utf-8")
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "# Current Life State" not in prompt
