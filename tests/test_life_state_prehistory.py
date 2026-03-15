from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.companion.life_state.memory_interference import apply_interference_penalty
from nanobot.companion.life_state.memory_utils import now_local, parse_iso
from nanobot.companion.life_state.service import LifeStateService


def _raw_events(service: LifeStateService) -> list[dict]:
    return list(service.memory_engine.store.iter_raw_events())


def _entries(service: LifeStateService):
    return service.memory_engine._load_entries()


@pytest.mark.asyncio
async def test_first_run_bootstrap_builds_raw_events_state_and_index(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    status = await service.ensure_prehistory_bootstrap()

    assert status["bootstrapped"] is True
    assert (tmp_path / "memory" / "LIFE_EVENTS.jsonl").exists()
    assert (tmp_path / "memory" / "LIFE_MEMORY_INDEX.json").exists()
    assert (tmp_path / "memory" / "PREHISTORY_META.json").exists()
    assert _raw_events(service)
    assert _entries(service)

    state = await service.get_state()
    assert state.get("last_tick")
    assert state.get("next_transition_at")
    assert state.get("activity")
    assert state.get("location")


@pytest.mark.asyncio
async def test_generated_history_spans_days_weeks_and_months(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    now = now_local()
    ages: list[float] = []
    for event in _raw_events(service):
        stamp = parse_iso(event.get("time"))
        if not stamp:
            continue
        ages.append((now - stamp).total_seconds() / 86400.0)

    assert ages
    assert any(age <= 3.0 for age in ages)
    assert any(7.0 < age <= 30.0 for age in ages)
    assert any(age > 45.0 for age in ages)


@pytest.mark.asyncio
async def test_current_snapshot_tracks_latest_generated_timeline_state(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    raw = _raw_events(service)
    assert raw
    latest_with_activity = next((e for e in reversed(raw) if str(e.get("activity") or "").strip()), None)
    latest_with_location = next((e for e in reversed(raw) if str(e.get("location") or "").strip()), None)
    assert latest_with_activity is not None
    assert latest_with_location is not None

    state = await service.get_state()
    state_tick = parse_iso(state.get("last_tick"))
    latest_stamp = parse_iso(raw[-1].get("time"))

    assert state_tick is not None
    assert latest_stamp is not None
    assert state_tick >= latest_stamp
    assert state.get("activity") == latest_with_activity.get("activity")
    assert state.get("location") == latest_with_location.get("location")


@pytest.mark.asyncio
async def test_old_ordinary_events_are_partially_forgotten_after_bootstrap(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    now = now_local()
    ordinary = []
    for entry in _entries(service):
        stamp = parse_iso(entry.timestamp_last)
        if not stamp:
            continue
        age_days = (now - stamp).total_seconds() / 86400.0
        if age_days > 7 and entry.permanence_tier in {"normal", "volatile"}:
            ordinary.append(entry)

    assert ordinary
    assert any(e.gist_strength < e.gist_strength_base for e in ordinary)
    assert any(e.detail_strength < e.detail_strength_base for e in ordinary)


@pytest.mark.asyncio
async def test_pinned_core_events_remain_retrievable(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    entries = _entries(service)
    pinned = [e for e in entries if e.pinned_flag or e.permanence_tier == "pinned"]
    assert pinned

    evidence = service.memory_engine.retrieve("identity promise", now=now_local(), limit=8)
    assert evidence
    assert any(item.pinned_flag for item in evidence)


@pytest.mark.asyncio
async def test_repeated_routine_traces_create_interference_clusters(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    routine_entries = [e for e in _entries(service) if e.memory_type == "routine"]
    assert len(routine_entries) >= 20
    assert max(e.similarity_cluster_pressure for e in routine_entries) > 0.0

    target = max(routine_entries, key=lambda e: e.similarity_cluster_pressure)
    detail_eff, gist_eff = apply_interference_penalty(target, service.memory_engine.config)
    assert detail_eff < target.detail_strength
    assert gist_eff <= target.gist_strength
    detail_ratio = detail_eff / target.detail_strength if target.detail_strength > 0 else 0.0
    gist_ratio = gist_eff / target.gist_strength if target.gist_strength > 0 else 0.0
    assert detail_ratio <= gist_ratio


@pytest.mark.asyncio
async def test_rebuild_from_raw_events_recreates_valid_index(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    raw_count = len(_raw_events(service))
    service.memory_engine.store.memory_index_path.write_text("{}", encoding="utf-8")
    rebuilt = await service.rebuild_memory_index()

    assert rebuilt == raw_count
    entries = _entries(service)
    assert len(entries) == raw_count
    assert all(e.id and e.event_ids for e in entries)


@pytest.mark.asyncio
async def test_restart_does_not_regenerate_existing_prehistory(tmp_path: Path) -> None:
    first = LifeStateService(tmp_path)
    first_status = await first.ensure_prehistory_bootstrap()
    assert first_status["bootstrapped"] is True

    raw_before = len(_raw_events(first))
    meta_before = await first.get_prehistory_summary()

    second = LifeStateService(tmp_path)
    second_status = await second.ensure_prehistory_bootstrap()
    assert second_status["bootstrapped"] is False
    assert second_status["reason"] in {"history_exists", "state_history_exists", "already_checked"}
    assert len(_raw_events(second)) == raw_before

    meta_after = await second.get_prehistory_summary()
    assert meta_after.get("seed") == meta_before.get("seed")
    assert meta_after.get("generator_version") == meta_before.get("generator_version")


@pytest.mark.asyncio
async def test_relationship_history_only_generated_when_relationship_context_exists(tmp_path: Path) -> None:
    service_plain = LifeStateService(tmp_path / "plain")
    await service_plain.ensure_prehistory_bootstrap()
    raw_plain = _raw_events(service_plain)
    assert not any(str(e.get("type") or "").startswith("relationship") for e in raw_plain)

    rel_ws = tmp_path / "rel"
    rel_ws.mkdir(parents=True, exist_ok=True)
    (rel_ws / "RELATIONSHIP.json").write_text(
        json.dumps(
            {
                "stage": "close",
                "intimacy": 0.82,
                "trust": 0.76,
                "conflict_last7d": 0.18,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    service_rel = LifeStateService(rel_ws)
    await service_rel.ensure_prehistory_bootstrap()
    raw_rel = _raw_events(service_rel)
    assert any(str(e.get("type") or "").startswith("relationship") for e in raw_rel)


@pytest.mark.asyncio
async def test_context_uses_recent_window_and_memory_retrieval_not_backstory_dump(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()

    raw = _raw_events(service)
    assert len(raw) > 40
    old = next(
        (
            e
            for e in raw
            if str(e.get("type") or "") in {"identity_milestone", "milestone", "preference_forming", "salient_incident"}
            and parse_iso(e.get("time"))
            and (now_local() - parse_iso(e.get("time"))).days > 30
        ),
        None,
    )
    assert old is not None

    context = ContextBuilder(tmp_path)
    system_prompt = context.build_system_prompt()
    assert str(old.get("summary")) not in system_prompt

    payload = await service.retrieve_memory_evidence("milestone identity routine", limit=6)
    assert isinstance(payload, dict)
    assert "prompt_block" in payload
    assert "Memory retrieval policy" in payload["prompt_block"]

    recent_events = context.get_recent_life_events(limit=3)
    assert len(recent_events) <= 3


@pytest.mark.asyncio
async def test_rebootstrap_with_same_seed_is_reproducible(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    first = await service.regenerate_prehistory(
        confirm_token="REGENERATE_PREHISTORY",
        seed=424242,
        dry_run=False,
    )
    raw_first = [dict(x) for x in _raw_events(service)]

    second = await service.regenerate_prehistory(
        confirm_token="REGENERATE_PREHISTORY",
        seed=424242,
        dry_run=False,
    )
    raw_second = [dict(x) for x in _raw_events(service)]

    assert first["bootstrapped"] is True
    assert second["bootstrapped"] is True
    assert len(raw_first) == len(raw_second)

    first_signature = [(x.get("time"), x.get("type"), x.get("summary")) for x in raw_first[:25]]
    second_signature = [(x.get("time"), x.get("type"), x.get("summary")) for x in raw_second[:25]]
    assert first_signature == second_signature


@pytest.mark.asyncio
async def test_destructive_rebootstrap_requires_explicit_confirm_token(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    await service.ensure_prehistory_bootstrap()
    with pytest.raises(ValueError):
        await service.regenerate_prehistory(dry_run=False)
