"""Deterministic event-stream prehistory generation for LifeState bootstrap."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nanobot.companion.life_state.memory_utils import now_local, parse_iso, to_iso
from nanobot.companion.life_state.prehistory_models import (
    PrehistoryEvent,
    PrehistoryProfile,
    PrehistoryResult,
)
from nanobot.companion.life_state.prehistory_templates import (
    PREFERENCE_TEMPLATES,
    RELATIONSHIP_EVENT_TEMPLATES,
    ROLE_ROUTINES,
    SALIENT_INCIDENT_TEMPLATES,
)


class PrehistoryBootstrapGenerator:
    """Generate structured prior-life events that feed the existing memory pipeline."""

    VERSION = "prehistory-v1"

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def build_profile(self, *, overrides: dict[str, Any] | None = None) -> PrehistoryProfile:
        """Load profile from workspace + optional explicit overrides."""
        return PrehistoryProfile.from_workspace(self.workspace, overrides=overrides)

    def generate(
        self,
        *,
        profile: PrehistoryProfile,
        now: datetime | None = None,
        seed: int | None = None,
        horizon_days: int | None = None,
    ) -> PrehistoryResult:
        """Generate deterministic event timeline and a causally derived current snapshot."""
        target_now = (now or now_local()).replace(microsecond=0)
        horizon = int(horizon_days or self._resolve_horizon_days(profile))
        horizon = max(45, min(540, horizon))
        start = (target_now - timedelta(days=horizon)).replace(hour=6, minute=0, second=0, microsecond=0)
        resolved_seed = self._resolve_seed(profile=profile, now=target_now, explicit_seed=seed)
        rng = random.Random(resolved_seed)

        events: list[PrehistoryEvent] = []
        events.extend(self._generate_routine_events(profile, start=start, now=target_now, rng=rng))
        events.extend(self._generate_identity_events(profile, start=start, now=target_now, rng=rng))
        events.extend(self._generate_preference_events(profile, start=start, now=target_now, rng=rng))
        events.extend(self._generate_salient_incidents(profile, start=start, now=target_now, rng=rng))
        events.extend(self._generate_role_milestones(profile, start=start, now=target_now, rng=rng))
        events.extend(self._generate_relationship_events(profile, start=start, now=target_now, rng=rng))

        normalized = self._normalize_and_assign_ids(events, start=start, now=target_now)
        final_state = self._synthesize_state_from_timeline(normalized, now=target_now)
        recent_log_events = self._pick_recent_log_events(normalized, now=target_now)
        summary = self._build_summary(normalized, now=target_now)
        metadata = {
            "generator_version": self.VERSION,
            "seed": resolved_seed,
            "generated_at": to_iso(target_now),
            "horizon_days": horizon,
            "profile": profile.to_metadata(),
            "summary": summary,
            "relationship_mode": "known_user" if profile.relationship_with_user else "no_shared_history",
        }
        return PrehistoryResult(
            events=normalized,
            final_state=final_state,
            metadata=metadata,
            recent_log_events=recent_log_events,
            summary=summary,
        )

    def _resolve_horizon_days(self, profile: PrehistoryProfile) -> int:
        stage = profile.life_stage.strip().lower()
        if stage in {"teen", "adolescent"}:
            return 90
        if stage in {"young_adult", "student"}:
            return 120
        if stage in {"adult", "working_adult"}:
            return 180
        if stage in {"midlife"}:
            return 240
        return 120

    def _resolve_seed(
        self,
        *,
        profile: PrehistoryProfile,
        now: datetime,
        explicit_seed: int | None,
    ) -> int:
        if explicit_seed is not None:
            return int(explicit_seed)
        if profile.explicit_seed is not None:
            return int(profile.explicit_seed)
        base = "|".join(
            [
                self.VERSION,
                str(self.workspace.resolve()),
                profile.profile_hash(),
                now.strftime("%Y-%m-%d"),
            ]
        )
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    def _generate_routine_events(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        role_templates = ROLE_ROUTINES.get(profile.role) or ROLE_ROUTINES["general"]
        phases = [x for x in profile.routine_phases if x in role_templates] or list(role_templates.keys())

        events: list[PrehistoryEvent] = []
        total_days = max(1, (now.date() - start.date()).days)
        for offset in range(total_days + 1):
            day = start + timedelta(days=offset)
            age_days = max(0, (now.date() - day.date()).days)
            if age_days > 30 and rng.random() < 0.42:
                continue

            phase_count = self._phase_count_for_age(age_days, rng)
            selected_phases = self._pick_phases(phases, phase_count=phase_count, rng=rng)
            for phase in selected_phases:
                options = role_templates.get(phase) or []
                if not options:
                    continue
                tpl = dict(rng.choice(options))
                hour_min, hour_max = tpl.get("hour_range", (9, 18))
                hour = rng.randint(int(hour_min), int(hour_max))
                minute = rng.choice([0, 10, 20, 30, 40, 50])
                stamp = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if stamp > now:
                    continue
                volatility = 0.10 if age_days <= 7 else (0.26 if age_days <= 30 else 0.50)
                importance = max(0.12, float(tpl.get("importance", 0.35)) - volatility * rng.uniform(0.05, 0.40))
                events.append(
                    PrehistoryEvent(
                        time=to_iso(stamp),
                        type="routine",
                        summary=str(tpl.get("summary") or "Routine life trace."),
                        gist=str(tpl.get("gist") or "Routine life trace."),
                        detail=self._detail_line(
                            summary=str(tpl.get("summary") or ""),
                            phase=phase,
                            profile=profile,
                        ),
                        source="prehistory_bootstrap",
                        importance=importance,
                        self_relevance=0.82,
                        relationship_relevance=0.14,
                        emotional_weight=0.22 + rng.uniform(-0.05, 0.08),
                        novelty=max(0.05, 0.45 - volatility - rng.uniform(0.0, 0.10)),
                        source_confidence=0.94,
                        location=str(tpl.get("location") or "home"),
                        activity=str(tpl.get("activity") or "rest"),
                        tags=["routine", phase],
                    )
                )
        return events

    def _phase_count_for_age(self, age_days: int, rng: random.Random) -> int:
        if age_days <= 7:
            return rng.randint(3, 5)
        if age_days <= 30:
            return rng.randint(2, 3)
        if age_days <= 90:
            return rng.randint(1, 2)
        return 1 if rng.random() < 0.72 else 0

    def _pick_phases(self, phases: list[str], *, phase_count: int, rng: random.Random) -> list[str]:
        if phase_count <= 0:
            return []
        if phase_count >= len(phases):
            return list(phases)
        picks = list(phases)
        rng.shuffle(picks)
        return sorted(picks[:phase_count], key=lambda x: phases.index(x))

    def _generate_identity_events(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        anchors = profile.identity_facts + profile.seed_facts
        if not anchors:
            return []
        span_days = max(15, (now - start).days)
        events: list[PrehistoryEvent] = []
        for idx, fact in enumerate(anchors):
            frac = (idx + 1) / float(len(anchors) + 1)
            days_after_start = int(frac * max(10, span_days - 10))
            stamp = start + timedelta(days=days_after_start, hours=9 + (idx % 3))
            stamp += timedelta(days=rng.randint(-4, 4))
            if stamp >= now:
                stamp = now - timedelta(days=3 + idx, hours=2)
            text = str(fact).strip()
            if not text:
                continue
            events.append(
                PrehistoryEvent(
                    time=to_iso(stamp),
                    type="identity_milestone",
                    summary=f"Identity milestone: {text}",
                    gist=f"Core identity anchor: {text}",
                    detail=f"Identity anchor retained from past period: {text}",
                    source="prehistory_bootstrap",
                    importance=3.0,
                    self_relevance=0.96,
                    relationship_relevance=0.46 if "relationship" in text.lower() else 0.30,
                    emotional_weight=0.74,
                    novelty=0.84,
                    source_confidence=0.98,
                    pinned=True,
                    core_memory=True,
                    tags=["identity", "pinned"],
                )
            )
        return events

    def _generate_preference_events(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        templates = list(PREFERENCE_TEMPLATES)
        rng.shuffle(templates)
        use = templates[: min(3, len(templates))]
        events: list[PrehistoryEvent] = []
        for idx, tpl in enumerate(use):
            when = now - timedelta(days=45 - idx * 11 + rng.randint(-3, 4), hours=rng.randint(3, 10))
            if when < start:
                when = start + timedelta(days=12 + idx * 5, hours=10)
            events.append(
                PrehistoryEvent(
                    time=to_iso(when),
                    type="preference_forming",
                    summary=str(tpl["summary"]),
                    gist=str(tpl["gist"]),
                    detail=self._detail_line(
                        summary=str(tpl["summary"]),
                        phase="preference_forming",
                        profile=profile,
                    ),
                    source="prehistory_bootstrap",
                    importance=1.75,
                    self_relevance=0.88,
                    relationship_relevance=0.18,
                    emotional_weight=0.58,
                    novelty=0.62,
                    source_confidence=0.96,
                    tags=["preference"],
                )
            )
        return events

    def _generate_salient_incidents(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        templates = list(SALIENT_INCIDENT_TEMPLATES)
        rng.shuffle(templates)
        take = rng.randint(2, min(4, len(templates)))
        events: list[PrehistoryEvent] = []
        for idx, tpl in enumerate(templates[:take]):
            when = now - timedelta(days=18 + idx * 17 + rng.randint(-4, 5), hours=rng.randint(1, 9))
            if when < start:
                when = start + timedelta(days=20 + idx * 7, hours=15)
            events.append(
                PrehistoryEvent(
                    time=to_iso(when),
                    type="salient_incident",
                    summary=str(tpl["summary"]),
                    gist=str(tpl["gist"]),
                    detail=self._detail_line(
                        summary=str(tpl["summary"]),
                        phase="salient_incident",
                        profile=profile,
                    ),
                    source="prehistory_bootstrap",
                    importance=float(tpl["importance"]),
                    self_relevance=0.90,
                    relationship_relevance=0.24,
                    emotional_weight=float(tpl["emotional_weight"]),
                    novelty=0.76,
                    source_confidence=0.97,
                    tags=["salient"],
                )
            )
        return events

    def _generate_role_milestones(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        interests = list(profile.interests) or ["daily work"]
        count = min(4, max(2, len(interests)))
        events: list[PrehistoryEvent] = []
        for idx in range(count):
            interest = interests[idx % len(interests)]
            when = now - timedelta(days=72 - idx * 18 + rng.randint(-5, 5), hours=rng.randint(1, 8))
            if when < start:
                when = start + timedelta(days=15 + idx * 10, hours=11)
            events.append(
                PrehistoryEvent(
                    time=to_iso(when),
                    type="milestone",
                    summary=f"Completed a notable {profile.role} milestone in {interest}.",
                    gist=f"Reached a meaningful milestone in {interest}.",
                    detail=self._detail_line(
                        summary=f"Milestone progression in {interest}",
                        phase="milestone",
                        profile=profile,
                    ),
                    source="prehistory_bootstrap",
                    importance=2.25,
                    self_relevance=0.91,
                    relationship_relevance=0.30,
                    emotional_weight=0.66,
                    novelty=0.71,
                    source_confidence=0.97,
                    tags=["milestone", profile.role],
                )
            )
        return events

    def _generate_relationship_events(
        self,
        profile: PrehistoryProfile,
        *,
        start: datetime,
        now: datetime,
        rng: random.Random,
    ) -> list[PrehistoryEvent]:
        if not profile.relationship_with_user:
            return []

        events: list[PrehistoryEvent] = []
        span_days = max(25, (now - start).days)
        first_contact_days_ago = min(span_days - 7, max(15, int(span_days * 0.55) + rng.randint(-7, 8)))
        first_contact = now - timedelta(days=first_contact_days_ago, hours=11)
        first_tpl = RELATIONSHIP_EVENT_TEMPLATES["first_contact"]
        events.append(
            PrehistoryEvent(
                time=to_iso(first_contact),
                type="relationship_milestone",
                summary=str(first_tpl["summary"]),
                gist=str(first_tpl["gist"]),
                detail="Relationship history anchor: first meaningful user contact.",
                source="prehistory_bootstrap",
                importance=2.45,
                self_relevance=0.86,
                relationship_relevance=0.98,
                emotional_weight=0.71,
                novelty=0.82,
                source_confidence=0.99,
                tags=["relationship", "user"],
            )
        )

        early_tpl = RELATIONSHIP_EVENT_TEMPLATES["early_impression"]
        events.append(
            PrehistoryEvent(
                time=to_iso(first_contact + timedelta(days=3, hours=2)),
                type="relationship_milestone",
                summary=str(early_tpl["summary"]),
                gist=str(early_tpl["gist"]),
                detail="Captured early user impression from repeated interactions.",
                source="prehistory_bootstrap",
                importance=2.05,
                self_relevance=0.84,
                relationship_relevance=0.90,
                emotional_weight=0.60,
                novelty=0.69,
                source_confidence=0.98,
                tags=["relationship", "user"],
            )
        )

        cur = first_contact + timedelta(days=5)
        while cur < now - timedelta(days=1):
            gap = rng.randint(3, 10)
            cur += timedelta(days=gap, hours=rng.randint(0, 6))
            if cur >= now:
                break
            routine_tpl = RELATIONSHIP_EVENT_TEMPLATES["routine_touch"]
            events.append(
                PrehistoryEvent(
                    time=to_iso(cur),
                    type="relationship_trace",
                    summary=str(routine_tpl["summary"]),
                    gist=str(routine_tpl["gist"]),
                    detail="Repeated user interaction trace.",
                    source="prehistory_bootstrap",
                    importance=0.72,
                    self_relevance=0.70,
                    relationship_relevance=0.84,
                    emotional_weight=0.44,
                    novelty=0.28,
                    source_confidence=0.97,
                    tags=["relationship", "user", "routine"],
                )
            )

        shared_tpl = RELATIONSHIP_EVENT_TEMPLATES["shared_moment"]
        shared_when = now - timedelta(days=max(4, min(28, int(span_days * 0.18))), hours=5)
        events.append(
            PrehistoryEvent(
                time=to_iso(shared_when),
                type="relationship_milestone",
                summary=str(shared_tpl["summary"]),
                gist=str(shared_tpl["gist"]),
                detail="High-value shared moment with user.",
                source="prehistory_bootstrap",
                importance=2.35,
                self_relevance=0.86,
                relationship_relevance=0.95,
                emotional_weight=0.74,
                novelty=0.78,
                source_confidence=0.99,
                tags=["relationship", "user", "milestone"],
            )
        )

        if profile.relationship_trust >= 0.55:
            promise_tpl = RELATIONSHIP_EVENT_TEMPLATES["promise"]
            promise_when = now - timedelta(days=max(3, min(16, int(span_days * 0.12))), hours=2)
            events.append(
                PrehistoryEvent(
                    time=to_iso(promise_when),
                    type="promise",
                    summary=str(promise_tpl["summary"]),
                    gist=str(promise_tpl["gist"]),
                    detail="Explicit user-facing promise retained as core memory.",
                    source="prehistory_bootstrap",
                    importance=2.9,
                    self_relevance=0.90,
                    relationship_relevance=0.99,
                    emotional_weight=0.81,
                    novelty=0.73,
                    source_confidence=1.0,
                    pinned=True,
                    core_memory=True,
                    tags=["relationship", "user", "promise", "pinned"],
                )
            )

        if profile.relationship_conflict_last7d >= 0.20:
            conflict_tpl = RELATIONSHIP_EVENT_TEMPLATES["conflict"]
            conflict_when = now - timedelta(days=max(2, min(14, int(span_days * 0.09))), hours=4)
            events.append(
                PrehistoryEvent(
                    time=to_iso(conflict_when),
                    type="relationship_incident",
                    summary=str(conflict_tpl["summary"]),
                    gist=str(conflict_tpl["gist"]),
                    detail="Resolved conflict trace with user.",
                    source="prehistory_bootstrap",
                    importance=1.95,
                    self_relevance=0.84,
                    relationship_relevance=0.93,
                    emotional_weight=0.68,
                    novelty=0.61,
                    source_confidence=0.99,
                    tags=["relationship", "user", "conflict"],
                )
            )
        return events

    def _normalize_and_assign_ids(
        self,
        events: list[PrehistoryEvent],
        *,
        start: datetime,
        now: datetime,
    ) -> list[PrehistoryEvent]:
        normalized: list[tuple[datetime, PrehistoryEvent]] = []
        for event in events:
            stamp = parse_iso(event.time)
            if stamp is None:
                continue
            if stamp < start or stamp > now:
                continue
            summary = (event.summary or "").strip()
            if not summary:
                continue
            normalized.append((stamp, event))

        normalized.sort(key=lambda item: (item[0], item[1].type, item[1].summary))
        out: list[PrehistoryEvent] = []
        for idx, (stamp, event) in enumerate(normalized, start=1):
            key = f"{to_iso(stamp)}|{event.type}|{event.summary}"
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
            eid = event.event_id or f"pre_{idx:05d}_{digest}"
            out.append(replace(event, time=to_iso(stamp), event_id=eid))
        return out

    def _pick_recent_log_events(self, events: list[PrehistoryEvent], *, now: datetime) -> list[PrehistoryEvent]:
        cutoff = now - timedelta(days=3)
        recent = [e for e in events if (parse_iso(e.time) or now) >= cutoff]
        if not recent:
            recent = events[-12:]
        return recent[-12:]

    def _build_summary(self, events: list[PrehistoryEvent], *, now: datetime) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        tier_counts = {"pinned": 0, "durable": 0, "normal": 0, "volatile": 0}
        time_scales = {"recent_days": 0, "medium_weeks": 0, "deep_months": 0}
        routine_count = 0
        relationship_count = 0

        oldest = None
        newest = None
        for e in events:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
            stamp = parse_iso(e.time) or now
            oldest = stamp if oldest is None else min(oldest, stamp)
            newest = stamp if newest is None else max(newest, stamp)
            age_days = max(0.0, (now - stamp).total_seconds() / 86400.0)
            if age_days <= 7:
                time_scales["recent_days"] += 1
            elif age_days <= 30:
                time_scales["medium_weeks"] += 1
            else:
                time_scales["deep_months"] += 1

            if e.pinned or e.core_memory:
                tier_counts["pinned"] += 1
            elif e.importance >= 2.0 or e.emotional_weight >= 0.70 or e.relationship_relevance >= 0.70:
                tier_counts["durable"] += 1
            elif e.importance <= 0.45 and e.novelty <= 0.35:
                tier_counts["volatile"] += 1
            else:
                tier_counts["normal"] += 1

            if e.type == "routine" or "routine" in e.tags:
                routine_count += 1
            if e.relationship_relevance >= 0.70 or "relationship" in e.type:
                relationship_count += 1

        return {
            "event_count": len(events),
            "type_counts": type_counts,
            "tier_counts": tier_counts,
            "time_scales": time_scales,
            "routine_traces": routine_count,
            "relationship_events": relationship_count,
            "oldest_event_at": to_iso(oldest) if oldest else None,
            "latest_event_at": to_iso(newest) if newest else None,
        }

    def _synthesize_state_from_timeline(self, events: list[PrehistoryEvent], *, now: datetime) -> dict[str, Any]:
        state: dict[str, Any] = {
            "location": "home",
            "activity": "rest",
            "mood": "calm",
            "mood_score": 1,
            "energy": 68,
            "social_battery": 64,
            "urgency_bias": 35,
            "busy_level": 40,
            "reply_delay_s": 8,
            "verbosity": 0.6,
            "override_until": None,
            "override_reason": None,
            "override_activity": None,
            "override_location": None,
            "override_busy_level": None,
        }
        last_tick = now - timedelta(minutes=35)

        for event in events:
            stamp = parse_iso(event.time) or now
            self._apply_event_to_state(state, event)
            last_tick = stamp

        state["last_tick"] = to_iso(last_tick)
        state["last_update"] = to_iso(last_tick)
        # next_transition_at is set by LifeStateService after schema normalization.
        state["next_transition_at"] = to_iso(last_tick + timedelta(minutes=35))
        return state

    def _apply_event_to_state(self, state: dict[str, Any], event: PrehistoryEvent) -> None:
        activity = (event.activity or "").strip().lower()
        location = (event.location or "").strip()
        summary = event.summary.lower()

        if location:
            state["location"] = location

        if activity:
            state["activity"] = activity
        elif event.type in {"relationship_trace", "relationship_milestone"}:
            state["activity"] = "chat"
        elif event.type in {"milestone", "salient_incident", "preference_forming"}:
            state["activity"] = "reflection"

        delta_energy, delta_social = self._activity_delta(str(state.get("activity") or "rest"))
        if "stress" in summary or "conflict" in summary:
            delta_energy -= 3
        if "meal" in summary or "lunch" in summary or "dinner" in summary:
            delta_energy += 3
        if "user" in summary or event.relationship_relevance >= 0.75:
            delta_social += 3

        state["energy"] = _clamp_int(int(state.get("energy", 68)) + delta_energy, 0, 100, 68)
        state["social_battery"] = _clamp_int(int(state.get("social_battery", 64)) + delta_social, 0, 100, 64)

        mood_score = _clamp_int(state.get("mood_score"), -5, 5, 1)
        if event.emotional_weight >= 0.75 and ("stress" in summary or "conflict" in summary):
            mood_score -= 2
        elif event.emotional_weight >= 0.65:
            mood_score += 1
        if state["energy"] <= 25:
            mood_score -= 1
        if state["social_battery"] <= 20:
            mood_score -= 1
        if state["energy"] >= 72 and state["social_battery"] >= 55:
            mood_score += 1
        mood_score = max(-5, min(5, mood_score))
        state["mood_score"] = mood_score
        state["mood"] = _mood_label(mood_score)

        busy_map = {
            "study": 76,
            "work": 78,
            "commute": 62,
            "meal": 38,
            "rest": 28,
            "chat": 46,
            "reflection": 34,
            "sleep": 12,
        }
        urgency_map = {
            "study": 72,
            "work": 75,
            "commute": 58,
            "meal": 32,
            "rest": 24,
            "chat": 42,
            "reflection": 30,
            "sleep": 14,
        }
        busy = busy_map.get(str(state.get("activity") or "rest"), 40)
        urgency = urgency_map.get(str(state.get("activity") or "rest"), 35)
        if event.importance >= 2.1:
            busy += 8
            urgency += 7
        state["busy_level"] = _clamp_int(busy, 0, 100, 40)
        state["urgency_bias"] = _clamp_int(urgency, 0, 100, 35)

        if state["busy_level"] >= 70:
            state["reply_delay_s"] = 18
            state["verbosity"] = 0.42
        elif state["activity"] == "sleep":
            state["reply_delay_s"] = 28
            state["verbosity"] = 0.28
        else:
            state["reply_delay_s"] = 8
            state["verbosity"] = 0.62 if state["social_battery"] >= 40 else 0.48

    def _activity_delta(self, activity: str) -> tuple[int, int]:
        if activity == "sleep":
            return 12, 7
        if activity == "meal":
            return 5, 1
        if activity in {"study", "work"}:
            return -8, -6
        if activity == "commute":
            return -7, -3
        if activity == "chat":
            return -2, 4
        if activity == "reflection":
            return -1, 1
        return 4, 3

    def _detail_line(self, *, summary: str, phase: str, profile: PrehistoryProfile) -> str:
        role = profile.role
        city = profile.city or "local area"
        return f"{summary} | phase={phase} | role={role} | context={city}"


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(low, min(high, int(value)))
    if isinstance(value, str):
        try:
            return max(low, min(high, int(float(value.strip()))))
        except Exception:
            return default
    return default


def _mood_label(score: int) -> str:
    if score <= -3:
        return "low"
    if score <= -1:
        return "tense"
    if score <= 1:
        return "calm"
    return "good"

