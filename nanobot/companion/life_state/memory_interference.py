"""Similarity-cluster interference for life-memory retrieval."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_models import MemoryEntry
from nanobot.companion.life_state.memory_scoring import looks_routine
from nanobot.companion.life_state.memory_utils import parse_iso


def recompute_cluster_pressure(
    entries: list[MemoryEntry],
    *,
    now: datetime,
    cfg: MemoryForgettingConfig,
) -> None:
    """Update per-entry cluster pressure from recency-weighted density."""
    by_cluster: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        by_cluster[entry.similarity_cluster_id].append(entry)

    for cluster_id, bucket in by_cluster.items():
        density = 0.0
        routine_count = 0
        for entry in bucket:
            stamp = parse_iso(entry.timestamp_last) or now
            age_hours = max(0.0, (now - stamp).total_seconds() / 3600.0)
            decay = math.exp(-age_hours / max(1.0, cfg.interference.cluster_half_life_hours))
            density += decay
            if looks_routine(entry.gist_summary) or entry.memory_type in {"state_transition", "routine"}:
                routine_count += 1

        pressure = max(0.0, density - 1.0)
        if routine_count > 2:
            pressure += cfg.interference.routine_pressure_boost * float(routine_count - 2)

        for entry in bucket:
            entry.similarity_cluster_pressure = max(0.0, pressure)


def estimate_cluster_pressure(
    entries: list[MemoryEntry],
    *,
    cluster_id: str,
    now: datetime,
    cfg: MemoryForgettingConfig,
) -> float:
    """Estimate pressure for a candidate entry before insertion."""
    if not cluster_id:
        return 0.0

    density = 0.0
    routine_count = 0
    for entry in entries:
        if entry.similarity_cluster_id != cluster_id:
            continue
        stamp = parse_iso(entry.timestamp_last) or now
        age_hours = max(0.0, (now - stamp).total_seconds() / 3600.0)
        density += math.exp(-age_hours / max(1.0, cfg.interference.cluster_half_life_hours))
        if looks_routine(entry.gist_summary) or entry.memory_type in {"state_transition", "routine"}:
            routine_count += 1

    pressure = max(0.0, density - 1.0)
    if routine_count > 2:
        pressure += cfg.interference.routine_pressure_boost * float(routine_count - 2)
    return max(0.0, pressure)


def apply_interference_penalty(
    entry: MemoryEntry,
    cfg: MemoryForgettingConfig,
) -> tuple[float, float]:
    """Compute effective retrieval strengths after interference."""
    pressure = max(0.0, entry.similarity_cluster_pressure)
    detail = entry.detail_strength / (1.0 + cfg.interference.gamma_detail * pressure)
    gist = entry.gist_strength / (1.0 + cfg.interference.gamma_gist * pressure)
    return max(0.0, detail), max(0.0, gist)

