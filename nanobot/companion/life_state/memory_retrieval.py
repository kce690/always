"""Query-time retrieval and detail/gist threshold gating."""

from __future__ import annotations

import math
from datetime import datetime

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_interference import apply_interference_penalty
from nanobot.companion.life_state.memory_models import MemoryEntry, MemoryEvidence
from nanobot.companion.life_state.memory_utils import parse_iso, tokenize


def retrieve_memories(
    entries: list[MemoryEntry],
    *,
    query: str,
    now: datetime,
    cfg: MemoryForgettingConfig,
    limit: int | None = None,
) -> list[MemoryEvidence]:
    """Retrieve memories with threshold-based detail/gist gating."""
    out: list[tuple[float, MemoryEvidence]] = []
    query_tokens = tokenize(query)
    max_results = limit or cfg.retrieval.max_results

    for entry in entries:
        relevance = _relevance_score(entry, query_tokens=query_tokens, now=now)
        detail_eff, gist_eff = apply_interference_penalty(entry, cfg)
        recall_level = _recall_level(detail_eff, gist_eff, cfg)
        if recall_level == "none":
            continue
        if relevance < cfg.retrieval.min_relevance and not entry.pinned_flag:
            continue

        score = relevance * (detail_eff if recall_level == "detail" else gist_eff * 0.92)
        evidence = MemoryEvidence(
            id=entry.id,
            recall_level=recall_level,
            text=entry.detail_text if recall_level == "detail" else entry.gist_summary,
            gist_summary=entry.gist_summary,
            event_ids=list(entry.event_ids),
            relevance_score=relevance,
            detail_strength_effective=detail_eff,
            gist_strength_effective=gist_eff,
            similarity_cluster_pressure=entry.similarity_cluster_pressure,
            permanence_tier=entry.permanence_tier,
            pinned_flag=entry.pinned_flag,
        )
        out.append((score, evidence))

    out.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in out[: max(1, int(max_results))]]


def _recall_level(detail_eff: float, gist_eff: float, cfg: MemoryForgettingConfig) -> str:
    if detail_eff >= cfg.retrieval.T_detail:
        return "detail"
    if gist_eff >= cfg.retrieval.T_gist:
        return "gist"
    return "none"


def _relevance_score(
    entry: MemoryEntry,
    *,
    query_tokens: list[str],
    now: datetime,
) -> float:
    gist_tokens = set(tokenize(entry.gist_summary))
    detail_tokens = set(tokenize(entry.detail_text))
    query_set = set(query_tokens)
    if not query_set:
        overlap = 0.08
    else:
        gist_overlap = len(query_set & gist_tokens) / max(1.0, float(len(query_set | gist_tokens)))
        detail_overlap = len(query_set & detail_tokens) / max(1.0, float(len(query_set | detail_tokens)))
        overlap = 0.62 * detail_overlap + 0.38 * gist_overlap

    stamp = parse_iso(entry.timestamp_last) or now
    age_hours = max(0.0, (now - stamp).total_seconds() / 3600.0)
    recency = 0.15 * math.exp(-age_hours / 336.0)
    pin_bonus = 0.10 if entry.pinned_flag else 0.0
    return max(0.0, min(1.0, overlap + recency + pin_bonus))

