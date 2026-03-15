"""Config for life-memory forgetting, reinforcement, and retrieval thresholds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ScoringWeights:
    importance: float = 0.24
    self_relevance: float = 0.16
    relationship_relevance: float = 0.14
    emotional_weight: float = 0.16
    novelty: float = 0.15
    source_confidence: float = 0.15


@dataclass
class StrengthConfig:
    alpha_detail: float = 0.72
    alpha_gist: float = 0.90
    max_strength: float = 1.0


@dataclass
class DecayConfig:
    lambda_detail: float = 0.045
    lambda_gist: float = 0.018
    salience_shield: float = 0.70
    emotional_shield: float = 0.65
    relationship_shield: float = 0.50
    self_relevance_shield: float = 0.30
    retrieval_shield: float = 0.22
    min_protection: float = 0.30


@dataclass
class RetrievalConfig:
    T_detail: float = 0.42
    T_gist: float = 0.24
    min_relevance: float = 0.08
    max_results: int = 6


@dataclass
class ReinforcementConfig:
    r_detail: float = 0.24
    r_gist: float = 0.12
    k_detail: float = 0.28
    k_gist: float = 0.20


@dataclass
class InterferenceConfig:
    gamma_detail: float = 0.85
    gamma_gist: float = 0.25
    cluster_half_life_hours: float = 72.0
    routine_pressure_boost: float = 0.35


@dataclass
class PermanenceConfig:
    pinned_detail_floor: float = 0.52
    pinned_gist_floor: float = 0.70
    tier_decay_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "pinned": 0.15,
            "durable": 0.45,
            "normal": 1.00,
            "volatile": 1.80,
        }
    )
    tier_min_floors: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "pinned": {"detail": 0.52, "gist": 0.70},
            "durable": {"detail": 0.12, "gist": 0.20},
            "normal": {"detail": 0.00, "gist": 0.00},
            "volatile": {"detail": 0.00, "gist": 0.00},
        }
    )
    pinned_salience_threshold: float = 0.82
    durable_salience_threshold: float = 0.68
    volatile_salience_threshold: float = 0.32


@dataclass
class MemoryForgettingConfig:
    """Top-level tunable forgetting config."""

    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    strength: StrengthConfig = field(default_factory=StrengthConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reinforcement: ReinforcementConfig = field(default_factory=ReinforcementConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    permanence: PermanenceConfig = field(default_factory=PermanenceConfig)

    @classmethod
    def from_workspace(cls, workspace: Path) -> "MemoryForgettingConfig":
        """Load optional workspace overrides from MEMORY_FORGETTING.json."""
        path = workspace / "MEMORY_FORGETTING.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Memory config ignored (not an object): {}", path)
                return cls()
            cfg = cls()
            _merge_dataclass(cfg, data)
            return cfg
        except Exception as exc:
            logger.warning("Memory config load failed {}: {}", path, exc)
            return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(obj: Any, updates: dict[str, Any]) -> None:
    """Recursively merge dict values into a dataclass object."""
    if not is_dataclass(obj):
        return
    field_map = {f.name: f for f in fields(obj)}
    for key, value in updates.items():
        if key not in field_map:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
            continue
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
            setattr(obj, key, current)
            continue
        setattr(obj, key, value)

