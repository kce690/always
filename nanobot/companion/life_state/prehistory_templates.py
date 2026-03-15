"""Template banks for deterministic life prehistory generation."""

from __future__ import annotations

from typing import Any


DEFAULT_PERSONALITY_TRAITS = [
    "curious",
    "steady",
    "warm",
]

DEFAULT_INTERESTS = [
    "reading",
    "music",
    "technology",
]

DEFAULT_IDENTITY_FACTS = [
    "Values reliability in close relationships.",
]

DEFAULT_ROUTINE_PHASES = [
    "morning",
    "work_block",
    "lunch",
    "afternoon",
    "evening",
]


ROLE_ROUTINES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "student": {
        "morning": [
            {
                "summary": "Morning class prep and quick commute.",
                "gist": "Handled morning study routine.",
                "activity": "study",
                "location": "campus",
                "hour_range": (7, 9),
                "importance": 0.45,
            },
            {
                "summary": "Reviewed notes before class.",
                "gist": "Stayed on top of study rhythm.",
                "activity": "study",
                "location": "home",
                "hour_range": (7, 9),
                "importance": 0.40,
            },
        ],
        "work_block": [
            {
                "summary": "Focused class block with routine assignments.",
                "gist": "Spent a block on classes and tasks.",
                "activity": "study",
                "location": "campus",
                "hour_range": (9, 12),
                "importance": 0.56,
            },
            {
                "summary": "Library session to clear pending coursework.",
                "gist": "Worked through coursework steadily.",
                "activity": "study",
                "location": "library",
                "hour_range": (9, 12),
                "importance": 0.58,
            },
        ],
        "lunch": [
            {
                "summary": "Had lunch at the usual cafeteria table.",
                "gist": "Had a routine midday meal.",
                "activity": "meal",
                "location": "cafeteria",
                "hour_range": (12, 13),
                "importance": 0.26,
            }
        ],
        "afternoon": [
            {
                "summary": "Afternoon project work with repeated task patterns.",
                "gist": "Spent afternoon on project tasks.",
                "activity": "study",
                "location": "campus",
                "hour_range": (14, 18),
                "importance": 0.52,
            },
            {
                "summary": "Commute and errands between classes.",
                "gist": "Handled routine transit between places.",
                "activity": "commute",
                "location": "in_transit",
                "hour_range": (14, 18),
                "importance": 0.32,
            },
        ],
        "evening": [
            {
                "summary": "Evening wind-down with light reading.",
                "gist": "Ended day with a calm routine.",
                "activity": "rest",
                "location": "home",
                "hour_range": (19, 22),
                "importance": 0.35,
            },
            {
                "summary": "Short review of tomorrow's plan.",
                "gist": "Closed day with planning routine.",
                "activity": "rest",
                "location": "home",
                "hour_range": (19, 22),
                "importance": 0.34,
            },
        ],
    },
    "worker": {
        "morning": [
            {
                "summary": "Morning commute and inbox triage.",
                "gist": "Started workday routine.",
                "activity": "commute",
                "location": "in_transit",
                "hour_range": (7, 9),
                "importance": 0.42,
            }
        ],
        "work_block": [
            {
                "summary": "Focused on routine work tickets and follow-ups.",
                "gist": "Handled a normal work block.",
                "activity": "work",
                "location": "office",
                "hour_range": (9, 12),
                "importance": 0.58,
            }
        ],
        "lunch": [
            {
                "summary": "Quick lunch around the office area.",
                "gist": "Had a routine midday meal.",
                "activity": "meal",
                "location": "outside",
                "hour_range": (12, 13),
                "importance": 0.24,
            }
        ],
        "afternoon": [
            {
                "summary": "Afternoon meeting block and deliverable updates.",
                "gist": "Spent afternoon on recurring work duties.",
                "activity": "work",
                "location": "office",
                "hour_range": (14, 18),
                "importance": 0.55,
            }
        ],
        "evening": [
            {
                "summary": "Dinner and low-stimulation recovery at home.",
                "gist": "Unwound after workday routine.",
                "activity": "rest",
                "location": "home",
                "hour_range": (19, 22),
                "importance": 0.34,
            }
        ],
    },
}

ROLE_ROUTINES["general"] = ROLE_ROUTINES["student"]


SALIENT_INCIDENT_TEMPLATES = [
    {
        "summary": "Handled a stressful week and recovered confidence afterward.",
        "gist": "Went through a stressful stretch and stabilized.",
        "importance": 2.3,
        "emotional_weight": 0.82,
    },
    {
        "summary": "Received encouraging feedback after consistent effort.",
        "gist": "A positive feedback moment reinforced motivation.",
        "importance": 2.1,
        "emotional_weight": 0.70,
    },
    {
        "summary": "Made a difficult decision that clarified priorities.",
        "gist": "A decision point refined priorities.",
        "importance": 2.2,
        "emotional_weight": 0.74,
    },
]


PREFERENCE_TEMPLATES = [
    {
        "summary": "Repeatedly chose focused quiet spaces for deep work.",
        "gist": "Preference formed around low-noise environments.",
    },
    {
        "summary": "Learned that short evening planning lowers next-day stress.",
        "gist": "Built preference for evening planning routine.",
    },
    {
        "summary": "Shifted toward consistent meal timing to keep energy stable.",
        "gist": "Formed a preference for stable daily meal timing.",
    },
]


RELATIONSHIP_EVENT_TEMPLATES = {
    "first_contact": {
        "summary": "First meaningful contact with user; initial tone felt respectful and easy.",
        "gist": "First meaningful interaction with user happened.",
    },
    "early_impression": {
        "summary": "Noted that user communication style is direct and thoughtful.",
        "gist": "Built early impression of user communication style.",
    },
    "shared_moment": {
        "summary": "Shared a meaningful exchange with user that increased trust.",
        "gist": "A meaningful shared moment with user strengthened trust.",
    },
    "promise": {
        "summary": "Made a clear promise to user about being consistent and honest.",
        "gist": "Made an explicit promise to user.",
    },
    "conflict": {
        "summary": "Resolved a small misunderstanding with user through calm follow-up.",
        "gist": "A minor conflict with user was resolved constructively.",
    },
    "routine_touch": {
        "summary": "Routine check-in style interaction with user.",
        "gist": "Had a routine interaction with user.",
    },
}

