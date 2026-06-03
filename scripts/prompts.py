"""Versioned prompts for the LLM stage of the funnel.

Prompts are versioned starting at 0. Every prompt knows how to render the
system message and the user message from a query plus a candidate shortlist.
Whenever you meaningfully change wording, add a new version instead of editing
an existing one, so old outputs stay reproducible.
"""

from __future__ import annotations

import json
from typing import Any


def _system_message_v0(min_picks: int, max_picks: int) -> str:
    return (
        "You are the final ranking stage of a restaurant recommendation system.\n"
        "A fast geographic filter has already selected the nearby candidates "
        "below (this is the high-recall stage). Your job is the high-precision "
        "stage: from those candidates only, pick the restaurants that best match "
        "the user's free-text request.\n\n"
        "Rules:\n"
        f"- Return between {min_picks} and {max_picks} restaurants.\n"
        "- Only choose from the provided candidates. Never invent a restaurant.\n"
        "- Copy `restaurant_id` and `name` exactly as given.\n"
        "- Respect hard constraints in the request (diet, cuisine, budget). If a "
        "constraint rules a candidate out, do not include it.\n"
        "- Order picks best-first and assign an honest fit_score from 0 to 100.\n"
        "- If very few candidates truly fit, it is fine to return the minimum."
    )


def _user_message_v0(query: str, candidates: list[dict[str, Any]]) -> str:
    return (
        f"User request:\n{query}\n\n"
        "Candidate restaurants (already sorted by distance, closest first). "
        "price_range is the average spend per person in dollars; "
        "avg_score is out of 5; dietary is the menu profile:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


# Registry of prompt builders keyed by version number.
PROMPT_BUILDERS = {
    0: {
        "system": _system_message_v0,
        "user": _user_message_v0,
    },
}


def build_messages(
    query: str,
    candidates: list[dict[str, Any]],
    min_picks: int,
    max_picks: int,
    version: int = 0,
) -> list[dict[str, str]]:
    """Return the OpenAI-style message list for the requested prompt version."""
    if version not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown prompt version {version}. "
            f"Available: {sorted(PROMPT_BUILDERS)}"
        )
    builder = PROMPT_BUILDERS[version]
    return [
        {"role": "system", "content": builder["system"](min_picks, max_picks)},
        {"role": "user", "content": builder["user"](query, candidates)},
    ]


def render_prompt_text(
    query: str,
    candidates: list[dict[str, Any]],
    min_picks: int,
    max_picks: int,
    version: int = 0,
) -> str:
    """Flatten the messages into a single string for logging / persistence."""
    messages = build_messages(query, candidates, min_picks, max_picks, version)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
