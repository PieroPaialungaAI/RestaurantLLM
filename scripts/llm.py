"""Thin functional wrapper around the OpenAI API.

This module deliberately stays object-free: it exposes a couple of small
functions that the ``RestaurantRecommender`` object calls. All the OpenAI
specifics (client creation, structured parsing) live here so the rest of the
codebase never imports ``openai`` directly.
"""

from __future__ import annotations

from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from scripts.constants import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE

T = TypeVar("T", bound=BaseModel)


def build_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client, falling back to the key in constants/env."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment "
            "variable before running the LLM stage."
        )
    return OpenAI(api_key=key)


def complete_structured(
    client: OpenAI,
    messages: list[dict[str, str]],
    response_model: Type[T],
    model: str = OPENAI_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
) -> T:
    """Call the chat completions API and parse the answer into ``response_model``.

    Uses OpenAI structured outputs so the response is guaranteed to match the
    pydantic schema (or raises if the model refused).
    """
    completion = client.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=messages,
        response_format=response_model,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"The model refused to answer: {message.refusal}")
    return message.parsed
