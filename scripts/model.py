"""Pydantic models used as the structured-output contract with the LLM.

Keeping these in one place means the prompt, the parser and the rest of the
code all agree on exactly what a "recommendation" looks like.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RestaurantPick(BaseModel):
    """A single restaurant the LLM decided to keep at the bottom of the funnel."""

    restaurant_id: int = Field(
        ..., description="The restaurant_id taken verbatim from the candidate list."
    )
    name: str = Field(..., description="The restaurant name, copied from the candidate.")
    fit_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="How well this restaurant matches the user request (0-100).",
    )
    reason: str = Field(
        ...,
        description="One or two sentences explaining why this restaurant was chosen.",
    )


class RecommendationResponse(BaseModel):
    """The full structured answer returned by the LLM stage of the funnel."""

    picks: list[RestaurantPick] = Field(
        ..., description="Ordered best-first list of recommended restaurants."
    )
    summary: str = Field(
        ...,
        description="A short, friendly summary of the overall recommendation set.",
    )
