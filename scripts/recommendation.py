"""The recommendation funnel.

``RestaurantRecommender`` ties the two stages together:

1. **Distance shortlist (high recall).** Cheaply keep the geographically
   closest candidates around the user. We would rather over-include here.
2. **LLM rerank (high precision).** Hand the shortlist plus the free-text
   request to the model and let it return only the best 5-10 matches.

The object stays small: the geographic math lives in ``utils``, the prompt in
``prompts``, the API call in ``llm`` and the schema in ``model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from scripts import llm, prompts
from scripts.constants import (
    CITY_COORDINATES,
    DEFAULT_PROMPT_VERSION,
    MAX_RECOMMENDATIONS,
    MIN_RECOMMENDATIONS,
    N_DISTANCE_CANDIDATES,
    OPENAI_MODEL,
)
from scripts.dataloader import RestaurantDataLoader
from scripts.model import RecommendationResponse
from scripts.utils import candidates_to_payload, haversine_km_vectorized


@dataclass
class RecommendationResult:
    """Everything produced by a single recommendation run."""

    query: str
    city: str
    prompt_version: int
    prompt_text: str
    candidates: pd.DataFrame
    response: RecommendationResponse
    user_location: tuple[float, float]
    model: str
    extra: dict[str, Any] = field(default_factory=dict)


class RestaurantRecommender:
    """Runs the distance + LLM funnel for a single user request."""

    def __init__(
        self,
        loader: RestaurantDataLoader,
        n_candidates: int = N_DISTANCE_CANDIDATES,
        min_picks: int = MIN_RECOMMENDATIONS,
        max_picks: int = MAX_RECOMMENDATIONS,
        prompt_version: int = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self.loader = loader
        self.n_candidates = n_candidates
        self.min_picks = min_picks
        self.max_picks = max_picks
        self.prompt_version = prompt_version
        self._client = None

    # ----------------------------------------------------------------- #
    # Stage 1: geographic shortlist
    # ----------------------------------------------------------------- #
    def shortlist_by_distance(
        self, city: str, user_lat: float, user_lon: float
    ) -> pd.DataFrame:
        """Return the closest ``n_candidates`` restaurants in the city."""
        frame = self.loader.by_city(city)
        frame = frame.assign(
            distance_km=haversine_km_vectorized(
                frame["latitude"], frame["longitude"], user_lat, user_lon
            )
        )
        return frame.sort_values("distance_km").head(self.n_candidates).reset_index(
            drop=True
        )

    # ----------------------------------------------------------------- #
    # Stage 2: LLM rerank
    # ----------------------------------------------------------------- #
    def recommend(
        self,
        query: str,
        city: str,
        user_location: tuple[float, float] | None = None,
    ) -> RecommendationResult:
        """Run the full funnel and return a structured result."""
        user_lat, user_lon = user_location or CITY_COORDINATES[city]
        candidates = self.shortlist_by_distance(city, user_lat, user_lon)

        payload = candidates_to_payload(candidates)
        messages = prompts.build_messages(
            query=query,
            candidates=payload,
            min_picks=self.min_picks,
            max_picks=self.max_picks,
            version=self.prompt_version,
        )
        prompt_text = prompts.render_prompt_text(
            query=query,
            candidates=payload,
            min_picks=self.min_picks,
            max_picks=self.max_picks,
            version=self.prompt_version,
        )

        response = llm.complete_structured(
            client=self._get_client(),
            messages=messages,
            response_model=RecommendationResponse,
        )

        return RecommendationResult(
            query=query,
            city=city,
            prompt_version=self.prompt_version,
            prompt_text=prompt_text,
            candidates=candidates,
            response=response,
            user_location=(user_lat, user_lon),
            model=OPENAI_MODEL,
            extra={"n_candidates": len(candidates)},
        )

    # ----------------------------------------------------------------- #
    # Internal
    # ----------------------------------------------------------------- #
    def _get_client(self):
        if self._client is None:
            self._client = llm.build_client()
        return self._client
