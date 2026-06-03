"""End-to-end orchestration.

``Pipeline`` is the single entry point the app talks to. It owns one instance
of every other object and wires the four phases together:

1. data generation (only if the dataset is missing),
2. data loading,
3. recommendation (distance + LLM funnel),
4. output saving / loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.constants import DEFAULT_PROMPT_VERSION
from scripts.datagenerator import RestaurantDataGenerator
from scripts.dataloader import RestaurantDataLoader
from scripts.output import OutputGenerator
from scripts.recommendation import RecommendationResult, RestaurantRecommender


class Pipeline:
    """Owns and coordinates every stage of the recommendation system."""

    def __init__(self, prompt_version: int = DEFAULT_PROMPT_VERSION) -> None:
        self.generator = RestaurantDataGenerator()
        self.loader = RestaurantDataLoader()
        self.recommender = RestaurantRecommender(
            loader=self.loader, prompt_version=prompt_version
        )
        self.output = OutputGenerator()

    # ----------------------------------------------------------------- #
    # Phase 1 + 2: data
    # ----------------------------------------------------------------- #
    def ensure_data(self, force: bool = False) -> Path:
        """Generate the dataset if it does not exist yet (or if forced)."""
        if force or not self.loader.is_available:
            return self.generator.save()
        return self.loader.path

    def available_cities(self) -> list[str]:
        """Cities the user can pick from."""
        self.ensure_data()
        return self.loader.cities()

    # ----------------------------------------------------------------- #
    # Phase 3 + 4: recommend and persist
    # ----------------------------------------------------------------- #
    def run(
        self,
        query: str,
        city: str,
        user_location: tuple[float, float] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Run the whole funnel for one request and persist the result."""
        self.ensure_data()
        result: RecommendationResult = self.recommender.recommend(
            query=query, city=city, user_location=user_location
        )
        saved_path = self.output.save(result) if save else None
        return {"result": result, "saved_path": saved_path}

    # ----------------------------------------------------------------- #
    # Output reloading
    # ----------------------------------------------------------------- #
    def load_run(self, path: Path) -> dict[str, Any]:
        return self.output.load(path)

    def list_runs(self) -> list[Path]:
        return self.output.list_runs()
