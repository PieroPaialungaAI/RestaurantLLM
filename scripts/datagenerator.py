"""Synthetic restaurant dataset generation.

``RestaurantDataGenerator`` builds a reproducible table of ~10k restaurants
spread across a handful of cities. The generator is meant to run once; the
``Pipeline`` only calls it when the data file does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.constants import (
    CITY_COORDINATES,
    CITY_SPREAD_DEG,
    CUISINE_STYLES,
    DIETARY_OPTIONS,
    DIETARY_WEIGHTS,
    N_RESTAURANTS,
    PRICE_RANGES,
    PRICE_WEIGHTS,
    RANDOM_SEED,
    RESTAURANTS_PATH,
    SCORE_RANGE,
    VOTES_RANGE,
)
from scripts.utils import ensure_dir

_NAME_PREFIXES = [
    "The", "Casa", "Bella", "Golden", "Little", "Old", "Royal", "Blue",
    "Green", "Urban", "Rustic", "Maison", "El", "La", "Sakura", "Spice",
]
_NAME_CORES = [
    "Garden", "Table", "Kitchen", "Fork", "Spoon", "Olive", "Lantern",
    "Harbor", "Corner", "Market", "House", "Bistro", "Grill", "Terrace",
    "Brasserie", "Cantina", "Noodle", "Tavern", "Plate", "Hearth",
]
_NAME_SUFFIXES = ["", "", "& Co.", "Room", "No. 7", "Express", "Deluxe"]


class RestaurantDataGenerator:
    """Generates and persists the synthetic restaurant table."""

    def __init__(
        self,
        n_restaurants: int = N_RESTAURANTS,
        seed: int = RANDOM_SEED,
        output_path: Path = RESTAURANTS_PATH,
    ) -> None:
        self.n_restaurants = n_restaurants
        self.seed = seed
        self.output_path = output_path
        self._rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """Build the dataset in memory and return it as a DataFrame."""
        cities = list(CITY_COORDINATES.keys())
        city_choices = self._rng.choice(cities, size=self.n_restaurants)

        latitudes = np.empty(self.n_restaurants)
        longitudes = np.empty(self.n_restaurants)
        for i, city in enumerate(city_choices):
            center_lat, center_lon = CITY_COORDINATES[city]
            latitudes[i] = center_lat + self._rng.uniform(
                -CITY_SPREAD_DEG, CITY_SPREAD_DEG
            )
            longitudes[i] = center_lon + self._rng.uniform(
                -CITY_SPREAD_DEG, CITY_SPREAD_DEG
            )

        frame = pd.DataFrame(
            {
                "restaurant_id": np.arange(self.n_restaurants),
                "name": [self._random_name() for _ in range(self.n_restaurants)],
                "city": city_choices,
                "latitude": np.round(latitudes, 6),
                "longitude": np.round(longitudes, 6),
                "style": self._rng.choice(CUISINE_STYLES, size=self.n_restaurants),
                "dietary": self._rng.choice(
                    DIETARY_OPTIONS, size=self.n_restaurants, p=DIETARY_WEIGHTS
                ),
                "avg_score": self._random_scores(),
                "n_votes": self._random_votes(),
                "price_range": self._rng.choice(
                    PRICE_RANGES, size=self.n_restaurants, p=PRICE_WEIGHTS
                ),
            }
        )
        return frame

    def save(self, frame: pd.DataFrame | None = None) -> Path:
        """Persist the dataset to CSV, generating it first if needed."""
        if frame is None:
            frame = self.generate()
        ensure_dir(self.output_path.parent)
        frame.to_csv(self.output_path, index=False)
        return self.output_path

    # ----------------------------------------------------------------- #
    # Internal sampling helpers
    # ----------------------------------------------------------------- #
    def _random_name(self) -> str:
        prefix = self._rng.choice(_NAME_PREFIXES)
        core = self._rng.choice(_NAME_CORES)
        suffix = self._rng.choice(_NAME_SUFFIXES)
        return " ".join(part for part in (prefix, core, suffix) if part)

    def _random_scores(self) -> np.ndarray:
        # Beta skews scores towards the upper half, like real review platforms.
        low, high = SCORE_RANGE
        raw = self._rng.beta(a=5, b=2, size=self.n_restaurants)
        return np.round(low + raw * (high - low), 1)

    def _random_votes(self) -> np.ndarray:
        low, high = VOTES_RANGE
        # Log-uniform so most places have few votes and a handful are popular.
        raw = self._rng.uniform(np.log(low), np.log(high), size=self.n_restaurants)
        return np.exp(raw).astype(int)
