"""Loading and light querying of the restaurant table.

``RestaurantDataLoader`` is the only object that knows where the CSV lives and
how to slice it by city. It keeps the loaded frame in memory so repeated
queries are cheap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.constants import RESTAURANTS_PATH


class RestaurantDataLoader:
    """Loads the synthetic restaurant table and exposes simple lookups."""

    def __init__(self, path: Path = RESTAURANTS_PATH) -> None:
        self.path = path
        self._frame: pd.DataFrame | None = None

    @property
    def is_available(self) -> bool:
        """Whether the underlying data file exists on disk."""
        return self.path.exists()

    def load(self) -> pd.DataFrame:
        """Load the table from disk (cached after the first call)."""
        if self._frame is None:
            if not self.is_available:
                raise FileNotFoundError(
                    f"No restaurant data at {self.path}. "
                    "Generate it first with RestaurantDataGenerator."
                )
            self._frame = pd.read_csv(self.path)
        return self._frame

    def cities(self) -> list[str]:
        """Sorted list of cities present in the dataset."""
        return sorted(self.load()["city"].unique().tolist())

    def by_city(self, city: str) -> pd.DataFrame:
        """All restaurants located in a given city."""
        frame = self.load()
        mask = frame["city"].str.lower() == city.lower()
        subset = frame.loc[mask].copy()
        if subset.empty:
            raise ValueError(
                f"No restaurants found for city '{city}'. "
                f"Available cities: {self.cities()}"
            )
        return subset
