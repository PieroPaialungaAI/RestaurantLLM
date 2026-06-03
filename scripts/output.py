"""Persisting and reloading recommendation runs.

``OutputGenerator`` remembers, for every run, three things the article cares
about: the (versioned) prompt, the user query and the model output. Each run is
written to its own timestamped JSON file so prompt iterations stay auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.constants import OUTPUT_DIR
from scripts.recommendation import RecommendationResult
from scripts.utils import ensure_dir, read_json, slugify, utc_timestamp, write_json


class OutputGenerator:
    """Serializes a ``RecommendationResult`` into a versioned record on disk."""

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        ensure_dir(self.output_dir)

    def to_record(self, result: RecommendationResult) -> dict[str, Any]:
        """Build the JSON-serializable record for a run."""
        return {
            "timestamp": utc_timestamp(),
            "city": result.city,
            "user_location": list(result.user_location),
            "model": result.model,
            "prompt_version": result.prompt_version,
            "query": result.query,
            "prompt": result.prompt_text,
            "n_candidates": int(result.extra.get("n_candidates", len(result.candidates))),
            "candidates": result.candidates.to_dict(orient="records"),
            "output": result.response.model_dump(),
        }

    def save(self, result: RecommendationResult) -> Path:
        """Write the run to ``outputs/`` and return the file path."""
        record = self.to_record(result)
        filename = (
            f"{record['timestamp']}__v{result.prompt_version}__"
            f"{slugify(result.query)}.json"
        )
        path = self.output_dir / filename
        write_json(path, record)
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Reload a previously saved run."""
        return read_json(path)

    def list_runs(self) -> list[Path]:
        """All saved runs, newest first."""
        return sorted(self.output_dir.glob("*.json"), reverse=True)
