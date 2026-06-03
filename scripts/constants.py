"""Central place for every constant used across the project.

Nothing in here should contain logic. If you find yourself writing an `if`
statement in this file, it probably belongs in ``utils.py`` instead.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESTAURANTS_PATH = DATA_DIR / "restaurants.csv"

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
N_RESTAURANTS = 10_000

# Each city is the "anchor" location the user can be standing in. Restaurants
# are scattered around the city center within ``CITY_SPREAD_DEG`` degrees.
CITY_COORDINATES = {
    "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194),
    "Chicago": (41.8781, -87.6298),
    "Austin": (30.2672, -97.7431),
    "Seattle": (47.6062, -122.3321),
    "Boston": (42.3601, -71.0589),
    "Miami": (25.7617, -80.1918),
    "Denver": (39.7392, -104.9903),
}
CITY_SPREAD_DEG = 0.12  # ~13 km radius around the city center

CUISINE_STYLES = [
    "Italian",
    "Japanese",
    "Mexican",
    "Indian",
    "Thai",
    "French",
    "Chinese",
    "Mediterranean",
    "American",
    "Korean",
    "Vietnamese",
    "Greek",
    "Spanish",
    "Ethiopian",
    "Lebanese",
]

# Dietary profile of the restaurant's menu.
DIETARY_OPTIONS = ["omnivore", "vegetarian", "vegan"]
DIETARY_WEIGHTS = [0.7, 0.2, 0.1]

# Price range is encoded as an order-of-magnitude average ticket per person.
PRICE_RANGES = [10, 100, 1000]
PRICE_WEIGHTS = [0.55, 0.38, 0.07]

SCORE_RANGE = (1.0, 5.0)
VOTES_RANGE = (5, 5000)

# --------------------------------------------------------------------------- #
# Recommendation funnel
# --------------------------------------------------------------------------- #
# Stage 1 (high recall): keep the geographically closest candidates.
N_DISTANCE_CANDIDATES = 50
# Stage 2 (high precision): how many the LLM is allowed to return.
MIN_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 10

# --------------------------------------------------------------------------- #
# OpenAI / LLM
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = 0.2

# Prompt version that the pipeline ships with by default.
DEFAULT_PROMPT_VERSION = 0
