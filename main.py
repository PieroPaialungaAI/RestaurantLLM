"""Command-line app for the restaurant recommendation funnel.

Pick the city you are in, type your preferences in plain English, and the
pipeline returns the LLM's final 5-10 picks.

Usage:
    python main.py                       # interactive
    python main.py --city "Austin" --query "cheap vegan tacos, lively spot"
"""

from __future__ import annotations

import argparse

from scripts.pipeline import Pipeline


def _choose_city(pipeline: Pipeline) -> str:
    cities = pipeline.available_cities()
    print("\nWhere are you? Available cities:")
    for i, city in enumerate(cities, start=1):
        print(f"  {i}. {city}")
    while True:
        raw = input("\nPick a city (number or name): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(cities):
            return cities[int(raw) - 1]
        match = [c for c in cities if c.lower() == raw.lower()]
        if match:
            return match[0]
        print("  Sorry, I didn't recognize that. Try again.")


def _print_result(payload: dict) -> None:
    result = payload["result"]
    response = result.response
    print("\n" + "=" * 70)
    print(f"Top picks in {result.city} (prompt v{result.prompt_version}, "
          f"{result.model}):")
    print("=" * 70)
    for rank, pick in enumerate(response.picks, start=1):
        print(f"\n{rank}. {pick.name}  [fit {pick.fit_score:.0f}/100]")
        print(f"   {pick.reason}")
    print("\n" + "-" * 70)
    print(f"Summary: {response.summary}")
    if payload["saved_path"] is not None:
        print(f"\nSaved run to: {payload['saved_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restaurant recommendation funnel")
    parser.add_argument("--city", help="City you are in.")
    parser.add_argument("--query", help="Free-text description of what you want.")
    parser.add_argument(
        "--prompt-version", type=int, default=0, help="Prompt version to use."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not persist the run."
    )
    args = parser.parse_args()

    pipeline = Pipeline(prompt_version=args.prompt_version)

    city = args.city or _choose_city(pipeline)
    query = args.query or input(
        "\nWhat are you in the mood for? Describe it freely:\n> "
    ).strip()

    payload = pipeline.run(query=query, city=city, save=not args.no_save)
    _print_result(payload)


if __name__ == "__main__":
    main()
