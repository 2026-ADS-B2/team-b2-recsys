"""Generate a reproducible data-quality report for the MovieLens dataset (T8)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORT = Path("reports/quality.json")


def _column_missing(df: pd.DataFrame) -> dict[str, int]:
    """Return the number of missing values in each column."""
    return {str(column): int(count) for column, count in df.isna().sum().items()}



def _duplicate_count(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Count duplicated rows, including list and NumPy array values."""
    frame = df[subset].copy() if subset is not None else df.copy()

    def make_hashable(value: Any) -> Any:
        # Convert NumPy arrays and similar objects to Python values.
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            converted = value.tolist()
            if converted is not value:
                return make_hashable(converted)

        if isinstance(value, (list, tuple, set)):
            return tuple(make_hashable(item) for item in value)

        if isinstance(value, dict):
            return tuple(
                sorted((key, make_hashable(item)) for key, item in value.items())
            )

        return value

    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].map(make_hashable)

    return int(frame.duplicated().sum())



def _rating_distribution(ratings: pd.DataFrame) -> dict[str, int]:
    """Count ratings by score, with stable, JSON-friendly string keys."""
    if "rating" not in ratings.columns:
        return {}
    counts = ratings["rating"].value_counts(dropna=False).sort_index()
    return {str(score): int(count) for score, count in counts.items()}


def build_quality_report(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    raw_ratings: pd.DataFrame | None = None,
    raw_movies: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build quality metrics from processed data and, when supplied, raw CSV data."""
    report: dict[str, Any] = {
        "processed": {
            "ratings": {
                "rows": int(len(ratings)),
                "users": int(ratings["user_id"].nunique()) if "user_id" in ratings else 0,
                "movies": int(ratings["movie_id"].nunique()) if "movie_id" in ratings else 0,
                "missing_values": _column_missing(ratings),
                "duplicate_rows": _duplicate_count(ratings),
                "duplicate_user_movie_pairs": (
                    _duplicate_count(ratings, ["user_id", "movie_id"])
                    if {"user_id", "movie_id"}.issubset(ratings.columns)
                    else 0
                ),
                "rating_distribution": _rating_distribution(ratings),
            },
            "movies": {
                "rows": int(len(movies)),
                "unique_movies": int(movies["movie_id"].nunique()) if "movie_id" in movies else 0,
                "missing_values": _column_missing(movies),
                "duplicate_rows": _duplicate_count(movies),
                "duplicate_movie_ids": (
                    int(movies["movie_id"].duplicated().sum()) if "movie_id" in movies else 0
                ),
            },
        }
    }

    if raw_ratings is not None:
        report["raw"] = report.get("raw", {})
        report["raw"]["ratings"] = {
            "rows": int(len(raw_ratings)),
            "users": int(raw_ratings["userId"].nunique()) if "userId" in raw_ratings else 0,
            "movies": int(raw_ratings["movieId"].nunique()) if "movieId" in raw_ratings else 0,
            "missing_values": _column_missing(raw_ratings),
            "duplicate_rows": _duplicate_count(raw_ratings),
            "rating_distribution": _rating_distribution(
                raw_ratings.rename(columns={"rating": "rating"})
            ),
        }
    if raw_movies is not None:
        report["raw"] = report.get("raw", {})
        report["raw"]["movies"] = {
            "rows": int(len(raw_movies)),
            "unique_movies": int(raw_movies["movieId"].nunique()) if "movieId" in raw_movies else 0,
            "missing_values": _column_missing(raw_movies),
            "duplicate_rows": _duplicate_count(raw_movies),
        }

    return report


def main() -> None:
    ratings_path = PROCESSED / "ratings.parquet"
    movies_path = PROCESSED / "movies.parquet"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `dvc repro clean` before the quality stage."
        )

    ratings = pd.read_parquet(ratings_path)
    movies = pd.read_parquet(movies_path)
    raw_ratings_path = RAW / "ratings.csv"
    raw_movies_path = RAW / "movies.csv"
    raw_ratings = pd.read_csv(raw_ratings_path) if raw_ratings_path.exists() else None
    raw_movies = pd.read_csv(raw_movies_path) if raw_movies_path.exists() else None

    report = build_quality_report(ratings, movies, raw_ratings, raw_movies)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Quality report written to {REPORT}")


if __name__ == "__main__":
    main()
