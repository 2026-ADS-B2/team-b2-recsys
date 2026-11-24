"""Generate a reproducible data-quality report for the MovieLens dataset.

T8: dataset metrics (rows, missing values, duplicates, rating distribution).
Sprint 2 T2: named validation checks, each with a result and details, in report["checks"].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORT = Path("reports/quality.json")

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "year", "genres"]
# year may be missing in MovieLens titles ("Movie without year"), so it is not required to be non-null
NOT_NULL_MOVIE_COLUMNS = ["movie_id", "title", "genres"]

DEFAULT_THRESHOLDS = {
    "min_ratings_per_user": 20,
    "min_ratings_per_movie": 1,
    "rating_min": 0.5,
    "rating_max": 5.0,
    "earliest_timestamp": "1995-01-01",
}


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


# --- Sprint 2 T2: named validation checks ------------------------------------------------

def _check(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "details": details}


def _examples(values: Any, limit: int = 5) -> list:
    return [v.item() if hasattr(v, "item") else v for v in list(values)[:limit]]


def check_nonempty_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    return _check(
        len(ratings) > 0 and len(movies) > 0,
        ratings_rows=int(len(ratings)),
        movies_rows=int(len(movies)),
    )


def check_required_columns(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    missing_ratings = [c for c in RATING_COLUMNS if c not in ratings.columns]
    missing_movies = [c for c in MOVIE_COLUMNS if c not in movies.columns]
    return _check(
        not missing_ratings and not missing_movies,
        missing_in_ratings=missing_ratings,
        missing_in_movies=missing_movies,
    )


def check_no_nulls(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    ratings_nulls = {c: int(ratings[c].isna().sum()) for c in RATING_COLUMNS if c in ratings}
    movies_nulls = {c: int(movies[c].isna().sum()) for c in NOT_NULL_MOVIE_COLUMNS if c in movies}
    return _check(
        sum(ratings_nulls.values()) == 0 and sum(movies_nulls.values()) == 0,
        ratings_nulls=ratings_nulls,
        movies_nulls=movies_nulls,
        movies_missing_year=int(movies["year"].isna().sum()) if "year" in movies else None,
        note="year may be missing and is reported for information only",
    )


def check_valid_half_step_ratings(ratings: pd.DataFrame, t: dict[str, Any]) -> dict[str, Any]:
    if "rating" not in ratings:
        return _check(False, error="column 'rating' is missing")
    r = pd.to_numeric(ratings["rating"], errors="coerce")
    in_range = r.between(t["rating_min"], t["rating_max"])
    half_step = (r * 2).round(9) % 1 == 0
    invalid = r[~(in_range & half_step)]
    return _check(
        len(invalid) == 0,
        allowed=f"{t['rating_min']} to {t['rating_max']} in steps of 0.5",
        invalid_count=int(len(invalid)),
        invalid_examples=_examples(invalid.drop_duplicates()),
    )


def check_known_movie_ids(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    if "movie_id" not in ratings or "movie_id" not in movies:
        return _check(False, error="column 'movie_id' is missing")
    unknown = ratings.loc[~ratings["movie_id"].isin(movies["movie_id"]), "movie_id"]
    return _check(
        unknown.empty,
        unknown_rating_rows=int(len(unknown)),
        unknown_movie_ids=_examples(unknown.drop_duplicates()),
    )


def check_unique_user_movie_ratings(ratings: pd.DataFrame) -> dict[str, Any]:
    if not {"user_id", "movie_id"}.issubset(ratings.columns):
        return _check(False, error="columns 'user_id' and 'movie_id' are required")
    dup = ratings[ratings.duplicated(["user_id", "movie_id"], keep=False)]
    pairs = dup[["user_id", "movie_id"]].drop_duplicates()
    return _check(
        pairs.empty,
        duplicated_pairs=int(len(pairs)),
        examples=[tuple(int(x) for x in row) for row in pairs.head(5).itertuples(index=False)],
    )


def check_valid_timestamps(ratings: pd.DataFrame, t: dict[str, Any]) -> dict[str, Any]:
    if "timestamp" not in ratings:
        return _check(False, error="column 'timestamp' is missing")
    ts = pd.to_datetime(ratings["timestamp"], errors="coerce")
    earliest = pd.Timestamp(t["earliest_timestamp"])
    now = pd.Timestamp.now()
    unparseable = int(ts.isna().sum())
    too_early = int((ts < earliest).sum())
    in_future = int((ts > now).sum())
    return _check(
        unparseable == 0 and too_early == 0 and in_future == 0,
        allowed=f"from {earliest.date()} until today",
        first=str(ts.min()) if ts.notna().any() else None,
        last=str(ts.max()) if ts.notna().any() else None,
        unparseable=unparseable,
        before_earliest=too_early,
        in_future=in_future,
    )


def check_min_ratings_per_user(ratings: pd.DataFrame, t: dict[str, Any]) -> dict[str, Any]:
    if "user_id" not in ratings:
        return _check(False, error="column 'user_id' is missing")
    counts = ratings.groupby("user_id").size()
    below = counts[counts < t["min_ratings_per_user"]]
    return _check(
        below.empty and not counts.empty,
        threshold=int(t["min_ratings_per_user"]),
        users=int(len(counts)),
        users_below_threshold=int(len(below)),
        min_observed=int(counts.min()) if len(counts) else 0,
        examples=_examples(below.index),
    )


def check_min_ratings_per_movie(
    ratings: pd.DataFrame, movies: pd.DataFrame, t: dict[str, Any]
) -> dict[str, Any]:
    if "movie_id" not in ratings:
        return _check(False, error="column 'movie_id' is missing")
    counts = ratings.groupby("movie_id").size()
    below = counts[counts < t["min_ratings_per_movie"]]
    unrated = (
        int((~movies["movie_id"].isin(counts.index)).sum()) if "movie_id" in movies else None
    )
    return _check(
        below.empty and not counts.empty,
        threshold=int(t["min_ratings_per_movie"]),
        rated_movies=int(len(counts)),
        movies_below_threshold=int(len(below)),
        min_observed=int(counts.min()) if len(counts) else 0,
        catalog_movies_without_ratings=unrated,
        note="checked over rated movies; catalog movies without ratings are reported for information",
    )


def run_checks(
    ratings: pd.DataFrame, movies: pd.DataFrame, thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run every named check and return {"checks": {...}, "summary": {...}}."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks = {
        "nonempty_data": check_nonempty_data(ratings, movies),
        "required_columns": check_required_columns(ratings, movies),
        "no_nulls": check_no_nulls(ratings, movies),
        "valid_half_step_ratings": check_valid_half_step_ratings(ratings, t),
        "known_movie_ids": check_known_movie_ids(ratings, movies),
        "unique_user_movie_ratings": check_unique_user_movie_ratings(ratings),
        "valid_timestamps": check_valid_timestamps(ratings, t),
        "min_ratings_per_user": check_min_ratings_per_user(ratings, t),
        "min_ratings_per_movie": check_min_ratings_per_movie(ratings, movies, t),
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    summary = {
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_checks": failed,
        "all_passed": not failed,
    }
    return {"checks": checks, "summary": summary}


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
    params = yaml.safe_load(open("params.yaml")) if Path("params.yaml").exists() else {}
    validation = run_checks(ratings, movies, (params or {}).get("quality"))
    report["checks"] = validation["checks"]
    report["summary"] = validation["summary"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"Quality report written to {REPORT}: {summary['passed']}/{summary['total']} checks passed")
    for name in summary["failed_checks"]:
        print(f"  FAILED {name}: {report['checks'][name]['details']}")


if __name__ == "__main__":
    main()
