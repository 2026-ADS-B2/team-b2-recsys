"""Unit tests for the T8 data-quality report."""
import pandas as pd

from src.data.quality import build_quality_report


def test_quality_report_counts_rows_users_movies_and_missing_values():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "movie_id": [10, 11, 10],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 11],
            "title": ["Film A", "Film B"],
            "year": [2000, None],
            "genres": [["Drama"], ["Comedy"]],
        }
    )

    report = build_quality_report(ratings, movies)
    rating_metrics = report["processed"]["ratings"]
    movie_metrics = report["processed"]["movies"]

    assert rating_metrics["rows"] == 3
    assert rating_metrics["users"] == 2
    assert rating_metrics["movies"] == 2
    assert rating_metrics["missing_values"]["rating"] == 0
    assert movie_metrics["rows"] == 2
    assert movie_metrics["missing_values"]["year"] == 1


def test_quality_report_detects_duplicates_and_rating_distribution():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "movie_id": [10, 10, 11],
            "rating": [4.0, 4.0, 5.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 10, 11],
            "title": ["Film A", "Film A", "Film B"],
            "year": [2000, 2000, 2001],
            "genres": [["Drama"], ["Drama"], ["Comedy"]],
        }
    )

    report = build_quality_report(ratings, movies)
    rating_metrics = report["processed"]["ratings"]
    movie_metrics = report["processed"]["movies"]

    assert rating_metrics["duplicate_rows"] == 1
    assert rating_metrics["duplicate_user_movie_pairs"] == 1
    assert rating_metrics["rating_distribution"] == {"4.0": 2, "5.0": 1}
    assert movie_metrics["duplicate_rows"] == 1
    assert movie_metrics["duplicate_movie_ids"] == 1
