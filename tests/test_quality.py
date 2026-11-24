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

def test_ratings_parquet_has_expected_columns(tmp_path):
    expected_columns = {"user_id", "movie_id", "rating", "timestamp"}

    ratings = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [10, 20],
        "rating": [4.0, 3.5],
        "timestamp": [1000, 2000],
    })

    parquet_path = tmp_path / "ratings.parquet"
    ratings.to_parquet(parquet_path, index=False)

    loaded_ratings = pd.read_parquet(parquet_path)

    assert expected_columns.issubset(set(loaded_ratings.columns))



# --- Sprint 2 T2: named validation checks ----------------------------------------------

import json

from src.data.quality import run_checks

LOW_THRESHOLDS = {"min_ratings_per_user": 2, "min_ratings_per_movie": 1}


def good_data():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 11, 10, 11],
            "rating": [4.0, 3.5, 5.0, 0.5],
            "timestamp": pd.to_datetime(["2000-01-01", "2000-01-02", "2001-01-01", "2001-01-02"]),
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 11, 12],
            "title": ["Film A", "Film B", "Film C"],
            "year": [2000, None, 1999],
            "genres": [["Drama"], ["Comedy"], []],
        }
    )
    return ratings, movies


def test_clean_data_passes_every_check_and_report_is_json():
    ratings, movies = good_data()

    result = run_checks(ratings, movies, LOW_THRESHOLDS)

    assert result["summary"]["all_passed"], result["summary"]["failed_checks"]
    assert result["summary"]["total"] == 9
    for check in result["checks"].values():
        assert set(check) == {"passed", "details"}
    json.dumps(result)  # the report must be writable as JSON


def test_each_problem_fails_its_own_check_with_details():
    ratings, movies = good_data()
    bad = pd.concat(
        [
            ratings,
            pd.DataFrame(
                {
                    "user_id": [1, 3],
                    "movie_id": [10, 99],                       # duplicate pair, unknown movie
                    "rating": [4.2, 6.0],                       # not a half step, out of range
                    "timestamp": pd.to_datetime(["1990-01-01", "2002-01-01"]),  # too early
                }
            ),
        ],
        ignore_index=True,
    )

    result = run_checks(bad, movies, LOW_THRESHOLDS)
    checks = result["checks"]

    assert not checks["valid_half_step_ratings"]["passed"]
    assert checks["valid_half_step_ratings"]["details"]["invalid_count"] == 2
    assert not checks["known_movie_ids"]["passed"]
    assert checks["known_movie_ids"]["details"]["unknown_movie_ids"] == [99]
    assert not checks["unique_user_movie_ratings"]["passed"]
    assert checks["unique_user_movie_ratings"]["details"]["examples"] == [(1, 10)]
    assert not checks["valid_timestamps"]["passed"]
    assert checks["valid_timestamps"]["details"]["before_earliest"] == 1
    assert not checks["min_ratings_per_user"]["passed"]          # user 3 has 1 rating
    assert set(result["summary"]["failed_checks"]) >= {
        "valid_half_step_ratings", "known_movie_ids", "unique_user_movie_ratings",
        "valid_timestamps", "min_ratings_per_user",
    }


def test_missing_columns_and_nulls_are_reported_without_crashing():
    ratings, movies = good_data()
    ratings = ratings.drop(columns=["timestamp"])
    movies.loc[0, "title"] = None

    result = run_checks(ratings, movies, LOW_THRESHOLDS)

    assert result["checks"]["required_columns"]["details"]["missing_in_ratings"] == ["timestamp"]
    assert result["checks"]["no_nulls"]["details"]["movies_nulls"]["title"] == 1
    assert not result["checks"]["valid_timestamps"]["passed"]


def test_empty_data_fails_nonempty_check():
    ratings, movies = good_data()

    result = run_checks(ratings.iloc[0:0], movies, LOW_THRESHOLDS)

    assert not result["checks"]["nonempty_data"]["passed"]
    assert result["checks"]["nonempty_data"]["details"]["ratings_rows"] == 0


def test_missing_year_is_allowed():
    ratings, movies = good_data()

    result = run_checks(ratings, movies, LOW_THRESHOLDS)

    assert result["checks"]["no_nulls"]["passed"]
    assert result["checks"]["no_nulls"]["details"]["movies_missing_year"] == 1    