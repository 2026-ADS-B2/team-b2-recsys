"""Unit tests for the T3 cleaning stage (T18)."""
import pandas as pd

from src.data.clean import clean_movies, clean_ratings


def test_clean_movies_extracts_year_and_strips_title():
    raw = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story (1995)", "Movie Without Year"],
        "genres": ["Comedy", "Drama"],
    })

    movies = clean_movies(raw)

    assert list(movies.columns) == ["movie_id", "title", "year", "genres"]
    assert movies.loc[0, "title"] == "Toy Story"
    assert movies.loc[0, "year"] == 1995
    assert movies.loc[1, "title"] == "Movie Without Year"
    assert pd.isna(movies.loc[1, "year"])


def test_clean_movies_turns_genres_into_a_list():
    raw = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["A (2000)", "B (2001)"],
        "genres": ["Adventure|Comedy", "(no genres listed)"],
    })

    movies = clean_movies(raw)

    assert movies.loc[0, "genres"] == ["Adventure", "Comedy"]
    assert movies.loc[1, "genres"] == []


def test_clean_movies_removes_duplicate_movie_ids():
    raw = pd.DataFrame({
        "movieId": [1, 1, 2],
        "title": ["A (2000)", "A (2000)", "B (2001)"],
        "genres": ["Drama", "Drama", "Comedy"],
    })

    movies = clean_movies(raw)

    assert len(movies) == 2
    assert movies["movie_id"].is_unique


def test_clean_ratings_keeps_latest_duplicate_and_drops_invalid_rows():
    raw = pd.DataFrame({
        "userId":    [1,   1,   1,   1,   2],
        "movieId":   [10,  10,  11,  99,  10],
        "rating":    [3.0, 5.0, 7.0, 4.0, 4.0],
        "timestamp": [100, 200, 300, 400, 500],   # unix seconds
    })

    ratings = clean_ratings(raw, movie_ids={10, 11})

    # user 1 rated movie 10 twice -> only the latest rating (5.0) is kept
    user1_movie10 = ratings[(ratings["user_id"] == 1) & (ratings["movie_id"] == 10)]
    assert list(user1_movie10["rating"]) == [5.0]
    # rating 7.0 is out of range and movie 99 is unknown -> both removed
    assert 7.0 not in set(ratings["rating"])
    assert 99 not in set(ratings["movie_id"])
    assert len(ratings) == 2
    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
