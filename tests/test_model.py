"""Unit tests for the T5 popularity model and the T6 precision@k metric (T18)."""
import pandas as pd

from src.models.evaluate import precision_at_k
from src.models.train import PopularityModel, build_ranking


def make_ranking(n_movies=15):
    return pd.DataFrame({
        "movie_id": range(1, n_movies + 1),
        "title": [f"Movie {i}" for i in range(1, n_movies + 1)],
        "score": [1 - i / 100 for i in range(n_movies)],
    })


def test_predict_returns_exactly_n_movies_per_user():
    model = PopularityModel(make_ranking(), default_n=10)
    request = pd.DataFrame({"user_id": [1, 2], "n": [3, 7]})

    recs = model.predict(None, request)

    assert len(recs[recs["user_id"] == 1]) == 3
    assert len(recs[recs["user_id"] == 2]) == 7
    assert list(recs[recs["user_id"] == 1]["rank"]) == [1, 2, 3]


def test_predict_uses_default_n_when_n_is_missing():
    model = PopularityModel(make_ranking(), default_n=10)

    recs = model.predict(None, pd.DataFrame({"user_id": [5]}))

    assert len(recs) == 10


def test_build_ranking_filters_by_min_ratings_and_sorts_by_mean_rating():
    train = pd.DataFrame({
        "user_id":  [1, 2, 3, 1, 2, 3, 1],
        "movie_id": [10, 10, 10, 20, 20, 20, 30],
        "rating":   [3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 5.0],
    })
    movies = pd.DataFrame({"movie_id": [10, 20, 30], "title": ["A", "B", "C"]})

    ranking = build_ranking(train, movies, min_ratings=2)

    # movie 30 has only 1 rating -> filtered out; movie 20 (mean 4.67) before movie 10 (mean 3.0)
    assert list(ranking["movie_id"]) == [20, 10]
    assert ranking.loc[0, "title"] == "B"


def test_precision_at_k_on_a_known_example():
    recs = pd.DataFrame({
        "user_id":  [1, 1, 2, 2],
        "movie_id": [10, 11, 10, 11],
    })
    test = pd.DataFrame({
        "user_id":  [1, 1, 2],
        "movie_id": [10, 12, 13],
    })

    # user 1: 1 of 2 recommended movies is in test -> 0.5
    # user 2: 0 of 2 -> 0.0;  average = 0.25
    assert precision_at_k(recs, test, k=2) == 0.25
