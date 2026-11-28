"""Unit tests for the Sprint 2 T7 movie catalog."""
import pandas as pd

from src.data.catalog import PLACEHOLDER_IMAGE, build_catalog


def make_inputs():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Toy Story", "Jumanji", "Unrated Movie"],
        "year": [1995, 1995, 2000],
        "genres": [["Animation", "Comedy"], ["Adventure"], []],
    })
    ratings = pd.DataFrame({
        "movie_id": [1, 2, 2, 2],
        "rating": [5.0, 3.0, 4.0, 5.0],
    })
    links = pd.DataFrame({
        "movieId": [1, 2],          # movie 3 has no link
        "imdbId": [114709, 113497],
        "tmdbId": [862.0, None],
    })
    return movies, ratings, links


def test_catalog_is_in_popularity_order():
    catalog = build_catalog(*make_inputs())

    # Jumanji has 3 ratings, Toy Story 1, the unrated movie 0 -> last
    assert list(catalog["title"]) == ["Jumanji", "Toy Story", "Unrated Movie"]
    assert list(catalog["popularity_rank"]) == [1, 2, 3]
    assert list(catalog["n_ratings"]) == [3, 1, 0]


def test_imdb_links_and_missing_links():
    catalog = build_catalog(*make_inputs()).set_index("movie_id")

    assert catalog.loc[1, "imdb_url"] == "https://www.imdb.com/title/tt0114709/"
    assert catalog.loc[1, "imdb_id"] == "tt0114709"
    assert pd.isna(catalog.loc[3, "imdb_url"])
    assert pd.isna(catalog.loc[2, "tmdb_url"])


def test_every_movie_has_the_fields_for_a_card():
    catalog = build_catalog(*make_inputs())

    for column in ["movie_id", "title", "genres", "popularity_rank", "image_url"]:
        assert catalog[column].notna().all()
    assert (catalog["image_url"] == PLACEHOLDER_IMAGE).all()
    assert catalog["movie_id"].is_unique
