"""Build the movie catalog for browsing in the app (Sprint 2, T7).

One row per movie with everything the app needs to render a movie card:
title, year, genres, popularity (number of ratings and mean rating),
popularity rank, IMDb / TMDB links and an image URL (placeholder for now,
MovieLens has no posters).

Popularity order: most rated movies first; ties broken by mean rating, then movie_id.
Movies without ratings are kept at the end of the catalog.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
CATALOG_PATH = PROCESSED / "catalog.parquet"

# MovieLens has no posters: every card gets the same placeholder image (easy to swap later)
PLACEHOLDER_IMAGE = "https://placehold.co/300x450?text=No+poster"

CATALOG_COLUMNS = [
    "movie_id", "title", "year", "genres",
    "n_ratings", "mean_rating", "popularity_rank",
    "imdb_id", "imdb_url", "tmdb_url", "image_url",
]


def imdb_url(imdb_id) -> Optional[str]:
    """0114709 -> https://www.imdb.com/title/tt0114709/ (None if the id is missing)."""
    if pd.isna(imdb_id):
        return None
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"


def tmdb_url(tmdb_id) -> Optional[str]:
    if pd.isna(tmdb_id):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def build_catalog(movies: pd.DataFrame, ratings: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """movies: movie_id, title, year, genres; ratings: movie_id, rating; links: movieId, imdbId, tmdbId."""
    stats = ratings.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean").reset_index()
    links = links.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"})

    catalog = (
        movies.merge(stats, on="movie_id", how="left")
              .merge(links[["movie_id", "imdb_id", "tmdb_id"]], on="movie_id", how="left")
    )
    catalog["n_ratings"] = catalog["n_ratings"].fillna(0).astype(int)
    catalog["mean_rating"] = catalog["mean_rating"].round(2)

    catalog = catalog.sort_values(
        ["n_ratings", "mean_rating", "movie_id"], ascending=[False, False, True], na_position="last"
    ).reset_index(drop=True)
    catalog["popularity_rank"] = range(1, len(catalog) + 1)

    catalog["imdb_url"] = catalog["imdb_id"].map(imdb_url)
    catalog["tmdb_url"] = catalog["tmdb_id"].map(tmdb_url)
    catalog["imdb_id"] = catalog["imdb_id"].map(lambda x: None if pd.isna(x) else f"tt{int(x):07d}")
    catalog["image_url"] = PLACEHOLDER_IMAGE
    return catalog[CATALOG_COLUMNS]


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """For the app/API: the catalog in popularity order (rank 1 = most popular)."""
    return pd.read_parquet(path).sort_values("popularity_rank").reset_index(drop=True)


def main() -> None:
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    links = pd.read_csv(RAW / "links.csv")

    catalog = build_catalog(movies, ratings, links)
    assert catalog["movie_id"].is_unique
    catalog.to_parquet(CATALOG_PATH, index=False)

    no_imdb = catalog["imdb_url"].isna().sum()
    print(f"catalog: {len(catalog)} movies -> {CATALOG_PATH} ({no_imdb} without IMDb link)")
    print(catalog.head(5)[["popularity_rank", "title", "n_ratings", "mean_rating"]].to_string(index=False))


if __name__ == "__main__":
    main()
