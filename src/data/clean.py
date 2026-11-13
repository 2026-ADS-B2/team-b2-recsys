"""Clean MovieLens raw CSV files and save them as parquet (T3)."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"movieId": "movie_id"})
    # "Toy Story (1995)" -> year 1995, title "Toy Story"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    # "Adventure|Comedy" -> ["Adventure", "Comedy"]
    df["genres"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    df = df.dropna(subset=["movie_id"]).drop_duplicates(subset="movie_id")
    return df[["movie_id", "title", "year", "genres"]]


def clean_ratings(df: pd.DataFrame, movie_ids: set) -> pd.DataFrame:
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df = df[df["rating"].between(0.5, 5.0)]
    df = df[df["movie_id"].isin(movie_ids)]
    # if a user rated the same movie twice, keep the latest rating
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "movie_id"], keep="last")
    return df[["user_id", "movie_id", "rating", "timestamp"]].reset_index(drop=True)


def check_schema(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert list(movies.columns) == ["movie_id", "title", "year", "genres"]
    assert ratings["rating"].between(0.5, 5.0).all()
    assert not ratings.duplicated(["user_id", "movie_id"]).any()
    assert movies["movie_id"].is_unique


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    movies = clean_movies(pd.read_csv(RAW / "movies.csv"))
    ratings = clean_ratings(pd.read_csv(RAW / "ratings.csv"), set(movies["movie_id"]))
    check_schema(ratings, movies)

    movies.to_parquet(OUT / "movies.parquet", index=False)
    ratings.to_parquet(OUT / "ratings.parquet", index=False)
    print(f"movies:  {len(movies)} rows -> {OUT / 'movies.parquet'}")
    print(f"ratings: {len(ratings)} rows, {ratings['user_id'].nunique()} users -> {OUT / 'ratings.parquet'}")


if __name__ == "__main__":
    main()
