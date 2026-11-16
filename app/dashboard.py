
"""Streamlit dashboard for MovieLens data quality and basic dataset insights (T9)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT_PATH = Path("reports/quality.json")
MOVIES_PATH = Path("data/processed/movies.parquet")
RATINGS_PATH = Path("data/processed/ratings.parquet")


st.set_page_config(
    page_title="MovieLens · Data Dashboard",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 MovieLens — Data Dashboard")
st.caption("Dataset overview, rating distribution and data quality checks")

if not REPORT_PATH.exists():
    st.warning(
        "reports/quality.json was not found. "
        "Run `dvc repro quality` from the project root first."
    )
    st.stop()

with REPORT_PATH.open(encoding="utf-8") as file:
    quality = json.load(file)

processed = quality.get("processed", {})
rating_info = processed.get("ratings", {})
movie_info = processed.get("movies", {})

c1, c2, c3, c4 = st.columns(4)

c1.metric("Ratings", f"{rating_info.get('rows', 0):,}")
c2.metric("Users", f"{rating_info.get('users', 0):,}")
c3.metric(
    "Movies",
    f"{movie_info.get('unique_movies', rating_info.get('movies', 0)):,}",
)
c4.metric(
    "Duplicate ratings",
    f"{rating_info.get('duplicate_user_movie_pairs', 0):,}",
)

st.subheader("Rating Distribution")

distribution = rating_info.get("rating_distribution", {})

if distribution:
    rating_series = pd.Series(distribution, dtype="int64")
    rating_series.index = rating_series.index.astype(str)
    st.bar_chart(rating_series)
else:
    st.info("The report does not contain the rating distribution.")

st.subheader("Most Popular Genres")

if MOVIES_PATH.exists():
    movies = pd.read_parquet(MOVIES_PATH)

    if "genres" in movies.columns:
        genre_counts: dict[str, int] = {}

        for genres in movies["genres"].dropna():
            if isinstance(genres, str):
                # Also supports data created before genres were stored as lists.
                genres = genres.split("|")

            if hasattr(genres, "tolist"):
                genres = genres.tolist()

            if isinstance(genres, (list, tuple, set)):
                for genre in genres:
                    if genre and genre != "(no genres listed)":
                        genre_counts[str(genre)] = (
                            genre_counts.get(str(genre), 0) + 1
                        )
        if genre_counts:
            top_genres = (
                pd.Series(genre_counts)
                .sort_values(ascending=False)
                .head(10)
            )
            st.bar_chart(top_genres)
        else:
            st.info("No genres are available.")
    else:
        st.info("The movies table does not contain a genres column.")
else:
    st.info("data/processed/movies.parquet was not found.")

st.subheader("Data Quality Checks")

quality_tabs = st.tabs(["Ratings", "Movies"])

with quality_tabs[0]:
    st.write("**Missing values by column**")
    st.json(rating_info.get("missing_values", {}))

    st.metric(
        "Duplicate rows",
        rating_info.get("duplicate_rows", 0),
    )
    st.metric(
        "Duplicate user-movie pairs",
        rating_info.get("duplicate_user_movie_pairs", 0),
    )

with quality_tabs[1]:
    st.write("**Missing values by column**")
    st.json(movie_info.get("missing_values", {}))

    st.metric(
        "Duplicate rows",
        movie_info.get("duplicate_rows", 0),
    )
    st.metric(
        "Duplicate movie IDs",
        movie_info.get("duplicate_movie_ids", 0),
    )

with st.expander("View full JSON report"):
    st.json(quality)

if "raw" in quality:
    st.caption(
        "The report also includes metrics from the original CSV files "
        "in the raw section."
    )
