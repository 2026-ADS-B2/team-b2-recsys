import json
from pathlib import Path
import pandas as pd
import streamlit as st

REPORT_PATH = Path("reports/quality.json")
MOVIES_PATH = Path("data/processed/movies.parquet")
RATINGS_PATH = Path("data/processed/ratings.parquet")

st.set_page_config(
    page_title="MovieLens Dataset",
    layout="wide",
)

st.title("MovieLens dataset")
st.write("Overview of the dataset and its quality.")

if not REPORT_PATH.exists():
    st.warning(
        "Quality report not found. "
        "Run `dvc repro quality` from the project root."
    )
    st.stop()

with REPORT_PATH.open(encoding="utf-8") as file:
    quality = json.load(file)

processed = quality.get("processed", {})
rating_info = processed.get("ratings", {})
movie_info = processed.get("movies", {})


st.header("Dataset overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Ratings", f"{rating_info.get('rows', 0):,}")
col2.metric("Users", f"{rating_info.get('users', 0):,}")
col3.metric(
    "Movies",
    f"{movie_info.get('unique_movies', rating_info.get('movies', 0)):,}",
)
col4.metric(
    "Duplicate ratings",
    f"{rating_info.get('duplicate_user_movie_pairs', 0):,}",
)


st.header("Rating distribution")

distribution = rating_info.get("rating_distribution", {})

if distribution:
    rating_series = pd.Series(distribution, dtype="int64")
    rating_series.index = rating_series.index.astype(str)
    st.bar_chart(rating_series, y_label="Number of ratings")
else:
    st.info("Rating distribution is not available.")


st.header("Genre analysis")

if MOVIES_PATH.exists() and RATINGS_PATH.exists():
    movies = pd.read_parquet(MOVIES_PATH)
    ratings = pd.read_parquet(RATINGS_PATH)

    if "genres" in movies.columns:
        genre_data = movies[["movie_id", "genres"]].copy()

        genre_data["genres"] = genre_data["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str)
            else x.tolist() if hasattr(x, "tolist")
            else x if isinstance(x, (list, tuple, set))
            else []
        )

        genre_data = genre_data.explode("genres")

        genre_data = genre_data[
            genre_data["genres"].notna()
            & (genre_data["genres"] != "")
            & (genre_data["genres"] != "(no genres listed)")
        ]


        common_genres = (
            genre_data.groupby("genres")["movie_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        st.subheader("Most common genres")
        st.caption("Number of movies associated with each genre.")

        st.bar_chart(
            common_genres,
            horizontal=True,
            y_label="Number of movies",
        )

        
        genre_ratings = genre_data.merge(
            ratings[["movie_id", "rating"]],
            on="movie_id",
            how="inner",
        )

        highest_rated_genres = (
        genre_ratings.groupby("genres")["rating"]
        .mean()
        .sort_values(ascending=False)
        )

        st.subheader("Highest-rated genres")
        st.caption("Average rating of movies in each genre.")
        
        st.bar_chart(
            highest_rated_genres,
            horizontal=True,
            y_label="Average rating",
        )

    else:
        st.info("The movies table does not contain a genres column.")

else:
    st.info("Processed movies or ratings data is not available.")


st.header("Data quality")

left, right = st.columns(2)

with left:
    st.subheader("Ratings")

    st.write("Missing values by column")
    st.json(rating_info.get("missing_values", {}))

    st.write("Duplicate rows:", rating_info.get("duplicate_rows", 0))
    st.write(
        "Duplicate user-movie pairs:",
        rating_info.get("duplicate_user_movie_pairs", 0),
    )

with right:
    st.subheader("Movies")

    st.write("Missing values by column")
    st.json(movie_info.get("missing_values", {}))

    st.write("Duplicate rows:", movie_info.get("duplicate_rows", 0))
    st.write(
        "Duplicate movie IDs:",
        movie_info.get("duplicate_movie_ids", 0),
    )

with st.expander("View complete quality report"):
    st.json(quality)

if "raw" in quality:
    st.caption(
        "The report also contains statistics from the original CSV files."
    )
