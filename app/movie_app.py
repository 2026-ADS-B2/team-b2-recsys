import json
from pathlib import Path

#import requests
import streamlit as st

from src.data.catalog import load_catalog

st.set_page_config(page_title="Movie Recommender", layout="wide")

# API and mock data
API_URL = "http://localhost:8000/recommend"
MOCK_FILE = Path(__file__).parent / "mock_response.json"

with open(MOCK_FILE, "r", encoding="utf-8") as file:
    mock_response = json.load(file)

@st.cache_data
def get_catalog():
    return load_catalog()


catalog = get_catalog()

# App title
st.title("Movie Recommender 🎬")
st.write("Find your next watch!")


st.subheader("Browse movies")

search_query = st.text_input("Search by title")

available_genres = sorted(
    {
        genre
        for movie_genres in catalog["genres"]
        for genre in movie_genres
    }
)

selected_genres = st.multiselect("Filter by genre", available_genres)

filtered_catalog = catalog

if search_query:
    filtered_catalog = filtered_catalog[
        filtered_catalog["title"].str.contains(
            search_query,
            case=False,
            na=False,
            regex=False,
        )
    ]

if selected_genres:
    filtered_catalog = filtered_catalog[
        filtered_catalog["genres"].apply(
            lambda movie_genres: any(
                genre in movie_genres for genre in selected_genres
            )
        )
    ]

st.caption(f"{len(filtered_catalog):,} movies match")

page_size = 15

if "movie_page" not in st.session_state:
    st.session_state.movie_page = 1

current_filters = (search_query, tuple(sorted(selected_genres)))
if st.session_state.get("movie_filters") != current_filters:
    st.session_state.movie_filters = current_filters
    st.session_state.movie_page = 1

total_pages = max(
    1,
    (len(filtered_catalog) + page_size - 1) // page_size,
)
st.session_state.movie_page = min(st.session_state.movie_page, total_pages)


page_start = (st.session_state.movie_page - 1) * page_size
page_movies = filtered_catalog.iloc[page_start : page_start + page_size]

if page_movies.empty:
    st.info("No movies match your search and genre filters. Try changing or clearing them.")
else:
    for row_start in range(0, len(page_movies), 5):
        card_columns = st.columns(5)
        movie_rows = page_movies.iloc[row_start : row_start + 5].iterrows()

        for column, (_, movie) in zip(card_columns, movie_rows):
            with column.container(border=True):
                st.image(movie["image_url"], use_container_width=True)
                st.markdown(f"**{movie['title']}**")

                if movie["n_ratings"] > 0:
                    st.write(f"Average rating: {movie['mean_rating']:.2f} / 5")
                else:
                    st.write("No ratings yet")

                st.select_slider(
                    "Your rating",
                    options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    value=3.0,
                    key=f"rating_{movie['movie_id']}",
                )

                imdb_url = movie["imdb_url"]
                if isinstance(imdb_url, str) and imdb_url:
                    st.link_button("View on IMDb", imdb_url)

_, previous_col, page_col, next_col, _ = st.columns([5, 1, 2, 1, 5])

if previous_col.button(
    "Previous",
    disabled=st.session_state.movie_page == 1,
):
    st.session_state.movie_page -= 1
    st.rerun()

page_col.write(f"Page {st.session_state.movie_page} of {total_pages}")

if next_col.button(
    "Next",
    disabled=st.session_state.movie_page == total_pages,
):
    st.session_state.movie_page += 1
    st.rerun()


# User input (commented out since we're not using it for now)
# user_id = st.number_input(
#     "Enter your user ID:",
#     min_value=1,
#     step=1
# )

# # Recommend button
# if st.button("Recommend"):
#     try:
#         response = requests.get(
#             API_URL,
#             params={"user_id": int(user_id), "n": 10},
#             timeout=3
#         )
#         response.raise_for_status()

#         recommendations = response.json()["items"]
#         st.subheader(f"Recommended movies for user {user_id}")

#     except (requests.RequestException, ValueError, KeyError):
#         recommendations = mock_response["items"]
#         st.info(
#             "The recommendation service is currently unavailable. "
#             "Here are some popular movies!"
#         )
#         st.subheader("Popular now")

#     for movie in recommendations:
#         st.write(f"**{movie['title']}** — Score: {movie['score']:.2f}")
