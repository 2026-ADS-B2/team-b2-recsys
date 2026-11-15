import json
from pathlib import Path

import streamlit as st

# Load mock recommendations
MOCK_FILE = Path(__file__).parent / "mock_response.json"

with open(MOCK_FILE, "r", encoding="utf-8") as file:
    mock_response = json.load(file)

# App title
st.title("Movie Recommender 🎬")
st.write("Find movies you might enjoy!")

# User input
user_id = st.number_input(
    "Enter your user ID:",
    min_value=1,
    step=1
)

# Recommend button
if st.button("Recommend"):
    if user_id == mock_response["user_id"]:
        recommendations = mock_response["items"]
    else:
        recommendations = mock_response["items"]

    st.subheader(f"Recommended movies for user {user_id}")

    for movie in recommendations:
        st.write(f"**{movie['title']}** — Score: {movie['score']:.2f}")