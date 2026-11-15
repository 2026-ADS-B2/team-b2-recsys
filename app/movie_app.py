import json
from pathlib import Path

import requests
import streamlit as st

# API and mock data
API_URL = "http://localhost:8000/recommend"
MOCK_FILE = Path(__file__).parent / "mock_response.json"

with open(MOCK_FILE, "r", encoding="utf-8") as file:
    mock_response = json.load(file)

# App title
st.title("Movie Recommender 🎬")
st.write("Find your next watch!")

# User input
user_id = st.number_input(
    "Enter your user ID:",
    min_value=1,
    step=1
)

# Recommend button
if st.button("Recommend"):
    try:
        response = requests.get(
            API_URL,
            params={"user_id": int(user_id), "n": 10},
            timeout=3
        )
        response.raise_for_status()

        recommendations = response.json()["items"]
        st.subheader(f"Recommended movies for user {user_id}")

    except (requests.RequestException, ValueError, KeyError):
        recommendations = mock_response["items"]
        st.info(
            "The recommendation service is currently unavailable. "
            "Here are some popular movies!"
        )
        st.subheader("Popular now")

    for movie in recommendations:
        st.write(f"**{movie['title']}** — Score: {movie['score']:.2f}")
