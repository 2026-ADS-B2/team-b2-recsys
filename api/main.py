from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query

MOVIES_PATH = Path("data/processed/movies.parquet")

# Fixed popular movies (ml-latest-small ids) with a fake score
POPULAR = [
    (318, 0.93), (858, 0.91), (50, 0.90), (1221, 0.89), (527, 0.88),
    (2959, 0.87), (1193, 0.86), (593, 0.85), (260, 0.84), (296, 0.83),
]

movies = pd.read_parquet(MOVIES_PATH, columns=["movie_id", "title"])
TITLES = dict(zip(movies["movie_id"].astype(int), movies["title"]))

app = FastAPI(title="Movie Recommender API")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "mock"}


@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=1), n: int = Query(10, ge=1)):
    items = [
        {"movie_id": mid, "title": TITLES.get(mid, str(mid)), "score": score}
        for mid, score in POPULAR[:n]
    ]
    return {"user_id": user_id, "items": items}