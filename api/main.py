import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from mlflow import MlflowClient

MOVIES_PATH = Path("data/processed/movies.parquet")
# Same default as src/models/train.py and register.py, so run from the repo root
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "movie-recommender"
ALIAS = "champion"

state = {"model": None, "version": None, "titles": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load titles and the model
    movies = pd.read_parquet(MOVIES_PATH, columns=["movie_id", "title"])
    state["titles"] = dict(zip(movies["movie_id"].astype(int), movies["title"]))

    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    state["version"] = str(version)
    state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    yield


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["version"]}


@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=1), n: int = Query(10, ge=1)):
    try:
        request = pd.DataFrame({"user_id": [user_id], "n": [n]}).astype("int64")
        prediction = state["model"].predict(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model error: {exc}")

    items = [
        {
            "movie_id": int(row.movie_id),
            "title": state["titles"].get(int(row.movie_id), str(row.movie_id)),
            "score": round(float(row.score), 4),
        }
        for row in prediction.itertuples()
    ]
    return {"user_id": user_id, "items": items}