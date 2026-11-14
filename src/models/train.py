"""Train the baseline "most popular movies" model and log it to MLflow (T5).

Popularity = average rating of movies that have at least `min_ratings` ratings.
Every user gets the same top-N list; personalised models come in Sprint 2.
"""
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema

DATA = Path("data/processed")
REPORTS = Path("reports")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "recsys"


class PopularityModel(mlflow.pyfunc.PythonModel):
    """Recommends the same top-rated popular movies to every user.

    Input:  DataFrame with a `user_id` column and an optional `n` column.
    Output: DataFrame with user_id, rank, movie_id, title, score (one row per recommendation).
    """

    def __init__(self, ranking: pd.DataFrame, default_n: int):
        self.ranking = ranking.reset_index(drop=True)  # movie_id, title, score (best first)
        self.default_n = default_n

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = []
        for _, req in model_input.iterrows():
            n = int(req["n"]) if "n" in req and pd.notna(req["n"]) else self.default_n
            top = self.ranking.head(n).copy()
            top.insert(0, "rank", range(1, len(top) + 1))
            top.insert(0, "user_id", int(req["user_id"]))
            rows.append(top)
        return pd.concat(rows, ignore_index=True)


def build_ranking(train: pd.DataFrame, movies: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    stats = train.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean").reset_index()
    stats = stats[stats["n_ratings"] >= min_ratings]
    # best average rating first; ties broken by number of ratings, then movie_id (deterministic)
    stats = stats.sort_values(["mean_rating", "n_ratings", "movie_id"], ascending=[False, False, True])
    stats["score"] = (stats["mean_rating"] / 5.0).round(4)  # 0..1, like the API example
    ranking = stats.merge(movies[["movie_id", "title"]], on="movie_id", how="left")
    return ranking[["movie_id", "title", "score", "n_ratings", "mean_rating"]].reset_index(drop=True)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["train"]
    train = pd.read_parquet(DATA / "train.parquet")
    movies = pd.read_parquet(DATA / "movies.parquet")

    ranking = build_ranking(train, movies, params["min_ratings"])
    assert len(ranking) >= params["n"], "not enough movies with min_ratings; lower train.min_ratings"
    model = PopularityModel(ranking[["movie_id", "title", "score"]], default_n=params["n"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="popularity-baseline") as run:
        mlflow.log_params({
            "model_type": "popularity",
            "n": params["n"],
            "min_ratings": params["min_ratings"],
            "train_rows": len(train),
            "candidate_movies": len(ranking),
        })
        example = pd.DataFrame({"user_id": [1], "n": [params["n"]]})
        signature = ModelSignature(
            inputs=Schema([ColSpec("long", "user_id"), ColSpec("long", "n", required=False)]),
            outputs=infer_signature(example, model.predict(None, example)).outputs,
        )
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            signature=signature,
            input_example=example,
        )
        run_id = run.info.run_id

    REPORTS.mkdir(exist_ok=True)
    json.dump({"run_id": run_id, "experiment": EXPERIMENT, "model_uri": f"runs:/{run_id}/model"},
              open(REPORTS / "run.json", "w"), indent=2)
    print(f"run {run_id}: {len(ranking)} candidate movies, top {params['n']}:")
    print(ranking.head(params["n"])[["movie_id", "title", "n_ratings", "mean_rating"]].to_string(index=False))


if __name__ == "__main__":
    main()
