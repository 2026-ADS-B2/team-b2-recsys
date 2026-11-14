"""Evaluate the trained model on the test set: precision@10 (T6).

precision@10 for one user = (recommended movies the user rated in test) / 10.
The final metric is the average over all users in test.
The metric is saved to reports/metrics.json and logged to the same MLflow run as training.
"""
import json
import os
from pathlib import Path

import mlflow
import pandas as pd

DATA = Path("data/processed")
REPORTS = Path("reports")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
K = 10


def precision_at_k(recs: pd.DataFrame, test: pd.DataFrame, k: int = K) -> float:
    """recs: user_id, movie_id (top-k per user); test: user_id, movie_id."""
    relevant = test.groupby("user_id")["movie_id"].apply(set)
    recommended = recs.groupby("user_id")["movie_id"].apply(list)
    per_user = [
        len(set(recommended.get(user, [])[:k]) & movies) / k
        for user, movies in relevant.items()
    ]
    return float(sum(per_user) / len(per_user))


def main() -> None:
    run_id = json.load(open(REPORTS / "run.json"))["run_id"]
    test = pd.read_parquet(DATA / "test.parquet")

    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    users = pd.DataFrame({"user_id": sorted(test["user_id"].unique()), "n": K})
    recs = model.predict(users)

    metrics = {f"precision_at_{K}": round(precision_at_k(recs, test, K), 4)}
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_users", users["user_id"].nunique())

    json.dump(metrics, open(REPORTS / "metrics.json", "w"), indent=2)
    print(f"run {run_id}: {metrics} over {len(users)} test users")


if __name__ == "__main__":
    main()
