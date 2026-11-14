"""Register the trained model in the MLflow Model Registry with alias @champion (T7).

Reads the run id from reports/run.json (written by the train stage), registers
runs:/<run_id>/model as a new version of "movie-recommender" and points the
"champion" alias at it. The API loads models:/movie-recommender@champion.
"""
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

REPORTS = Path("reports")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "movie-recommender"
ALIAS = "champion"


def main() -> None:
    run = json.load(open(REPORTS / "run.json"))
    run_id = run["run_id"]

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # do not create a duplicate version if this run is already registered
    existing = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
    if existing:
        version = existing[0].version
        print(f"run {run_id} is already registered as version {version}")
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version

    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)

    # check that the API will be able to load it
    mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    print(f"{MODEL_NAME} version {version} is now @{ALIAS} (run {run_id})")


if __name__ == "__main__":
    main()
