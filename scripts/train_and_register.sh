#!/usr/bin/env bash
# T17: rebuild the model from scratch and register it as movie-recommender@champion
set -e
cd "$(dirname "$0")/.."

echo "1/3 Downloading data (dvc pull)..."
dvc pull

echo "2/3 Running the full pipeline, forcing all stages (dvc repro -f)..."
dvc repro -f

echo "3/3 Registering the model..."
python src/models/register.py

echo "Done! Start MLflow with:"
echo "  mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000"
echo "  (Mac: port 5000 is used by AirPlay, use --port 5001)"
