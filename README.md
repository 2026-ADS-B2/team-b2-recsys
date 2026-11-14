# Movie Recommender (team B2)

## Project structure

| Folder | What goes there |
|---|---|
| `data/` | Raw and processed data, tracked by DVC (not Git) |
| `src/data/` | Data cleaning, splitting and quality scripts |
| `src/models/` | Training, evaluation and model registration |
| `api/` | FastAPI service |
| `app/` | Streamlit apps (movie app, data dashboard) |
| `docs/` | Architecture and task documentation |
| `tests/` | Unit tests |
| `reports/` | Generated reports (quality.json, metrics.json) |

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

We use the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) dataset (~100k ratings, 610 users, ~9.7k movies).
Data is versioned with **DVC** and stored on **DagsHub**: https://dagshub.com/ejakovleva/team-b2-recsys

| File | Description |
|---|---|
| `data/raw/ratings.csv` | userId, movieId, rating, timestamp |
| `data/raw/movies.csv` | movieId, title, genres |
| `data/raw/tags.csv` | user tags for movies |
| `data/raw/links.csv` | links to IMDb and TMDB ids |

### How to get the data

1. Sign up on [DagsHub](https://dagshub.com) with your GitHub account and accept the collaborator invite.
2. Copy your token: DagsHub → avatar → **Settings → Tokens**.
3. Run (once), replacing `<your-dagshub-username>` and `<your-token>`:

```
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-token>
```

4. Download the data:

```
dvc pull
```

> `--local` saves your credentials in `.dvc/config.local`, which is not committed to Git. Never commit your token.

## How to start MLflow

Run these from the repository root (so MLflow finds `mlflow.db` and `mlruns/`):

```
source .venv/bin/activate
mlflow server --port 5000
```

Open http://localhost:5000 → experiment **recsys** shows the training runs (params, `precision_at_10`, model),
and **Models** shows `movie-recommender`.

### Register the model as @champion

After `dvc repro`, register the model from the latest run (`reports/run.json`):

```
python src/models/register.py
```

This creates a new version of `movie-recommender` (version 1 the first time) and sets the alias `champion` on it.
Running it again for the same run does not create a duplicate version.
Load the model anywhere (e.g. in the API) with:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/movie-recommender@champion")
```
