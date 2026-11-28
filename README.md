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
| `reports/` | Generated reports (quality.json, metrics.json, split_sizes.json) |

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

## Train / validation / test split

The `split` stage (`src/data/split.py`) splits ratings **per user, by time**:

| Partition | File | Share | What it is |
|---|---|---|---|
| train | `data/processed/train.parquet` | ~70% | each user's earliest ratings, used for training |
| validation | `data/processed/val.parquet` | ~10% | the ratings just before test, used to tune models (e.g. k in kNN) |
| test | `data/processed/test.parquet` | ~20% | each user's latest ratings, used only for the final evaluation |

- Shares are set in `params.yaml` (`split.test_size`, `split.val_size`).
- The test partition is identical to Sprint 1, so new models can be compared with the baseline.
- Every user keeps at least one rating in train.
- Partition sizes (rows, users, movies, share, date range) are written to `reports/split_sizes.json`.

```bash
dvc repro split
```

## Data quality report 

The data quality report is generated from the processed Parquet files and saved to `reports/quality.json`.

To generate or update the report, run:

```bash
dvc repro quality
```

The report includes:
- Total number of rows.
- Number of distinct users and movies.
- Missing values per column.
- Duplicate counts.
- Rating distribution.

The quality report is integrated into the DVC pipeline as the `quality` stage.

## How to start MLflow

Run these from the repository root (so MLflow finds `mlflow.db` and `mlruns/`):

```
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

> **Mac users:** port 5000 is used by AirPlay Receiver, so use `--port 5001` and open http://localhost:5001 instead.

Open http://localhost:5000 → experiment **recsys** shows the training runs (params, `precision_at_10`, model),
and **Models** shows `movie-recommender`.

### Build and register the model with one command

`reports/run.json` points to the MLflow run of whoever trained the model last, and that run only exists
on their laptop. So if you never trained the model yourself, `dvc repro` skips training ("didn't change")
and `register.py` fails with `Run with id=... not found`. To build everything from scratch and get
`movie-recommender@champion` locally, run:

```
./scripts/train_and_register.sh
```

It runs `dvc pull` → `dvc repro -f` (forces all stages, including training) → `python src/models/register.py`.
Then start MLflow (see above) and check **Models → movie-recommender** has the alias `champion`.

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

## How to run the movie app

The movie recommendation app first tries to get recommendations from the API at `http://localhost:8000/recommend`.

If the API is unavailable or takes longer than 3 seconds to respond, the app displays popular movies using the mock data stored in `app/mock_response.json`.

To launch the app, run the following command from the repository root:

```bash
streamlit run app/movie_app.py --server.port 8502
```

Open http://localhost:8502 in your browser.

Enter a user ID and click **Recommend** to see the recommended movies and their scores.

## How to run the project

Follow these steps from the repository root.

### 1. Install dependencies

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data

Make sure you have configured your DagsHub credentials as explained in the **How to get the data** section above.

Download the project data:

```bash
dvc pull
```

### 3. Run the data pipeline

Run all pipeline stages (cleaning, splitting, training and evaluation):

```bash
dvc repro
```

### 4. Start MLflow

Start the MLflow server:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000 to access the MLflow UI (Mac: use `--port 5001`, port 5000 is taken by AirPlay).

> **First time / never trained the model?** Steps 2, 3 and 5 can be replaced by one command:
> `./scripts/train_and_register.sh`

### 5. Register the model

Open a new terminal, activate the virtual environment and run:

```bash
source .venv/bin/activate
python src/models/register.py
```

This registers the trained model as `movie-recommender` and assigns it the `champion` alias.

### 6. Start the API

Open another terminal and start the FastAPI service.

Make sure MLflow is running and the model is registered (steps 4 and 5), then run from the repository root:

```bash
source .venv/bin/activate
uvicorn api.main:app --port 8000
```

On Windows (Git Bash) activate with `source .venv/Scripts/activate`.

Check it: http://localhost:8000/health shows the model version, and http://localhost:8000/docs shows the API documentation.

The API should be available at `http://localhost:8000`.

### 7. Start the Streamlit apps

Open a new terminal, activate the virtual environment and start the movie recommendation app:

```bash
source .venv/bin/activate
streamlit run app/movie_app.py --server.port 8502
```

Open http://localhost:8502 to use the movie recommender.

### Data quality dashboard

The data dashboard displays key dataset statistics, the rating distribution, the most popular genres and data quality checks.

Make sure the data has been downloaded and the quality report has been generated with:

```bash
dvc pull
dvc repro quality
```

Start the dashboard from the repository root:

```bash
streamlit run app/dashboard.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### Continuous Integration 

GitHub Actions automatically runs the test suite on every pull request and on pushes to the `main` branch.

The workflow is defined in `.github/workflows/ci.yml`.

To run the tests locally:

```bash
python -m pytest -q
```



**Important:** MLflow and the API need to keep running in their own terminals while you use the app. The movie app can still display mock recommendations if the API isn't available.


## Demo

The step-by-step demo script for the sprint review is in [`docs/demo.md`](docs/demo.md).

Quick order: `dvc repro` → MLflow (5000) → `register.py` → API (8000) → movie app (8502) → dashboard (8501).
Note: the baseline model recommends the same most popular movies to every user; personalised models come in Sprint 2.

 
