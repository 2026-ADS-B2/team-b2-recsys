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
mlflow server --port 5000
```

Open http://localhost:5000 to access the MLflow UI.

### 5. Register the model

Open a new terminal, activate the virtual environment and run:

```bash
source .venv/bin/activate
python src/models/register.py
```

This registers the trained model as `movie-recommender` and assigns it the `champion` alias.

### 6. Start the API

Open another terminal and start the FastAPI service.

**API start command: to be added by Mirari.**

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
**Important:** MLflow and the API need to keep running in their own terminals while you use the app. The movie app can still display mock recommendations if the API isn't available.
