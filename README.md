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
