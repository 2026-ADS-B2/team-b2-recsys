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
