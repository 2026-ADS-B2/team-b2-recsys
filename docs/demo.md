# Sprint 1 Demo Script (T19)

**Length:** 5–7 minutes · **Sprint goal:** a minimal pipeline running end to end (data → model → registry → API → app), plus a data dashboard.

> Presenters below are a proposal (each person shows what they built). Please confirm or swap in the group chat.

## Before the demo (10 min earlier, on the presenting laptop)

Open **4 terminals**, all in the repository root with the virtual environment active (`source .venv/bin/activate`).

| Terminal | Command | Check |
|---|---|---|
| 1 | `./scripts/train_and_register.sh` | ends with `movie-recommender version N is now @champion` |
| 2 | `mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000` (Mac: `--port 5001`) | http://localhost:5000 opens |
| 3 | `uvicorn api.main:app --port 8000` (from T14, PR #113) | http://localhost:8000/docs shows the endpoints |
| 4 | `streamlit run app/dashboard.py --server.port 8501` and `streamlit run app/movie_app.py --server.port 8502` (one tab each) | both pages open |

Open these browser tabs in this order: GitHub Kanban → MLflow → Dashboard (8501) → API docs (8000) → Movie app (8502).

## The demo, step by step

| # | Time | Part | Presenter | What to show / say |
|---|---|---|---|---|
| 0 | 0:30 | **Intro & process** | Sıla | Sprint goal; the GitHub Kanban (Epics → User Stories → Tasks T1–T19, each with one owner); how we worked (branch → PR → review → merge, CI on every PR). |
| 1 | 1:00 | **Data & pipeline** | Katia | Data lives on DagsHub, versioned with DVC (`dvc pull`). Show `dvc.yaml`: `clean → split → train → evaluate → quality`. Run `dvc repro` and point out that unchanged stages are skipped. |
| 2 | 0:45 | **Reproducibility in one command** | Sıla | Anyone can rebuild the model from scratch: `./scripts/train_and_register.sh` (`dvc pull` → `dvc repro -f` → register). Show the last line of terminal 1. |
| 3 | 1:00 | **Model & MLflow** | Arina | MLflow UI: experiment **recsys**, the run's params and `precision_at_10` (≈ 0.026 for the popularity baseline). Models tab: `movie-recommender` with alias **@champion**. |
| 4 | 1:00 | **Data dashboard & quality** | Maria | Dashboard (8501): dataset size, rating distribution, top genres, quality checks from `reports/quality.json`. Mention CI: tests run on every pull request (`python -m pytest -q`). |
| 5 | 0:45 | **API** | Mirari | API docs (8000/docs): call `/recommend` for a user id. With T14 it returns mock data; once T15 is merged it loads `models:/movie-recommender@champion`. |
| 6 | 1:00 | **Movie app** | Azucena | Movie app (8502): enter a user id → top-10 movies with scores. Then stop the API and show the **fallback** (popular movies from `app/mock_response.json`). |
| 7 | 0:30 | **Wrap-up** | Sıla | What is done, what moves to Sprint 2 (e.g. E5 monitoring & retraining, E7 search / rate / preferences), and one lesson from the retro. |

## If something goes wrong

- **API not running:** the movie app still works with mock data; say so and show the fallback on purpose.
- **`register.py` fails with `Run with id=... not found`:** run `./scripts/train_and_register.sh` instead of `dvc repro`.
- **MLflow page does not open on Mac:** port 5000 is taken by AirPlay Receiver, use `--port 5001`.
- **`dvc pull` asks for credentials:** see "How to get the data" in the README (DagsHub user + token with `--local`).

## Rehearsal checklist (Tuesday)

- [ ] Each presenter ran their part once on the presenting laptop
- [ ] Whole demo fits in 7 minutes
- [ ] T14 (PR #113) merged; T15 (real model in the API) merged or demo uses the mock API
- [ ] Kanban is up to date (no unassigned tasks, cards in the right column)
