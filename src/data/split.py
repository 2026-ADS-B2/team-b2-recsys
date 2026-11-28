"""Split ratings into train, validation and test per user, by time (T4, extended in Sprint 2 T1).

For every user, ratings are sorted by time:
- the latest `test_size` share goes to test (same rule as in Sprint 1, so test is unchanged),
- the `val_size` share just before it goes to validation,
- everything earlier stays in train.
Every user keeps at least one rating in train.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

DATA = Path("data/processed")
REPORT = Path("reports/split_sizes.json")


def split_train_val_test(ratings: pd.DataFrame, test_size: float, val_size: float):
    """Return (train, val, test), split per user by time."""
    df = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    position = df.groupby("user_id").cumcount()              # 0, 1, 2, ... per user
    n_ratings = df.groupby("user_id")["movie_id"].transform("size")

    n_test = (n_ratings * test_size).astype(int).clip(lower=1)
    n_val = (n_ratings * val_size).astype(int)
    if val_size > 0:
        n_val = n_val.clip(lower=1)
    # never take a user's last train rating: at least one rating stays in train
    n_val = n_val.clip(upper=(n_ratings - n_test - 1).clip(lower=0))

    is_test = position >= n_ratings - n_test
    is_val = ~is_test & (position >= n_ratings - n_test - n_val)
    is_train = ~is_test & ~is_val

    def part(mask):
        return df[mask].reset_index(drop=True)

    return part(is_train), part(is_val), part(is_test)


def split_by_time(ratings: pd.DataFrame, test_size: float):
    """Sprint 1 train/test split, kept for backward compatibility."""
    train, _, test = split_train_val_test(ratings, test_size, val_size=0.0)
    return train, test


def split_sizes(parts: dict) -> dict:
    """Rows, users, movies, share and time range of each partition."""
    total = sum(len(df) for df in parts.values())
    report = {}
    for name, df in parts.items():
        report[name] = {
            "rows": int(len(df)),
            "share": round(len(df) / total, 4) if total else 0.0,
            "users": int(df["user_id"].nunique()),
            "movies": int(df["movie_id"].nunique()),
            "first_rating": str(df["timestamp"].min()) if len(df) else None,
            "last_rating": str(df["timestamp"].max()) if len(df) else None,
        }
    report["total_rows"] = int(total)
    return report


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["split"]
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    train, val, test = split_train_val_test(ratings, params["test_size"], params["val_size"])

    assert len(train) + len(val) + len(test) == len(ratings), "no rating may be lost"
    assert set(val["user_id"]) <= set(train["user_id"]), "every validation user must be in train"
    assert set(test["user_id"]) <= set(train["user_id"]), "every test user must be in train"

    train.to_parquet(DATA / "train.parquet", index=False)
    val.to_parquet(DATA / "val.parquet", index=False)
    test.to_parquet(DATA / "test.parquet", index=False)

    report = split_sizes({"train": train, "val": val, "test": test})
    report["params"] = {"test_size": params["test_size"], "val_size": params["val_size"]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"train: {len(train)} rows, val: {len(val)} rows, test: {len(test)} rows, "
        f"users: {train['user_id'].nunique()} -> {REPORT}"
    )


if __name__ == "__main__":
    main()
