"""Split ratings into train and test per user, by time (T4)."""
from pathlib import Path

import pandas as pd
import yaml

DATA = Path("data/processed")


def split_by_time(ratings: pd.DataFrame, test_size: float):
    df = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    position = df.groupby("user_id").cumcount()              # 0, 1, 2, ... per user
    n_ratings = df.groupby("user_id")["movie_id"].transform("size")
    n_test = (n_ratings * test_size).astype(int).clip(lower=1)
    is_test = position >= n_ratings - n_test                  # last ratings go to test
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["split"]
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    train, test = split_by_time(ratings, params["test_size"])

    assert set(test["user_id"]) <= set(train["user_id"]), "every test user must be in train"
    assert len(train) + len(test) == len(ratings)

    train.to_parquet(DATA / "train.parquet", index=False)
    test.to_parquet(DATA / "test.parquet", index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows, users: {train['user_id'].nunique()}")


if __name__ == "__main__":
    main()
