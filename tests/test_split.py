"""Unit tests for the T4 train/test split (T18)."""
import pandas as pd

from src.data.split import split_by_time


def make_ratings():
    # user 1 has 5 ratings, user 2 has 10 ratings, timestamps in increasing order
    rows = [(1, m, 4.0, t) for t, m in enumerate(range(100, 105))]
    rows += [(2, m, 3.0, t) for t, m in enumerate(range(200, 210))]
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return df.sample(frac=1, random_state=0)  # shuffle, the split must sort by time itself


def test_train_plus_test_is_all_rows():
    ratings = make_ratings()

    train, test = split_by_time(ratings, test_size=0.2)

    assert len(train) + len(test) == len(ratings)
    both = pd.concat([train, test])
    assert not both.duplicated(["user_id", "movie_id"]).any()


def test_every_test_user_also_appears_in_train():
    train, test = split_by_time(make_ratings(), test_size=0.2)

    assert set(test["user_id"]) <= set(train["user_id"])


def test_the_latest_ratings_of_each_user_go_to_test():
    train, test = split_by_time(make_ratings(), test_size=0.2)

    # 20% of 5 = 1 test rating for user 1, 20% of 10 = 2 test ratings for user 2
    assert sorted(test[test["user_id"] == 1]["movie_id"]) == [104]
    assert sorted(test[test["user_id"] == 2]["movie_id"]) == [208, 209]
    for user in [1, 2]:
        assert train[train["user_id"] == user]["timestamp"].max() < test[test["user_id"] == user]["timestamp"].min()
