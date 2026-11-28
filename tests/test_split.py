"""Unit tests for the T4 train/test split (T18)."""
import pandas as pd

from src.data.split import split_by_time
from src.data.split import split_sizes, split_train_val_test


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



# --- Sprint 2 T1: validation partition -------------------------------------------------


def test_three_partitions_cover_all_rows_without_overlap():
    ratings = make_ratings()

    train, val, test = split_train_val_test(ratings, test_size=0.2, val_size=0.1)

    assert len(train) + len(val) + len(test) == len(ratings)
    both = pd.concat([train, val, test])
    assert not both.duplicated(["user_id", "movie_id"]).any()


def test_validation_sits_between_train_and_test_in_time():
    train, val, test = split_train_val_test(make_ratings(), test_size=0.2, val_size=0.1)

    # user 2 has 10 ratings: 2 test, 1 val, 7 train
    assert sorted(test[test["user_id"] == 2]["movie_id"]) == [208, 209]
    assert sorted(val[val["user_id"] == 2]["movie_id"]) == [207]
    for user in [1, 2]:
        u_train = train[train["user_id"] == user]["timestamp"]
        u_val = val[val["user_id"] == user]["timestamp"]
        u_test = test[test["user_id"] == user]["timestamp"]
        assert u_train.max() < u_val.min() <= u_val.max() < u_test.min()


def test_test_partition_is_unchanged_from_sprint_1():
    ratings = make_ratings()

    _, old_test = split_by_time(ratings, test_size=0.2)
    _, _, new_test = split_train_val_test(ratings, test_size=0.2, val_size=0.1)

    pd.testing.assert_frame_equal(old_test, new_test)


def test_every_user_keeps_at_least_one_train_rating():
    # a user with only 2 ratings: 1 goes to test, validation must not take the last train rating
    ratings = pd.DataFrame(
        {"user_id": [1, 1], "movie_id": [10, 11], "rating": [4.0, 3.0], "timestamp": [1, 2]}
    )

    train, val, test = split_train_val_test(ratings, test_size=0.2, val_size=0.1)

    assert len(train) == 1 and len(val) == 0 and len(test) == 1


def test_split_sizes_report_counts_each_partition():
    train, val, test = split_train_val_test(make_ratings(), test_size=0.2, val_size=0.1)

    report = split_sizes({"train": train, "val": val, "test": test})

    assert report["total_rows"] == 15
    assert report["train"]["rows"] + report["val"]["rows"] + report["test"]["rows"] == 15
    assert report["val"]["users"] == 2
    assert abs(report["train"]["share"] + report["val"]["share"] + report["test"]["share"] - 1) < 1e-3