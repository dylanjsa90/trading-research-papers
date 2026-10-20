import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trader.data import store

BASE = pd.Timestamp("2024-01-01", tz="UTC")


def _rows(ids, offsets_s, part="a", text=None):
    return pd.DataFrame(
        {
            "id": ids,
            "available_at": [BASE + pd.Timedelta(seconds=s) for s in offsets_s],
            "part": part,
            "text": text if text is not None else [f"v{i}" for i in ids],
        }
    )


@settings(max_examples=150, deadline=None)
@given(
    offsets=st.lists(st.integers(-1000, 1000), min_size=1, max_size=25),
    probe=st.integers(-1000, 1000),
    data=st.data(),
)
def test_as_of_returns_exactly_the_rows_available_by_t(tmp_path_factory, offsets, probe, data):
    root = tmp_path_factory.mktemp("store")
    parts = data.draw(st.lists(st.sampled_from("abc"), min_size=len(offsets), max_size=len(offsets)))
    df = _rows(list(range(len(offsets))), offsets)
    df["part"] = parts
    store.upsert("t", df, key=["id"], partition="part", root=root)

    t = BASE + pd.Timedelta(seconds=probe)
    got = store.as_of("t", t, root=root)

    assert (got["available_at"] <= t).all()
    assert sorted(got["id"]) == [i for i, s in enumerate(offsets) if s <= probe]


def test_as_of_compares_instants_not_wall_clocks(tmp_path):
    # 20:30 UTC is 16:30 in New York; a wall-clock comparison would call this row "after" 16:30 ET.
    store.upsert("t", _rows([1], [20 * 3600 + 1800]), key=["id"], partition="part", root=tmp_path)
    t_ny = pd.Timestamp("2024-01-01 15:30", tz="America/New_York")  # 20:30 UTC
    assert list(store.as_of("t", t_ny, root=tmp_path)["id"]) == [1]


def test_as_of_rejects_naive_timestamp(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        store.as_of("t", pd.Timestamp("2024-01-01"), root=tmp_path)


def test_as_of_on_missing_table_is_empty(tmp_path):
    assert store.as_of("nothing_here", BASE, root=tmp_path).empty


def test_rerun_with_same_rows_adds_nothing(tmp_path):
    rows = _rows([1, 2, 3], [0, 10, 20])
    assert store.upsert("t", rows, key=["id"], partition="part", root=tmp_path) == 3
    assert store.upsert("t", rows, key=["id"], partition="part", root=tmp_path) == 0
    assert len(store.as_of("t", BASE + pd.Timedelta(days=1), root=tmp_path)) == 3


def test_overlapping_rerun_appends_new_rows_and_keeps_latest_version(tmp_path):
    store.upsert("t", _rows([1, 2], [0, 10]), key=["id"], partition="part", root=tmp_path)
    added = store.upsert("t", _rows([2, 3], [10, 20], text=["v2-edited", "v3"]), key=["id"], partition="part",
                         root=tmp_path)  # fmt: skip
    got = store.as_of("t", BASE + pd.Timedelta(days=1), root=tmp_path).set_index("id")["text"].to_dict()
    assert added == 1
    assert got == {1: "v1", 2: "v2-edited", 3: "v3"}


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"id": [1], "part": ["a"]}),  # no available_at column
        pd.DataFrame({"id": [1], "part": ["a"], "available_at": pd.Series([pd.NaT], dtype="datetime64[us, UTC]")}),
        pd.DataFrame({"id": [1], "part": ["a"], "available_at": [pd.Timestamp("2024-01-01")]}),  # naive
    ],
)
def test_upsert_refuses_rows_without_a_usable_available_at(tmp_path, bad):
    with pytest.raises(ValueError, match="available_at"):
        store.upsert("t", bad, key=["id"], partition="part", root=tmp_path)
    assert store.as_of("t", BASE, root=tmp_path).empty


def test_keys_lists_distinct_key_tuples_across_partitions(tmp_path):
    df = _rows([1, 2, 1], [0, 1, 2], text=["x", "y", "z"])
    df["part"], df["model"] = ["a", "b", "b"], ["m1", "m1", "m2"]
    store.upsert("t", df, key=["id", "model"], partition="part", root=tmp_path)

    assert store.keys("t", ["id", "model"], root=tmp_path) == {(1, "m1"), (2, "m1"), (1, "m2")}
    assert store.keys("missing", ["id"], root=tmp_path) == set()


def test_as_of_can_read_only_some_columns(tmp_path):
    store.upsert("t", _rows([1, 2], [0, 5]), key=["id"], partition="part", root=tmp_path)
    got = store.as_of("t", BASE + pd.Timedelta(seconds=1), root=tmp_path, columns=["id"])
    assert list(got.columns) == ["id"] and list(got["id"]) == [1]
