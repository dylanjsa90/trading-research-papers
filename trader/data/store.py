"""Point-in-time storage: one parquet file per partition under DATA_DIR/<table>/, queried with DuckDB.

Every row carries `available_at`, a timezone-aware UTC instant when the row could first have been known.
`as_of(table, t)` is the only read path for features, so nothing later than t can leak in (§4.2).
"""

import os
from pathlib import Path

import duckdb
import pandas as pd

from trader import config


def _check_available_at(df):
    col = df.get("available_at")
    if col is None or not isinstance(col.dtype, pd.DatetimeTZDtype):
        raise ValueError("every row needs a timezone-aware available_at column")
    if col.isna().any():
        raise ValueError("available_at has missing values")


def _path(root, table, part):
    return Path(root) / table / f"{part}.parquet"


def upsert(table, df, key, partition, root=None):
    """Merge df into the table, one file per value of the `partition` column. Rows whose `key` already
    exists are replaced by the new version, so re-runs never duplicate. Returns the number of new keys."""
    _check_available_at(df)
    df = df.assign(available_at=df["available_at"].dt.tz_convert("UTC"))
    root = root or config.DATA_DIR
    added = 0
    for part, new in df.groupby(partition, sort=False):
        path = _path(root, table, part)
        old = pd.read_parquet(path) if path.exists() else new.iloc[:0]
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(["available_at", *key], ignore_index=True)
        added += len(merged) - len(old)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path.with_suffix(".tmp"), index=False)
        os.replace(path.with_suffix(".tmp"), path)
    return added


def read_partition(table, part, root=None):
    """One partition's stored rows (empty if none yet), for incremental fetch decisions, not for features."""
    path = _path(root or config.DATA_DIR, table, part)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def latest(table, column, root=None):
    """Largest stored value of one column (None if the table is empty); reads only that column."""
    folder = Path(root or config.DATA_DIR) / table
    if not any(folder.glob("*.parquet")):
        return None
    with duckdb.connect() as con:
        con.execute("SET TimeZone = 'UTC'")
        value = con.execute(f'SELECT max("{column}") AS v FROM read_parquet(?)', [str(folder / "*.parquet")]).df()
    return value["v"].iloc[0]


def list_items(table, column, root=None):
    """Distinct items across a list-valued column, e.g. every symbol that any stored article mentions."""
    folder = Path(root or config.DATA_DIR) / table
    if not any(folder.glob("*.parquet")):
        return set()
    with duckdb.connect() as con:
        rows = con.execute(f'SELECT DISTINCT unnest("{column}") FROM read_parquet(?)', [str(folder / "*.parquet")])
        return {item for (item,) in rows.fetchall()}


def as_of(table, ts, root=None):
    """All rows of `table` with available_at <= ts. ts must be timezone-aware: a naive time is ambiguous by
    hours, which is exactly the size of the leaks this guards against."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("as_of needs a timezone-aware timestamp")
    folder = Path(root or config.DATA_DIR) / table
    if not any(folder.glob("*.parquet")):
        return pd.DataFrame()
    with duckdb.connect() as con:
        con.execute("SET TimeZone = 'UTC'")
        return con.execute(
            "SELECT * FROM read_parquet(?, union_by_name = true) WHERE available_at <= ? ORDER BY available_at",
            [str(folder / "*.parquet"), ts.to_pydatetime()],
        ).df()
