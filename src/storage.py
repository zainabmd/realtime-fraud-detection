"""
Parquet lakehouse writer + DuckDB query layer.

Write path (single writer = the consumer):
    * hive partitioning by event time:  <table>/dt=YYYY-MM-DD/hr=HH/part-<uuid>.parquet
    * atomic files: write ``*.tmp`` then rename – readers globbing ``*.parquet``
      never see a half-written file;
    * periodic compaction merges small micro-batch files (and de-duplicates
      replayed records by key) so queries stay fast.

Read path (dashboard, API): DuckDB scans the Parquet files in place – no
database server, no lock contention with the writer. Records can be duplicated
after an at-least-once replay, so the base views de-duplicate by ``txn_id``.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from src.utils import atomic_replace
from src.config import COMPACT_MIN_FILES, LABELS_DIR, LAKE_DIR, METRICS_DIR, SCORED_DIR, DLQ_DIR, get_logger

log = get_logger("storage")


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #
def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Microsecond timestamps: the portable Parquet unit (DuckDB cannot do interval math on ns).
    df.to_parquet(tmp, index=False, compression="zstd", coerce_timestamps="us", allow_truncated_timestamps=True)
    atomic_replace(tmp, path)


def write_partitioned(df: pd.DataFrame, base: Path, ts_col: str) -> int:
    """Append a micro-batch, split into dt/hr partitions on `ts_col` (UTC)."""
    if df.empty:
        return 0
    ts = pd.to_datetime(df[ts_col], utc=True)
    keys = pd.DataFrame({"dt": ts.dt.strftime("%Y-%m-%d"), "hr": ts.dt.strftime("%H")}, index=df.index)
    files = 0
    for (dt, hr), idx in keys.groupby(["dt", "hr"]).groups.items():
        _atomic_write(df.loc[idx], base / f"dt={dt}" / f"hr={hr}" / f"part-{uuid.uuid4().hex[:12]}.parquet")
        files += 1
    return files


def compact(base: Path, key: str | None, min_files: int = COMPACT_MIN_FILES) -> int:
    """Merge partitions holding many small files into one, de-duplicating on `key`.

    The merged file is written (atomically) *before* the sources are removed, so
    a reader may briefly see duplicates – which the de-duplicating views absorb –
    but never missing data.
    """
    merged = 0
    current_hour = time.strftime("hr=%H", time.gmtime())
    for part in sorted(p for p in base.glob("dt=*/hr=*") if p.is_dir()):
        files = sorted(part.glob("*.parquet"))
        closed = part.name != current_hour or part.parent.name != time.strftime("dt=%Y-%m-%d", time.gmtime())
        if len(files) < 2 or (len(files) < min_files and not closed):
            continue
        try:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if key and key in df.columns:
                df = df.drop_duplicates(subset=key, keep="first")
            _atomic_write(df, part / f"compacted-{uuid.uuid4().hex[:12]}.parquet")
            for f in files:
                f.unlink(missing_ok=True)
            merged += len(files)
        except Exception as exc:  # noqa: BLE001 – compaction is best-effort; never kill the consumer
            log.warning("Compaction of %s failed: %s", part, exc)
    if merged:
        log.info("Compacted %d files under %s", merged, base.name)
    return merged


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #
def _glob(base: Path) -> str:
    return (base / "**" / "*.parquet").as_posix()


def _has_files(base: Path) -> bool:
    return any(base.glob("dt=*/hr=*/*.parquet"))


def connect() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with de-duplicated views over each lake table."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads TO 4")
    if _has_files(SCORED_DIR):
        con.execute(f"""
            CREATE VIEW scored AS
            SELECT * EXCLUDE (dt, hr) FROM read_parquet('{_glob(SCORED_DIR)}', hive_partitioning = true, union_by_name = true)
            QUALIFY row_number() OVER (PARTITION BY txn_id ORDER BY processed_at) = 1""")
    if _has_files(LABELS_DIR):
        con.execute(f"""
            CREATE VIEW labels AS
            SELECT * EXCLUDE (dt, hr) FROM read_parquet('{_glob(LABELS_DIR)}', hive_partitioning = true, union_by_name = true)
            QUALIFY row_number() OVER (PARTITION BY txn_id ORDER BY label_time) = 1""")
    if _has_files(METRICS_DIR):
        con.execute(f"CREATE VIEW metrics AS SELECT * EXCLUDE (dt, hr) FROM "
                    f"read_parquet('{_glob(METRICS_DIR)}', hive_partitioning = true, union_by_name = true)")
    if _has_files(DLQ_DIR):
        con.execute(f"CREATE VIEW dlq AS SELECT * EXCLUDE (dt, hr) FROM "
                    f"read_parquet('{_glob(DLQ_DIR)}', hive_partitioning = true, union_by_name = true)")
    return con


def query(sql: str, params: list | None = None, retries: int = 3) -> pd.DataFrame:
    """Run SQL against the lake. Retries when compaction removes a file mid-scan."""
    for attempt in range(retries):
        try:
            con = connect()
            try:
                return con.execute(sql, params or []).df()
            finally:
                con.close()
        except (duckdb.IOException, duckdb.InvalidInputException) as exc:
            if attempt == retries - 1:
                raise
            log.debug("Lake read retry after %s", exc)
            time.sleep(0.2 * (attempt + 1))
        except duckdb.CatalogException:
            return pd.DataFrame()  # table has no files yet
    return pd.DataFrame()


def lake_summary() -> dict:
    """File counts / sizes per table – surfaced on the Pipeline Health page."""
    out = {}
    for name in ("scored", "labels", "metrics", "dlq"):
        files = list((LAKE_DIR / name).glob("dt=*/hr=*/*.parquet"))
        out[name] = {"files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
                     "partitions": len({f.parent for f in files})}
    return out
