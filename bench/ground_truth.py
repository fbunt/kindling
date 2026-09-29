"""Host-side ground-truth computation for the benchmark questions.

Reference Polars code from bench/questions.py runs here, in-process, against
the same parquet the sandbox queries — its (normalized) output defines the
expected answer. Results are cached per (parquet identity, reference-code hash)
so each reference runs once per dataset.
"""

import functools
import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import polars as pl

from app.query_engine import _build_lazyframe
from bench.questions import Question

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".bench-runs/ground_truth")


class GroundTruthError(Exception):
    pass


_PARQUET_MAGIC = b"PAR1"


def _parquet_footer(path: Path) -> bytes:
    """Raw Thrift FileMetaData footer of one parquet file (schema, row groups,
    column-chunk offsets and statistics): a content fingerprint that needs no
    full read."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size < 12:
            raise GroundTruthError(f"{path}: too small to be a parquet file")
        f.seek(size - 8)
        tail = f.read(8)
        if tail[4:] != _PARQUET_MAGIC:
            raise GroundTruthError(f"{path}: missing parquet magic bytes")
        footer_len = int.from_bytes(tail[:4], "little")
        if footer_len > size - 12:
            raise GroundTruthError(f"{path}: corrupt parquet footer length")
        f.seek(size - 8 - footer_len)
        return f.read(footer_len)


def parquet_parts(path: Path | str) -> list[Path]:
    """The parquet files behind a dataset path: a single file, or the sorted
    *.parquet parts of a (dask-written) directory."""
    p = Path(path).resolve()
    if p.is_dir():
        return sorted(e for e in p.iterdir() if e.suffix == ".parquet")
    return [p]


def parquet_identity(path: Path | str) -> str:
    """Stable 12-hex-char content identity for a parquet file or directory.

    Hashes each part's name (directories only), size, and raw parquet footer,
    plus the total row count. No mtime and no absolute path, so a copy, a
    re-symlink or a `touch` keeps the identity while any rewrite of the data
    changes it. The default dataset is a directory of ~600 dask part files.
    """
    p = Path(path).resolve()
    parts = parquet_parts(p)
    if not parts:
        raise GroundTruthError(f"{p}: no parquet files")
    h = hashlib.sha256()
    for part in parts:
        name = part.name if p.is_dir() else ""
        footer = _parquet_footer(part)
        h.update(f"{name}:{part.stat().st_size}:{len(footer)}\n".encode())
        h.update(footer)
    rows = pl.scan_parquet(parts).select(pl.len()).collect().item()
    h.update(f"rows:{rows}\n".encode())
    return h.hexdigest()[:12]


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary-least-squares slope, plain Python (cov(x,y) / var(x))."""
    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("ols_slope needs two same-length sequences, n >= 2")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    var = sum((x - mx) ** 2 for x in xs)
    return cov / var


def _normalize(value):
    """Coerce a reference result to a JSON-serializable form."""
    if isinstance(value, pl.LazyFrame):
        value = value.collect(engine="streaming")
    if isinstance(value, pl.DataFrame):
        if value.height == 1 and value.width == 1:
            return _normalize(value.item())
        return [{k: _normalize(v) for k, v in row.items()} for row in value.to_dicts()]
    if isinstance(value, pl.Series):
        return [_normalize(v) for v in value.to_list()]
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, int | float | str | bool):
        return _normalize(value.item())  # numpy scalars
    return value


def reference_sha(question: Question) -> str:
    return hashlib.sha256(question.reference_code.encode()).hexdigest()[:12]


def compute_expected(question: Question, lf: pl.LazyFrame) -> tuple[object, float]:
    """Execute reference code; return (normalized expected, elapsed seconds)."""
    ns = {"lf": lf, "pl": pl, "ols_slope": ols_slope}
    t0 = time.monotonic()
    try:
        exec(question.reference_code, ns)
    except Exception as e:
        raise GroundTruthError(f"{question.id}: reference code failed: {e}") from e
    elapsed = time.monotonic() - t0
    if "expected" not in ns:
        raise GroundTruthError(
            f"{question.id}: reference code never assigned `expected`"
        )
    return _normalize(ns["expected"]), elapsed


def _cache_path(identity: str) -> Path:
    return CACHE_DIR / f"{identity}.json"


def _load_cache(identity: str) -> dict:
    path = _cache_path(identity)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _store_cache(identity: str, cache: dict) -> None:
    path = _cache_path(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2, default=str))
    os.replace(tmp, path)


@functools.cache
def _identity_memo(resolved: str) -> str:
    return parquet_identity(resolved)


def get_expected(question: Question, parquet_path: Path | str, *, force: bool = False):
    """Cached ground truth for one question against one parquet."""
    identity = _identity_memo(str(Path(parquet_path).resolve()))
    cache = _load_cache(identity)
    entry = cache.get(question.id)
    if entry and not force and entry.get("reference_sha") == reference_sha(question):
        return entry["expected"]
    lf = _build_lazyframe(parquet_path)
    expected, elapsed = compute_expected(question, lf)
    logger.info("ground truth %s: %.1fs -> %r", question.id, elapsed, expected)
    cache[question.id] = {
        "expected": expected,
        "reference_sha": reference_sha(question),
        "elapsed_s": round(elapsed, 2),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _store_cache(identity, cache)
    return expected


def precompute(
    questions: list[Question], parquet_path: Path | str, *, force: bool = False
) -> dict:
    """Compute (or load) ground truth for all questions; the `gt` subcommand."""
    out = {}
    for q in questions:
        t0 = time.monotonic()
        out[q.id] = get_expected(q, parquet_path, force=force)
        logger.info(
            "%s [%s]%s: %.1fs total -> %r",
            q.id,
            q.category,
            " (expensive)" if q.expensive_gt else "",
            time.monotonic() - t0,
            out[q.id],
        )
    return out
