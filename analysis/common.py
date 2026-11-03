"""Shared loading, filtering, seeding and IO helpers for the MoltBook re-analysis.

Every step script imports from here so that the observation window, the
deduplication rule, the variant filters (autonomy audit, Jan 31 incident) and
the random seed are defined exactly once.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WORK = ROOT / "work"
HF_REPO = "SimulaMet/moltbook-observatory-archive"
REVISIONS = {
    "primary": {"sha": "ee98f6c8a6f2643d40bf852ced3bf7cac0417ca0",
                "dir": ROOT / "data_raw" / "moltbook-ee98f6c8",
                "note": "revision pinned by the repo's moltbook-observatory-archive gitlink (export of 2026-02-24); "
                        "the data the earlier run used"},
    "head": {"sha": "2bd2498ac0fc7d86148149e0f3d470d5c813538f",
             "dir": ROOT / "moltbook-observatory-archive",
             "note": "archive head on 2026-10-30; posts backfilled, most early comments no longer present"},
}
DATA_DIR = REVISIONS["primary"]["dir"] / "data"

WINDOW_START = "2026-01-28"
WINDOW_END = "2026-02-20"
INCIDENT_DATES = ["2026-01-30", "2026-01-31", "2026-02-01"]

VARIANTS = ("main", "no_autonomy", "no_incident", "merged")

TECHNICAL_KEYWORDS = [
    "error", "bug", "fix", "issue", "problem", "help", "solution", "solve",
    "debug", "crash", "exception", "fail", "broken", "stuck", "how to",
    "implement", "code", "function", "method", "class", "api", "library",
]

for d in (RESULTS, WORK):
    d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import igraph as ig
        ig.set_random_number_generator(random.Random(seed))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger(name)


def window_bounds(start: str = WINDOW_START, end: str = WINDOW_END) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return lo, hi


def _read_table_files(table: str, columns: list[str], data_dir: Path | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(str((data_dir or DATA_DIR) / table / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files for table {table} under {DATA_DIR}")
    frames = []
    for f in files:
        schema = pq.read_schema(f)
        cols = [c for c in columns if c in schema.names]
        df = pq.read_table(f, columns=cols).to_pandas()
        for c in columns:
            if c not in df.columns:
                df[c] = None
        for c in ("created_at", "first_seen_at", "last_seen_at"):
            if c in df.columns:
                df[c] = _to_naive_utc(df[c])
        frames.append(df[columns])
    return pd.concat(frames, ignore_index=True)


def _to_naive_utc(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.astype("datetime64[ns]")
    out = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return out.dt.tz_localize(None).astype("datetime64[ns]")


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "dump_date"], na_position="first", kind="mergesort")
    return df.drop_duplicates("id", keep="last").reset_index(drop=True)


def load_raw_window(revision: str = "primary", force: bool = False) -> dict[str, pd.DataFrame]:
    """Posts, comments and agents in the observation window, deduplicated by id.

    Cached to work/ because reading the parquet partitions takes minutes.
    """
    data_dir = REVISIONS[revision]["dir"] / "data"
    cache = {t: WORK / f"{t}_window_{revision}.parquet" for t in ("posts", "comments", "agents")}
    if not force and all(p.exists() for p in cache.values()):
        return {t: pd.read_parquet(p) for t, p in cache.items()}
    lo, hi = window_bounds()
    posts = _read_table_files("posts", ["id", "agent_id", "submolt", "title", "content",
                                        "comment_count", "created_at", "dump_date"], data_dir)
    posts = _dedupe(posts)
    posts = posts[(posts["created_at"] >= lo) & (posts["created_at"] < hi)].reset_index(drop=True)
    comments = _read_table_files("comments", ["id", "post_id", "agent_id", "parent_id", "content",
                                              "created_at", "dump_date"], data_dir)
    comments = _dedupe(comments)
    comments = comments[(comments["created_at"] >= lo) & (comments["created_at"] < hi)].reset_index(drop=True)
    agents = _read_table_files("agents", ["id", "name", "karma", "is_claimed", "first_seen_at",
                                          "last_seen_at", "created_at", "dump_date"], data_dir)
    agents = _dedupe(agents)
    for df in (posts, comments):
        df["agent_id"] = df["agent_id"].astype("string")
        df["content"] = df["content"].fillna("").astype(str)
    posts["title"] = posts["title"].fillna("").astype(str)
    posts["submolt"] = posts["submolt"].fillna("").astype(str)
    comments["parent_id"] = comments["parent_id"].astype("string")
    for t, df in (("posts", posts), ("comments", comments), ("agents", agents)):
        df.to_parquet(cache[t], index=False)
    return {"posts": posts, "comments": comments, "agents": agents}


def flagged_agents() -> set[str]:
    p = RESULTS / "autonomy_flagged_agents.csv"
    if not p.exists():
        raise FileNotFoundError("autonomy audit has not run yet (results/autonomy_flagged_agents.csv missing)")
    df = pd.read_csv(p, dtype={"agent_id": str})
    return set(df.loc[df["flagged"].astype(bool), "agent_id"])


def load_variant(variant: str = "main") -> dict[str, pd.DataFrame]:
    """Apply the variant filter used by the robustness reruns.

    main         full window
    no_autonomy  drop all posts and comments by agents flagged in the autonomy audit
    no_incident  drop all posts and comments created 2026-01-30 .. 2026-02-01 (UTC)
    merged       union of the pinned revision and the archive head, deduplicated by id
                 (latest dump_date wins); recovers the posts that most comments point to
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    d = load_raw_window("primary")
    if variant == "merged":
        h = load_raw_window("head")
        d = {t: _dedupe(pd.concat([d[t], h[t]], ignore_index=True)) for t in d}
    posts, comments, agents = d["posts"], d["comments"], d["agents"]
    if variant == "no_autonomy":
        bad = flagged_agents()
        posts = posts[~posts["agent_id"].isin(bad)].reset_index(drop=True)
        comments = comments[~comments["agent_id"].isin(bad)].reset_index(drop=True)
        agents = agents[~agents["id"].isin(bad)].reset_index(drop=True)
    elif variant == "no_incident":
        ex = set(INCIDENT_DATES)
        posts = posts[~posts["created_at"].dt.strftime("%Y-%m-%d").isin(ex)].reset_index(drop=True)
        comments = comments[~comments["created_at"].dt.strftime("%Y-%m-%d").isin(ex)].reset_index(drop=True)
    return {"posts": posts, "comments": comments, "agents": agents}


def population(d: dict[str, pd.DataFrame], variant: str = "main") -> pd.Index:
    """Agent population used for clustering and roles.

    Pinned revision: every agent in the revision's agents table, as in the earlier run
    (molt_dynamics.features.FeatureExtractor iterates storage.get_agents()).
    Head/merged data: the head agents table runs to September, so it is restricted to
    agents first seen before the end of the window, plus any agent active in the window.
    """
    agents = d["agents"]
    if variant in ("head", "merged"):
        _, hi = window_bounds()
        ids = set(agents.loc[agents["first_seen_at"] < hi, "id"].astype(str))
        ids |= set(pd.concat([d["posts"]["agent_id"], d["comments"]["agent_id"]]).dropna().astype(str))
    else:
        ids = set(agents["id"].astype(str))
    return pd.Index(np.sort(np.array(list(ids))), name="agent_id")


def suffix(variant: str) -> str:
    return "" if variant == "main" else f"_{variant}"


def to_jsonable(obj, path: str = ""):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj.tolist())]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v} at {path} cannot be written to results")
        return v
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, allow_nan=False)


def read_json(path: Path | str):
    with open(path) as f:
        return json.load(f)


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def cohens_d_two_sample(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp)
