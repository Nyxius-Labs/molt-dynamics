"""Step 1: download both archive revisions and write results/data_manifest.json."""

from __future__ import annotations

import glob
import hashlib
import os

import pyarrow.parquet as pq

from common import (HF_REPO, RESULTS, REVISIONS, WINDOW_END, WINDOW_START, get_logger,
                    load_raw_window, population, seed_everything, write_json)

log = get_logger("s01_data")


def download(rev: dict) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", revision=rev["sha"], local_dir=str(rev["dir"]),
                      max_workers=8)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_manifest(data_dir) -> dict:
    tables = {}
    for tdir in sorted(p for p in glob.glob(str(data_dir / "*")) if os.path.isdir(p)):
        files = sorted(glob.glob(os.path.join(tdir, "*.parquet")))
        agg = hashlib.sha256()
        rows = 0
        for f in files:
            rows += pq.read_metadata(f).num_rows
            agg.update(os.path.basename(f).encode())
            agg.update(sha256_file(f).encode())
        tables[os.path.basename(tdir)] = {"n_files": len(files), "raw_rows": rows,
                                          "sha256_of_file_hashes": agg.hexdigest()}
    return tables


def main() -> None:
    seed_everything()
    out = {"hf_repo": HF_REPO, "random_seed": 42, "primary_revision": "primary", "revisions": {}}
    for name, rev in REVISIONS.items():
        download(rev)
        tables = table_manifest(rev["dir"] / "data")
        d = load_raw_window(name, force=True)
        pop = population(d, "head" if name == "head" else "main")
        active = set(d["posts"]["agent_id"].astype(str)) | set(d["comments"]["agent_id"].astype(str))
        window = {
            "start_inclusive": WINDOW_START,
            "end_inclusive": WINDOW_END,
            "filter": "created_at (UTC) in [start 00:00, end+1 00:00); rows deduplicated by id keeping the latest dump_date",
            "posts": int(len(d["posts"])),
            "comments": int(len(d["comments"])),
            "comments_with_parent_id": int(d["comments"]["parent_id"].notna().sum()),
            "agents_table_unique_ids": int(len(d["agents"])),
            "agent_population": int(len(pop)),
            "active_agents_in_window": int(len(active)),
            "post_authors": int(d["posts"]["agent_id"].nunique()),
            "comment_authors": int(d["comments"]["agent_id"].nunique()),
            "submolts_with_posts": int(d["posts"]["submolt"].nunique()),
            "first_comment": d["comments"]["created_at"].min(),
            "last_comment": d["comments"]["created_at"].max(),
        }
        out["revisions"][name] = {"sha": rev["sha"], "local_dir": str(rev["dir"]), "note": rev["note"],
                                  "tables": tables, "window": window}
        log.info("%s window: %s", name, window)
    p_d = load_raw_window("primary")
    h_d = load_raw_window("head")
    tl = p_d["comments"][p_d["comments"]["parent_id"].isna()]
    union_posts = set(p_d["posts"]["id"]) | set(h_d["posts"]["id"])
    out["coverage"] = {
        "primary_window_comments_present_in_head": float(p_d["comments"]["id"].isin(set(h_d["comments"]["id"])).mean()),
        "primary_top_level_comments_with_post_in_primary": float(tl["post_id"].isin(set(p_d["posts"]["id"])).mean()),
        "primary_top_level_comments_with_post_in_primary_or_head_window": float(tl["post_id"].isin(union_posts).mean()),
        "head_window_posts": int(len(h_d["posts"])),
    }
    out["agent_population_definition"] = (
        "primary: all agents in the pinned revision's agents table (earlier-run definition); "
        "head: agents first seen before 2026-02-21 plus any agent active in the window")
    write_json(RESULTS / "data_manifest.json", out)


if __name__ == "__main__":
    main()
