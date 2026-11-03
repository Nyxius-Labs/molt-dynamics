"""Step 10: human-platform comparison on r/programming (HuggingFaceGECLM/REDDIT_comments).

The comparison window is the most recent contiguous run of whole days whose number of
distinct comment authors first reaches the number of active MoltBook agents in the
main window. Steps 2 (structure) and 5 (cascade power law) are rerun with the same code:
edges are comment -> parent-comment author (submission authors are not in this dataset,
so top-level comments give no edge); [deleted] and AutoModerator are removed.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, SEED, get_logger, read_json, seed_everything, write_json
from graphs import build_igraph, exact_betweenness, unique_rows, weighted_silhouette
from ngram_engine import extract
from s02_structure import K_RANGE, KMEANS_N_INIT, NETWORK_FEATURES, kmeans_weighted, standardize
from s05_powerlaw import fit_sizes

log = get_logger("s10_human")

REPO = "HuggingFaceGECLM/REDDIT_comments"
REVISION = "54779d3d1f1c1b12e5989f695e13d38b394a558f"
SUBREDDIT = "programming"
LOCAL = ROOT / "data_raw" / "reddit_programming"
DROP_AUTHORS = {"[deleted]", "AutoModerator", ""}


def download() -> list[str]:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=str(LOCAL),
                      allow_patterns=[f"data/{SUBREDDIT}-*.parquet"], max_workers=8)
    return sorted(glob.glob(str(LOCAL / "data" / f"{SUBREDDIT}-*.parquet")))


def load(files: list[str]) -> pd.DataFrame:
    cols = ["id", "author", "body", "created_utc", "parent_id", "link_id"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df.drop_duplicates("id")
    df["t"] = pd.to_datetime(pd.to_numeric(df["created_utc"], errors="coerce"), unit="s")
    df = df.dropna(subset=["t"])
    df = df[~df["author"].fillna("").isin(DROP_AUTHORS)]
    df["body"] = df["body"].fillna("").astype(str)
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    return df.reset_index(drop=True)


def choose_window(df: pd.DataFrame, target_authors: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = df["t"].dt.floor("D")
    last = day.max()
    start = last
    while True:
        sub = df[(day >= start) & (day <= last)]
        if sub["author"].nunique() >= target_authors or start <= day.min():
            return start, last + pd.Timedelta(days=1)
        start -= pd.Timedelta(days=1)


def main() -> None:
    seed_everything()
    files = download()
    df = load(files)
    man = read_json(RESULTS / "data_manifest.json")
    target = man["revisions"]["primary"]["window"]["active_agents_in_window"]
    lo, hi = choose_window(df, target)
    w = df[(df["t"] >= lo) & (df["t"] < hi)].reset_index(drop=True)
    log.info("window %s .. %s: %d comments, %d authors", lo, hi, len(w), w["author"].nunique())

    auth = pd.Series(w["author"].to_numpy(), index="t1_" + w["id"].astype(str))
    reply = w["parent_id"].astype(str).str.startswith("t1_")
    e = pd.DataFrame({"src": w.loc[reply, "author"].to_numpy(), "dst": w.loc[reply, "parent_id"].map(auth).to_numpy()})
    e = e.dropna()
    e = e[e["src"] != e["dst"]]
    wg = e.groupby(["src", "dst"]).size().rename("weight").reset_index()
    g = build_igraph(wg)
    names = pd.Index(g.vs["name"])
    gf = pd.DataFrame({"in_degree": g.indegree(), "out_degree": g.outdegree(), "betweenness": exact_betweenness(g),
                       "clustering_coefficient": g.as_undirected(mode="collapse").transitivity_local_undirected(mode="zero"),
                       "pagerank": g.pagerank(damping=0.85, weights="weight", directed=True)}, index=names)
    pop = pd.Index(np.sort(w["author"].unique()))
    F = pd.DataFrame(index=pop).join(gf).fillna(0.0)
    Z, _, _ = standardize(F[NETWORK_FEATURES].to_numpy(float))
    U, inv, cnt = unique_rows(Z)
    sweep, labs = [], {}
    for k in K_RANGE:
        km = kmeans_weighted(U, cnt, k, KMEANS_N_INIT, SEED)
        sil = weighted_silhouette(U, cnt, km.labels_)
        sizes = np.bincount(km.labels_[inv], minlength=k)
        sweep.append({"k": k, "silhouette": sil, "largest_cluster_share": float(sizes.max() / sizes.sum())})
        labs[k] = km.labels_[inv]
    best = max(sweep, key=lambda r: r["silhouette"])
    lab = labs[best["k"]]
    Zdf = pd.DataFrame(Z, columns=NETWORK_FEATURES)
    prof = Zdf.groupby(lab).mean()
    periph = int(prof.sum(axis=1).idxmin())
    tot = F["in_degree"] + F["out_degree"]
    pd.DataFrame(sweep).to_csv(RESULTS / "human_platform_k_sweep.csv", index=False)

    tms = w["t"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    codes, uni = pd.factorize(w["author"], sort=True)
    ad, counts = extract(w["body"].tolist(), codes.astype(np.int64), tms, ns=(2, 3, 4, 5), min_keep=10, thresholds=(2, 5, 10))
    sizes = ad.groupby(["h", "n"])["agent"].size().to_numpy()
    pl, _ = fit_sizes(sizes)
    ad35 = ad[ad["n"] >= 3]
    sizes35 = ad35.groupby(["h", "n"])["agent"].size().to_numpy()
    pl35, _ = fit_sizes(sizes35)
    pd.DataFrame({"size": np.sort(sizes)[::-1]}).to_csv(RESULTS / "human_platform_cascade_sizes.csv", index=False)

    ms = read_json(RESULTS / "structure.json")
    mp = read_json(RESULTS / "powerlaw.json")["fits"]
    out = {
        "dataset": {"repo": REPO, "revision": REVISION, "subreddit": SUBREDDIT, "window_start": lo, "window_end_exclusive": hi,
                    "n_days": int((hi - lo).days), "n_comments": int(len(w)), "n_authors": int(len(pop)),
                    "selection": f"most recent whole days until distinct authors >= {target} (active MoltBook agents)",
                    "edges": "comment author -> parent comment author; top-level comments give no edge (submission authors unavailable)"},
        "graph": {"n_nodes": int(g.vcount()), "n_edges": int(g.ecount()), "n_interactions": int(wg["weight"].sum()),
                  "reciprocity": float(g.reciprocity())},
        "structure": {"k_sweep": sweep, "selected_k": int(best["k"]), "silhouette": float(best["silhouette"]),
                      "largest_cluster_share": float(best["largest_cluster_share"]),
                      "periphery_share_lowest_cluster": float((lab == periph).mean()),
                      "periphery_share_total_degree_le_1": float((tot <= 1).mean()),
                      "share_not_in_graph": float((~F.index.isin(names)).mean())},
        "cascades": {"counts_by_n": counts, "n_cascades_n2to5": int(len(sizes)), "n_cascades_n3to5": int(len(sizes35))},
        "powerlaw_n2to5": pl,
        "powerlaw_n3to5": pl35,
        "comparison": {
            "moltbook_main": {"silhouette": ms["silhouette"], "selected_k": ms["selected_k"],
                              "periphery_share_lowest_cluster": ms["periphery_share_lowest_cluster"],
                              "periphery_share_total_degree_le_1": ms["periphery_share_total_degree_le_1"],
                              "alpha_primary_n2to5": mp["primary_uncapped_n2to5"]["alpha"],
                              "alpha_uncapped_n3to5": mp["uncapped_n3to5"]["alpha"]},
            "reddit_programming": {"silhouette": float(best["silhouette"]), "selected_k": int(best["k"]),
                                   "periphery_share_lowest_cluster": float((lab == periph).mean()),
                                   "periphery_share_total_degree_le_1": float((tot <= 1).mean()),
                                   "alpha_n2to5": pl["alpha"], "alpha_n3to5": pl35["alpha"]},
            "caveat": (f"the Reddit window spans {int((hi - lo).days)} days (the dataset samples r/programming sparsely) against MoltBook's 24, "
                       "so the platforms are matched on population size, not duration; MoltBook edges include comments on posts while Reddit "
                       "edges are comment replies only; the MoltBook population includes inactive agents from the agents table while the "
                       "Reddit population is active commenters"),
        },
    }
    write_json(RESULTS / "human_platform.json", out)
    log.info("reddit: k=%d silhouette=%.3f periphery=%.3f alpha=%.3f", best["k"], best["silhouette"],
             (lab == periph).mean(), pl["alpha"])


if __name__ == "__main__":
    main()
