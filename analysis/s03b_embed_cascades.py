"""Step 3b: embedding-based cascade extraction (robustness check for the n-gram cascades).

Items (post and comment contents, as in s03) are embedded with
sentence-transformers/all-MiniLM-L6-v2 (max 128 word pieces, unit-normalised). Two
items are linked when their cosine similarity is >= tau among each item's 20 nearest
neighbours (FAISS HNSW, inner product); a cascade is a connected component whose items
were written by >= 10 distinct agents, matching the n-gram adopter threshold.

tau is chosen on a held-out calibration sample of items: candidate pairs (each item's
10 nearest neighbours in the sample plus 5 random partners) are labelled by lexical
near-duplication (Jaccard of word 3-shingles >= 0.5 positive, < 0.2 negative, between
excluded) and tau maximises F1. A disjoint test sample reports precision/recall at tau.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import RESULTS, SEED, WORK, get_logger, load_variant, seed_everything, write_json
from s03_cascades import MEME_MIN_ADOPTERS, items_table

log = get_logger("s03b_embed")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ = 128
KNN = 20
CAL_N = 4000
TAU_GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)
WORD_RE = re.compile(r"\b\w+\b")


def norm_text(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def embed(texts: list[str]) -> np.ndarray:
    cache = WORK / "embeddings_main.npy"
    if cache.exists():
        e = np.load(cache)
        if len(e) == len(texts):
            return e
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    m = SentenceTransformer(MODEL, device="cpu")
    m.max_seq_length = MAX_SEQ
    e = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True).astype(np.float32)
    np.save(cache, e)
    return e


def shingles(t: str) -> set:
    w = WORD_RE.findall(t)
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def labelled_pairs(idx: np.ndarray, E: np.ndarray, texts: list[str], rng) -> pd.DataFrame:
    sub = E[idx]
    S = sub @ sub.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :10]
    pairs = {(min(i, j), max(i, j)) for i in range(len(idx)) for j in nn[i]}
    for i in range(len(idx)):
        for j in rng.choice(len(idx), 5, replace=False):
            if i != j:
                pairs.add((min(i, int(j)), max(i, int(j))))
    sh = [shingles(texts[k]) for k in idx]
    rows = []
    for i, j in pairs:
        a, b = sh[i], sh[j]
        jac = len(a & b) / len(a | b) if (a or b) else 0.0
        rows.append((i, j, float(S[i, j]), jac))
    df = pd.DataFrame(rows, columns=["i", "j", "cos", "jaccard"])
    df = df[(df["jaccard"] >= 0.5) | (df["jaccard"] < 0.2)].copy()
    df["pos"] = df["jaccard"] >= 0.5
    return df


def prf(df: pd.DataFrame, tau: float) -> dict:
    pred = df["cos"] >= tau
    tp = int((pred & df["pos"]).sum())
    fp = int((pred & ~df["pos"]).sum())
    fn = int((~pred & df["pos"]).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tau": float(tau), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def components(E: np.ndarray, tau: float, I: np.ndarray, D: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(E)
    rows = np.repeat(np.arange(n), I.shape[1])
    cols = I.ravel()
    keep = (D.ravel() >= tau) & (cols >= 0)
    A = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def cascade_sizes(comp_item: np.ndarray, it: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"comp": comp_item, "agent": it["agent_id"].to_numpy(), "t": it["created_at"].to_numpy()})
    g = df.groupby("comp").agg(size=("agent", "nunique"), n_items=("agent", "size"), seed_time=("t", "min"))
    return g[g["size"] >= MEME_MIN_ADOPTERS].reset_index()


def main() -> None:
    seed_everything()
    import faiss
    faiss.omp_set_num_threads(8)
    d = load_variant("main")
    it = items_table(d)
    key = norm_text(it["content"])
    codes, uniq = pd.factorize(key)
    texts = list(uniq)
    log.info("%d items, %d unique normalised texts", len(it), len(texts))
    E = embed(texts)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(texts))
    cal_idx, test_idx = perm[:CAL_N], perm[CAL_N:2 * CAL_N]
    cal = labelled_pairs(cal_idx, E, texts, rng)
    test = labelled_pairs(test_idx, E, texts, rng)
    curve = [prf(cal, t) for t in TAU_GRID]
    tau = max(curve, key=lambda r: (r["f1"], r["tau"]))["tau"]
    log.info("calibration pairs %d (%d positive); tau=%.2f", len(cal), int(cal["pos"].sum()), tau)

    index = faiss.IndexHNSWFlat(E.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.add(E)
    index.hnsw.efSearch = 64
    D, I = index.search(E, KNN + 1)

    runs = {}
    for t in sorted({tau, round(tau - 0.05, 2), round(tau + 0.05, 2)}):
        if t >= 1.0:
            continue
        comp = components(E, t, I, D)
        sizes = cascade_sizes(comp[codes], it)
        runs[f"{t:.2f}"] = {"tau": t, "n_cascades": int(len(sizes)),
                            "largest_component_item_share": float(np.bincount(comp[codes]).max() / len(it)),
                            "size_summary": sizes["size"].describe().to_dict()}
        if t == tau:
            sizes.rename(columns={"comp": "component"}).to_csv(RESULTS / "embedding_cascades.csv", index=False)
        log.info("tau=%.2f -> %d cascades", t, len(sizes))
    pd.DataFrame(curve).to_csv(RESULTS / "embedding_threshold_calibration.csv", index=False)
    write_json(RESULTS / "embedding_cascades.json", {
        "model": MODEL, "max_seq_length": MAX_SEQ, "knn": KNN, "index": "faiss HNSW32 inner product",
        "n_items": int(len(it)), "n_unique_texts": int(len(texts)),
        "calibration": {"n_items": CAL_N, "n_pairs": int(len(cal)), "n_positive": int(cal["pos"].sum()),
                        "label": "word 3-shingle Jaccard >= 0.5 positive, < 0.2 negative",
                        "selected_tau": float(tau), "at_tau": prf(cal, tau)},
        "test": {"n_items": CAL_N, "n_pairs": int(len(test)), "n_positive": int(test["pos"].sum()),
                 "at_tau": prf(test, tau)},
        "min_distinct_adopters": MEME_MIN_ADOPTERS,
        "selected": runs[f"{tau:.2f}"],
        "tau_sensitivity": runs,
    })


if __name__ == "__main__":
    main()
