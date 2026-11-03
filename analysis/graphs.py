"""Interaction graph construction and exact clustering metrics shared across steps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_edges(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Directed reply edges, following molt_dynamics.dataset_loader._extract_interactions.

    A comment with a parent_id points from its author to the parent comment's author
    (no edge if the parent comment is not in the data); a top-level comment points to
    the post author. Self-replies are dropped. Weight = number of interactions.
    """
    post_auth = posts.set_index("id")["agent_id"]
    com_auth = comments.set_index("id")["agent_id"]
    has_parent = comments["parent_id"].notna() & (comments["parent_id"].astype(str) != "")
    dst = pd.Series(pd.NA, index=comments.index, dtype="string")
    dst[has_parent] = comments.loc[has_parent, "parent_id"].map(com_auth).astype("string")
    dst[~has_parent] = comments.loc[~has_parent, "post_id"].map(post_auth).astype("string")
    e = pd.DataFrame({"src": comments["agent_id"].astype("string"), "dst": dst,
                      "t": comments["created_at"], "kind": np.where(has_parent, "reply", "post_comment")})
    e = e.dropna(subset=["src", "dst"])
    e = e[e["src"] != e["dst"]]
    return e.reset_index(drop=True)


def weighted_edges(e: pd.DataFrame) -> pd.DataFrame:
    return e.groupby(["src", "dst"], sort=True).size().rename("weight").reset_index()


def build_igraph(w: pd.DataFrame, names: pd.Index | None = None):
    import igraph as ig
    if names is None:
        names = pd.Index(np.sort(pd.unique(pd.concat([w["src"], w["dst"]]).astype(str))))
    idx = pd.Series(np.arange(len(names)), index=names)
    g = ig.Graph(n=len(names), edges=list(zip(idx[w["src"].astype(str)].to_numpy(),
                                              idx[w["dst"].astype(str)].to_numpy())), directed=True)
    g.vs["name"] = list(names)
    g.es["weight"] = w["weight"].astype(float).tolist()
    return g


def _btw_chunk(g, sources):
    return np.asarray(g.betweenness(directed=True, sources=sources), dtype=float)


def exact_betweenness(g, n_jobs: int = 8) -> np.ndarray:
    """Exact unweighted directed betweenness, normalised as networkx (1/((n-1)(n-2)))."""
    from joblib import Parallel, delayed
    n = g.vcount()
    chunks = np.array_split(np.arange(n), n_jobs * 8)
    parts = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_btw_chunk)(g, c.tolist()) for c in chunks if len(c))
    b = np.sum(parts, axis=0)
    return b / ((n - 1) * (n - 2)) if n > 2 else b


def unique_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique rows, inverse index and multiplicity."""
    U, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return U, inv.ravel(), cnt


def weighted_silhouette(U: np.ndarray, w: np.ndarray, lab: np.ndarray, chunk: int = 2048) -> float:
    """Exact silhouette for a dataset given as unique rows U with multiplicities w.

    Identical to sklearn.metrics.silhouette_score on the expanded data: distances to
    duplicates of the same point are 0 and the own-cluster mean divides by n_c - 1;
    singleton clusters score 0.
    """
    keep = w > 0
    U, w, lab = U[keep], w[keep].astype(float), lab[keep]
    ks = np.unique(lab)
    if len(ks) < 2:
        raise ValueError("silhouette needs at least two clusters")
    lab_i = np.searchsorted(ks, lab)
    k = len(ks)
    W = np.zeros((len(U), k))
    W[np.arange(len(U)), lab_i] = w
    n_c = W.sum(axis=0)
    sq = (U ** 2).sum(axis=1)
    total = 0.0
    for s in range(0, len(U), chunk):
        A = U[s:s + chunk]
        d2 = (A ** 2).sum(axis=1)[:, None] + sq[None, :] - 2 * A @ U.T
        D = np.sqrt(np.maximum(d2, 0))
        S = D @ W
        own = lab_i[s:s + chunk]
        rows = np.arange(len(A))
        own_n = n_c[own]
        a = np.where(own_n > 1, S[rows, own] / np.maximum(own_n - 1, 1), 0.0)
        M = S / n_c[None, :]
        M[rows, own] = np.inf
        b = M.min(axis=1)
        sil = np.where(own_n > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
        sil = np.where((own_n > 1) & (np.maximum(a, b) == 0), 0.0, sil)
        total += float((sil * w[s:s + chunk]).sum())
    return total / float(w.sum())
