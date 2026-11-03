"""Step 2: interaction graph, five structural features, k-means k=3..8, bootstrap,
GMM comparison and the rule-based behavioral role overlay.

Reuses the feature definitions of molt_dynamics.features.FeatureExtractor and the
clustering/bootstrap/role logic of molt_dynamics.rq1_roles.RoleAnalyzer, reimplemented
on vectorised data so that it runs on the full population.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common import (RESULTS, SEED, get_logger, load_variant, population, seed_everything, suffix,
                    write_json)
from graphs import (build_edges, build_igraph, exact_betweenness, unique_rows, weighted_edges,
                    weighted_silhouette)

log = get_logger("s02_structure")

NETWORK_FEATURES = ["in_degree", "out_degree", "betweenness", "clustering_coefficient", "pagerank"]
K_RANGE = range(3, 9)
KMEANS_N_INIT = 100
N_BOOT = 100

ROLE_RULES = [
    ("Specialist", "normalized_entropy < -0.5 SD", [("normalized_entropy", "<", -0.5)]),
    ("Connector", "betweenness > 0.5 SD", [("betweenness", ">", 0.5)]),
    ("Initiator", "post_comment_ratio > 0.5 SD and total_posts > 0 SD (above mean)",
     [("post_comment_ratio", ">", 0.5), ("total_posts", ">", 0.0)]),
    ("Synthesizer", "post_comment_ratio < -0.5 SD and in_degree > 0 SD (above mean)",
     [("post_comment_ratio", "<", -0.5), ("in_degree", ">", 0.0)]),
    ("Generalist", "none of the above", []),
]


def agent_features(d: dict, pop: pd.Index) -> tuple[pd.DataFrame, dict]:
    posts, comments = d["posts"], d["comments"]
    e = build_edges(posts, comments)
    w = weighted_edges(e)
    g = build_igraph(w)
    names = pd.Index(g.vs["name"])
    log.info("graph: %d nodes, %d edges, %d interactions", g.vcount(), g.ecount(), int(w["weight"].sum()))

    pr = np.asarray(g.pagerank(damping=0.85, weights="weight", directed=True))
    clu = np.asarray(g.as_undirected(mode="collapse").transitivity_local_undirected(mode="zero"))
    log.info("computing exact betweenness on %d nodes", g.vcount())
    btw = exact_betweenness(g)
    gf = pd.DataFrame({"in_degree": g.indegree(), "out_degree": g.outdegree(), "betweenness": btw,
                       "clustering_coefficient": clu, "pagerank": pr}, index=names)

    f = pd.DataFrame(index=pop)
    f = f.join(gf).fillna(0.0)
    f["in_graph"] = f.index.isin(names)
    f["total_posts"] = posts.groupby("agent_id").size().reindex(pop, fill_value=0)
    f["total_comments"] = comments.groupby("agent_id").size().reindex(pop, fill_value=0)
    f["post_comment_ratio"] = np.where(f["total_comments"] > 0,
                                       f["total_posts"] / f["total_comments"].clip(lower=1),
                                       f["total_posts"].astype(float))
    sub = posts[posts["submolt"] != ""].groupby(["agent_id", "submolt"]).size().rename("c").reset_index()
    sub["p"] = sub["c"] / sub.groupby("agent_id")["c"].transform("sum")
    sub["h"] = -sub["p"] * np.log2(sub["p"])
    ent = sub.groupby("agent_id").agg(H=("h", "sum"), m=("submolt", "size"))
    ent["norm"] = ent["H"] / np.where(ent["m"] > 1, np.log2(ent["m"].clip(lower=2)), 1.0)
    f["topic_entropy"] = ent["H"].reindex(pop, fill_value=0.0)
    f["normalized_entropy"] = ent["norm"].reindex(pop, fill_value=0.0)
    gstats = {"n_nodes": g.vcount(), "n_edges": g.ecount(), "n_interactions": int(w["weight"].sum()),
              "n_reply_edges": int((e["kind"] == "reply").sum()),
              "n_post_comment_edges": int((e["kind"] == "post_comment").sum()),
              "reciprocity": float(g.reciprocity()),
              "edge_rule": "commenter -> parent-comment author (reply) or -> post author (top-level); "
                           "self-loops dropped; weight = interaction count; mentions not available in the archive"}
    return f, gstats


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def kmeans_weighted(U, w, k, n_init, seed):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    km.fit(U, sample_weight=w)
    return km


def boot_one(b, Z, k, ref_labels):
    rng = np.random.RandomState(SEED + b)
    n = len(Z)
    idx = rng.choice(n, size=n, replace=True)
    Xb = Z[idx]
    U, inv, cnt = unique_rows(Xb)
    km = KMeans(n_clusters=k, n_init=5, random_state=b).fit(U, sample_weight=cnt)
    lab_u = km.labels_
    sil = weighted_silhouette(U, cnt, lab_u)
    boot_lab = lab_u[inv]
    uniq_idx, first_pos = np.unique(idx, return_index=True)
    ari = adjusted_rand_score(ref_labels[uniq_idx], boot_lab[first_pos])
    return sil, ari


def interpret(profile_z: pd.Series, size_frac: float) -> str:
    hi = [f for f in NETWORK_FEATURES if profile_z[f] > 1.0]
    if not hi and size_frac > 0.3:
        return "Periphery: below-average on all five features"
    if not hi:
        top = profile_z.idxmax()
        return f"Moderate: highest relative {top}"
    names = {"in_degree": "receivers", "out_degree": "broadcasters", "betweenness": "brokers",
             "clustering_coefficient": "clique members", "pagerank": "high-prestige"}
    return "High " + ", ".join(f"{f} ({names[f]})" for f in hi)


def classify_roles(F: pd.DataFrame) -> tuple[pd.Series, dict]:
    cols = sorted({c for _, _, conds in ROLE_RULES for c, _, _ in conds})
    mu = F[cols].mean()
    sd = F[cols].std(ddof=0).replace(0, 1.0)
    Zr = (F[cols] - mu) / sd
    role = pd.Series("Generalist", index=F.index)
    assigned = pd.Series(False, index=F.index)
    thresholds = {}
    for name, desc, conds in ROLE_RULES:
        if not conds:
            thresholds[name] = {"definition": desc, "conditions": []}
            continue
        m = pd.Series(True, index=F.index)
        cl = []
        for c, op, z in conds:
            m &= (Zr[c] < z) if op == "<" else (Zr[c] > z)
            cl.append({"feature": c, "operator": op, "threshold_z": z,
                       "threshold_raw": float(mu[c] + z * sd[c]),
                       "feature_mean": float(mu[c]), "feature_sd": float(sd[c])})
        new = m & ~assigned
        role[new] = name
        assigned |= m
        thresholds[name] = {"definition": desc, "conditions": cl}
    return role, thresholds


def run(variant: str) -> dict:
    seed_everything()
    sfx = suffix(variant)
    d = load_variant(variant)
    pop = population(d, variant)
    F, gstats = agent_features(d, pop)
    X = F[NETWORK_FEATURES].to_numpy(float)
    Z, mu, sd = standardize(X)
    U, inv, cnt = unique_rows(Z)
    log.info("population %d agents, %d unique standardized feature rows", len(Z), len(U))

    sweep = []
    models = {}
    for k in K_RANGE:
        km = kmeans_weighted(U, cnt, k, KMEANS_N_INIT, SEED)
        sil = weighted_silhouette(U, cnt, km.labels_)
        sizes = np.bincount(km.labels_[inv], minlength=k)
        sweep.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_),
                      "largest_cluster_share": float(sizes.max() / sizes.sum()),
                      "n_singleton_clusters": int((sizes == 1).sum())})
        models[k] = km
        log.info("k=%d silhouette=%.4f largest=%.3f", k, sil, sizes.max() / sizes.sum())
    best = max(sweep, key=lambda r: r["silhouette"])
    k = best["k"]
    labels = models[k].labels_[inv]
    order = np.argsort(-np.bincount(labels, minlength=k))
    relabel = np.empty(k, int)
    relabel[order] = np.arange(k)
    labels = relabel[labels]

    from joblib import Parallel, delayed
    boots = Parallel(n_jobs=8, prefer="processes")(delayed(boot_one)(b, Z, k, labels) for b in range(N_BOOT))
    bs = np.array([b[0] for b in boots])
    ba = np.array([b[1] for b in boots])

    def summ(a):
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "ci_lower": float(np.percentile(a, 2.5)),
                "ci_upper": float(np.percentile(a, 97.5)), "min": float(a.min())}

    gmm = GaussianMixture(n_components=k, n_init=3, random_state=SEED).fit(Z)
    gl = gmm.predict(Z)
    gU_lab = np.zeros(len(U), int)
    gU_lab[inv] = gl
    resp = gmm.predict_proba(Z).max(axis=1)
    gmm_bic = {}
    for kk in K_RANGE:
        gm = GaussianMixture(n_components=kk, n_init=1, random_state=SEED).fit(Z)
        gmm_bic[kk] = float(gm.bic(Z))
    gmm_cmp = {"k": k, "silhouette": weighted_silhouette(U, cnt, gU_lab) if len(np.unique(gl)) > 1 else 0.0,
               "bic": float(gmm.bic(Z)), "aic": float(gmm.aic(Z)),
               "ari_vs_kmeans": float(adjusted_rand_score(labels, gl)),
               "mean_max_posterior": float(resp.mean()),
               "share_max_posterior_below_0.9": float((resp < 0.9).mean()),
               "largest_component_share": float(np.bincount(gl).max() / len(gl)),
               "bic_by_k": gmm_bic, "bic_selected_k": int(min(gmm_bic, key=gmm_bic.get))}

    F["cluster"] = labels
    prof_raw = F.groupby("cluster")[NETWORK_FEATURES].mean()
    Zdf = pd.DataFrame(Z, columns=NETWORK_FEATURES, index=F.index)
    Zdf["cluster"] = labels
    prof_z = Zdf.groupby("cluster")[NETWORK_FEATURES].mean()
    sizes = F["cluster"].value_counts().sort_index()
    tot_deg = F["in_degree"] + F["out_degree"]
    periph_cluster = int(prof_z.sum(axis=1).idxmin())
    clusters = []
    for c in sizes.index:
        clusters.append({"cluster": int(c), "count": int(sizes[c]), "proportion": float(sizes[c] / len(F)),
                         "interpretation": interpret(prof_z.loc[c], sizes[c] / len(F)),
                         **{f"mean_{f}": float(prof_raw.loc[c, f]) for f in NETWORK_FEATURES},
                         **{f"z_{f}": float(prof_z.loc[c, f]) for f in NETWORK_FEATURES}})
    pd.DataFrame(clusters).to_csv(RESULTS / f"table_network_clusters{sfx}.csv", index=False)
    pd.DataFrame(sweep).to_csv(RESULTS / f"kmeans_k_sweep{sfx}.csv", index=False)
    pd.DataFrame({"bootstrap": np.arange(N_BOOT), "silhouette": bs, "ari": ba}).to_csv(
        RESULTS / f"bootstrap_clustering{sfx}.csv", index=False)

    role, thresholds = classify_roles(F)
    F["role"] = role
    rc = role.value_counts()
    roles_tab = [{"role": r, "count": int(rc.get(r, 0)), "proportion": float(rc.get(r, 0) / len(F)),
                  "definition": thresholds[r]["definition"]} for r, _, _ in ROLE_RULES]
    pd.DataFrame(roles_tab).to_csv(RESULTS / f"table_behavioral_roles{sfx}.csv", index=False)

    out_cols = ["in_graph"] + NETWORK_FEATURES + ["total_posts", "total_comments", "post_comment_ratio",
                                                  "topic_entropy", "normalized_entropy", "cluster", "role"]
    F[out_cols].reset_index().to_csv(RESULTS / f"agents_features_clusters{sfx}.csv", index=False)

    res = {
        "variant": variant,
        "n_agents": int(len(F)),
        "n_agents_in_graph": int(F["in_graph"].sum()),
        "n_unique_feature_rows": int(len(U)),
        "graph": gstats,
        "features": NETWORK_FEATURES,
        "standardization": {"mean": dict(zip(NETWORK_FEATURES, mu)), "sd": dict(zip(NETWORK_FEATURES, sd))},
        "k_sweep": sweep,
        "selected_k": int(k),
        "silhouette": float(best["silhouette"]),
        "largest_cluster_share": float(sizes.max() / len(F)),
        "periphery_cluster": periph_cluster,
        "periphery_share_lowest_cluster": float(sizes[periph_cluster] / len(F)),
        "periphery_share_total_degree_le_1": float((tot_deg <= 1).mean()),
        "share_not_in_graph": float((~F["in_graph"]).mean()),
        "clusters": clusters,
        "bootstrap": {"n": N_BOOT, "kmeans_n_init": 5, "silhouette": summ(bs), "ari": summ(ba)},
        "gmm": gmm_cmp,
        "roles": roles_tab,
        "role_thresholds": thresholds,
        "notes": {
            "silhouette": "exact silhouette over all agents (computed on unique feature rows with multiplicity weights)",
            "kmeans": f"k-means n_init={KMEANS_N_INIT}, seed {SEED}, fitted on unique rows with multiplicity sample weights (same objective as the expanded data)",
        },
    }
    write_json(RESULTS / f"structure{sfx}.json", res)
    if variant == "main":
        write_json(RESULTS / "role_thresholds.json", {
            "source": "molt_dynamics.rq1_roles.RoleAnalyzer._classify_single_agent (rules applied in order; first match wins)",
            "standardization": "z-scores over the agent population (population SD)",
            "roles": thresholds,
            "role_names_in_paper": ["Generalist", "Initiator", "Amplifier", "Synthesizer", "Curator"],
            "role_names_in_code": [r for r, _, _ in ROLE_RULES],
            "periphery_definition": {"total_degree_le": 1},
            "kmeans": {"k_range": [3, 8], "n_init": KMEANS_N_INIT, "selection": "max silhouette"},
        })
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args()
    r = run(a.variant)
    log.info("k=%d silhouette=%.4f boot ARI=%.3f", r["selected_k"], r["silhouette"], r["bootstrap"]["ari"]["mean"])
