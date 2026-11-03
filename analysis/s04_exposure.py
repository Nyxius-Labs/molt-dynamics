"""Step 4: exposure, logistic adoption model and Cox models.

Exposure follows molt_dynamics.rq2_diffusion.DiffusionModeler.compute_exposures; see
results/exposure_definition.md for the exact rule. The logistic model is fitted on
cascade x exposure-count cells (sufficient statistics for the individual-level logit
with predictors 1, E, E^2), which gives the same estimates and the same
cascade-clustered sandwich as the individual-level model.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from common import (RESULTS, SEED, WORK, get_logger, load_variant, seed_everything, suffix,
                    wilson_interval, write_json)
from graphs import build_edges, weighted_edges

log = get_logger("s04_exposure")
warnings.filterwarnings("ignore")

N_COX = 100
MAX_EMIT = 40_000_000


class Graph:
    def __init__(self, w: pd.DataFrame, reverse: bool = False):
        names = pd.Index(np.sort(pd.unique(pd.concat([w["src"], w["dst"]]).astype(str))))
        s = names.get_indexer(w["src"].astype(str))
        t = names.get_indexer(w["dst"].astype(str))
        if reverse:
            s, t = t, s
        o = np.lexsort((t, s))
        s, t = s[o], t[o]
        self.names = names
        self.n = len(names)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(s, minlength=self.n))])
        self.indices = t


def prepare_adoptions(adopt: pd.DataFrame, G: Graph, cascade_ids: list[str]) -> pd.DataFrame:
    a = adopt[adopt["cascade_id"].isin(set(cascade_ids))].copy()
    cmap = pd.Series(np.arange(len(cascade_ids)), index=cascade_ids)
    a["c"] = a["cascade_id"].map(cmap).astype(np.int64)
    a["v"] = G.names.get_indexer(a["agent"].astype(str))
    a["tms"] = a["t"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    return a.sort_values(["c", "tms"], kind="mergesort").reset_index(drop=True)


def emit(G: Graph, c: np.ndarray, u: np.ndarray, tu: np.ndarray):
    """For each adoption (c, u, tu) by a graph node u, the nodes v that u exposes."""
    deg = G.indptr[u + 1] - G.indptr[u]
    tot = int(deg.sum())
    start = np.repeat(G.indptr[u], deg)
    within = np.arange(tot) - np.repeat(np.cumsum(deg) - deg, deg)
    return np.repeat(c, deg), G.indices[start + within], np.repeat(tu, deg)


def chunks_by_emission(a: pd.DataFrame, G: Graph):
    ing = a[a["v"] >= 0]
    deg = (G.indptr[ing["v"].to_numpy() + 1] - G.indptr[ing["v"].to_numpy()])
    per_c = pd.Series(deg).groupby(ing["c"].to_numpy()).sum()
    cs = per_c.index.to_numpy()
    cum = np.cumsum(per_c.to_numpy())
    bounds, lo = [], 0
    while lo < len(cs):
        base = cum[lo - 1] if lo else 0
        hi = int(np.searchsorted(cum, base + MAX_EMIT, side="right"))
        hi = max(hi, lo + 1)
        bounds.append(set(cs[lo:hi].tolist()))
        lo = hi
    return bounds


def exposure_cells(a: pd.DataFrame, G: Graph, n_cascades: int) -> pd.DataFrame:
    """Aggregated (cascade, exposure count) -> (n agents, n adopters) over all graph nodes."""
    N = G.n
    out = []
    all_cs = np.arange(n_cascades)
    for cset in chunks_by_emission(a, G) + [set()]:
        sub = a[a["c"].isin(cset)] if cset else a.iloc[0:0]
        if not cset:
            continue
        ing = sub[sub["v"] >= 0]
        c, u, tu = ing["c"].to_numpy(), ing["v"].to_numpy(), ing["tms"].to_numpy()
        ec, ev, et = emit(G, c, u, tu)
        akey = c * N + u
        o = np.argsort(akey)
        akey, atime = akey[o], tu[o]
        ekey = ec * N + ev
        pos = np.searchsorted(akey, ekey)
        pos_c = np.minimum(pos, len(akey) - 1)
        is_ad = (pos < len(akey)) & (akey[pos_c] == ekey)
        valid = ~is_ad | (et < atime[pos_c])
        k_keys, k_cnt = np.unique(ekey[valid], return_counts=True)
        p2 = np.searchsorted(k_keys, akey)
        p2c = np.minimum(p2, max(len(k_keys) - 1, 0))
        ad_exp = np.where((p2 < len(k_keys)) & (len(k_keys) > 0) & (k_keys[p2c] == akey), k_cnt[p2c], 0)
        ad_c = akey // N
        is_ad_k = np.isin(k_keys, akey, assume_unique=True)
        na_c = k_keys[~is_ad_k] // N
        na_k = k_cnt[~is_ad_k]
        df_ad = pd.DataFrame({"c": ad_c, "k": ad_exp, "n": 1, "y": 1})
        df_na = pd.DataFrame({"c": na_c, "k": na_k, "n": 1, "y": 0})
        cells = pd.concat([df_ad, df_na]).groupby(["c", "k"], as_index=False)[["n", "y"]].sum()
        touched = pd.concat([pd.Series(ad_c), pd.Series(na_c)]).value_counts()
        cs_arr = np.array(sorted(cset))
        zero = pd.DataFrame({"c": cs_arr, "k": 0, "n": N - touched.reindex(cs_arr, fill_value=0).to_numpy(), "y": 0})
        out.append(pd.concat([cells, zero]).groupby(["c", "k"], as_index=False)[["n", "y"]].sum())
    cells = pd.concat(out, ignore_index=True)
    missing = np.setdiff1d(all_cs, cells["c"].unique())
    if len(missing):
        cells = pd.concat([cells, pd.DataFrame({"c": missing, "k": 0, "n": N, "y": 0})], ignore_index=True)
    return cells


def fit_logit_cells(cells: pd.DataFrame) -> dict:
    k = cells["k"].to_numpy(float)
    n = cells["n"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    s1 = max(np.sqrt((n * k ** 2).sum() / n.sum()), 1e-12)
    s2 = max(np.sqrt((n * k ** 4).sum() / n.sum()), 1e-12)
    X = np.column_stack([np.ones_like(k), k / s1, k ** 2 / s2])

    def loglik(bb):
        e = X @ bb
        return float((y * e - n * np.logaddexp(0, e)).sum())

    beta = np.zeros(3)
    beta[0] = np.log(y.sum() / (n.sum() - y.sum()))
    cur = loglik(beta)
    for it in range(500):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - n * p)
        H = (X * (n * p * (1 - p))[:, None]).T @ X
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while loglik(beta + t * step) < cur - 1e-9 and t > 1e-8:
            t /= 2
        beta = beta + t * step
        new = loglik(beta)
        if abs(new - cur) < 1e-10 * max(1.0, abs(cur)) and np.max(np.abs(t * step)) < 1e-8:
            cur = new
            break
        cur = new
    p = 1 / (1 + np.exp(-(X @ beta)))
    H = (X * (n * p * (1 - p))[:, None]).T @ X
    B = np.linalg.inv(H)
    sc = X * (y - n * p)[:, None]
    S = pd.DataFrame(sc).groupby(cells["c"].to_numpy()).sum().to_numpy()
    G_ = S.shape[0]
    N_obs = n.sum()
    corr = G_ / (G_ - 1) * (N_obs - 1) / (N_obs - 3)
    V_cl = corr * B @ (S.T @ S) @ B
    D = np.diag([1, 1 / s1, 1 / s2])
    b = D @ beta
    V_cl = D @ V_cl @ D
    V_nv = D @ B @ D
    from scipy import stats
    ll = loglik(beta)
    p0 = y.sum() / N_obs
    ll0 = float(y.sum() * np.log(p0) + (N_obs - y.sum()) * np.log(1 - p0))
    names = ["intercept", "exposure", "exposure_squared"]
    coefs = {}
    for i, nm in enumerate(names):
        se = float(np.sqrt(V_cl[i, i]))
        z = b[i] / se
        coefs[nm] = {"coef": float(b[i]), "se_cluster_cascade": se, "z": float(z),
                     "p": float(2 * stats.norm.sf(abs(z))), "se_model_based": float(np.sqrt(V_nv[i, i])),
                     "ci95_cluster": [float(b[i] - 1.96 * se), float(b[i] + 1.96 * se)]}
    vertex = -b[1] / (2 * b[2]) if b[2] != 0 else float("inf")
    return {"coefficients": coefs, "n_observations": int(N_obs), "n_adopted": int(y.sum()),
            "n_clusters_cascades": int(G_), "log_likelihood": ll, "mcfadden_pseudo_r2": float(1 - ll / ll0),
            "iterations": it + 1, "curvature": "concave (saturating)" if b[2] < 0 else "convex (accelerating)",
            "vertex_exposure": float(vertex) if np.isfinite(vertex) else 1e308,
            "max_exposure_observed": int(k.max())}


def binned_rates(cells: pd.DataFrame) -> pd.DataFrame:
    edges = list(range(0, 21)) + [21, 31, 51, 101, 10 ** 9]
    labels = [str(i) for i in range(0, 21)] + ["21-30", "31-50", "51-100", "101+"]
    b = pd.cut(cells["k"], bins=[-0.5] + [e - 0.5 for e in edges[1:]], labels=labels)
    t = cells.groupby(b, observed=False)[["n", "y"]].sum().reset_index().rename(columns={"k": "exposure_bin"})
    t.columns = ["exposure_bin", "n_agent_cascade_pairs", "n_adopted"]
    t = t[t["n_agent_cascade_pairs"] > 0]
    lo, hi = wilson_interval(t["n_adopted"].to_numpy(), t["n_agent_cascade_pairs"].to_numpy())
    t["adoption_rate"] = t["n_adopted"] / t["n_agent_cascade_pairs"]
    t["wilson_lower"], t["wilson_upper"] = lo, hi
    return t


def static_cox_rows(a: pd.DataFrame, G: Graph, cascade_ids: list[str], code_style: bool) -> pd.DataFrame:
    """Per-agent rows as in fit_cox_hazards: adopters (duration from seed > 0, event 1) and
    exposed non-adopters (censored at the cascade's last adoption)."""
    N = G.n
    ing = a[a["v"] >= 0]
    c, u, tu = ing["c"].to_numpy(), ing["v"].to_numpy(), ing["tms"].to_numpy()
    ec, ev, et = emit(G, c, u, tu)
    akey = c * N + u
    o = np.argsort(akey)
    akey_s, atime_s = akey[o], tu[o]
    ekey = ec * N + ev
    pos = np.minimum(np.searchsorted(akey_s, ekey), len(akey_s) - 1)
    is_ad = akey_s[pos] == ekey
    valid = ~is_ad | (et < atime_s[pos])
    kk, kc = np.unique(ekey[valid], return_counts=True)
    expo = pd.Series(kc, index=kk)
    seed = a.groupby("c")["tms"].min()
    last = a.groupby("c")["tms"].max()
    ad = pd.DataFrame({"c": c, "v": u, "t": tu})
    ad["exposure"] = expo.reindex(ad["c"] * N + ad["v"], fill_value=0).to_numpy()
    ad["duration"] = (ad["t"] - ad["c"].map(seed)) / 3.6e6
    ad["event"] = 1
    ad = ad[ad["duration"] > 0]
    na_keys = kk[~np.isin(kk, akey)]
    na = pd.DataFrame({"c": na_keys // N, "v": na_keys % N, "exposure": kc[~np.isin(kk, akey)]})
    na["duration"] = (na["c"].map(last) - na["c"].map(seed)) / 3.6e6
    na["event"] = 0
    na = na[na["duration"] > 0]
    rows = pd.concat([ad[["c", "v", "exposure", "duration", "event"]], na[["c", "v", "exposure", "duration", "event"]]])
    return rows.reset_index(drop=True)


def tv_cox_rows(a: pd.DataFrame, G: Graph) -> pd.DataFrame:
    """Counting-process rows with cumulative exposure as a time-varying covariate.

    Clock: hours since the cascade's seed. Every graph node is at risk from the seed until
    it adopts (event) or the cascade's last adoption (censored); its covariate is the
    number of distinct in-neighbours that have adopted so far. Seed-time adopters are
    excluded. Identical rows (never-exposed non-adopters) are collapsed with weights.
    """
    N = G.n
    ing = a[a["v"] >= 0]
    c, u, tu = ing["c"].to_numpy(), ing["v"].to_numpy(), ing["tms"].to_numpy()
    ec, ev, et = emit(G, c, u, tu)
    seed = a.groupby("c")["tms"].min()
    last = a.groupby("c")["tms"].max()
    adt = pd.Series(tu, index=c * N + u)
    ex = pd.DataFrame({"key": ec * N + ev, "t": et})
    ex["stop_t"] = adt.reindex(ex["key"]).to_numpy()
    ex = ex[ex["stop_t"].isna() | (ex["t"] < ex["stop_t"])]
    ex = ex.groupby(["key", "t"]).size().rename("dk").reset_index().sort_values(["key", "t"])
    ex["cum"] = ex.groupby("key")["dk"].cumsum()
    keys_exposed = ex["key"].unique()
    subj = pd.DataFrame({"key": np.union1d(keys_exposed, adt.index.to_numpy())})
    subj["c"] = subj["key"] // N
    subj["t0"] = subj["c"].map(seed)
    subj["end"] = adt.reindex(subj["key"]).to_numpy()
    subj["event"] = subj["end"].notna().astype(int)
    subj["end"] = subj["end"].fillna(subj["c"].map(last))
    subj = subj[subj["end"] > subj["t0"]]
    br = ex.merge(subj[["key", "t0", "end"]], on="key")
    br = br[br["t"] < br["end"]]
    br["t"] = br["t"].clip(lower=br["t0"])
    br = br.groupby("key").agg(ts=("t", list), cums=("cum", list))
    rows = []
    for key, t0, end, ev_ in subj[["key", "t0", "end", "event"]].itertuples(index=False):
        cut = [t0]
        vals = [0]
        if key in br.index:
            for t, cm in zip(br.at[key, "ts"], br.at[key, "cums"]):
                if t == cut[-1]:
                    vals[-1] = cm
                else:
                    cut.append(t)
                    vals.append(cm)
        cut.append(end)
        for j in range(len(vals)):
            rows.append((key // N, (cut[j] - t0) / 3.6e6, (cut[j + 1] - t0) / 3.6e6, vals[j],
                         int(ev_ and j == len(vals) - 1)))
    df = pd.DataFrame(rows, columns=["c", "start", "stop", "exposure", "event"])
    nodes_c = a.groupby("c")["c"].size().index
    touched = subj.groupby("c").size().reindex(nodes_c, fill_value=0)
    ad_in_graph_seed = a[(a["v"] >= 0)].groupby("c").apply(lambda g: int((g["tms"] == g["tms"].min()).sum()))
    never = pd.DataFrame({"c": nodes_c, "start": 0.0,
                          "stop": ((last - seed) / 3.6e6).reindex(nodes_c).to_numpy(),
                          "exposure": 0, "event": 0,
                          "weight": (N - touched.to_numpy() - ad_in_graph_seed.reindex(nodes_c, fill_value=0).to_numpy())})
    df["weight"] = 1
    df = pd.concat([df, never[never["stop"] > 0]], ignore_index=True)
    df = df.groupby(["c", "start", "stop", "exposure", "event"], as_index=False)["weight"].sum()
    return df[df["stop"] > df["start"]]


def cox_summary(fitter, label: str, extra: dict) -> dict:
    s = fitter.summary.loc["exposure"]
    return {"model": label, "coef": float(s["coef"]), "se": float(s["se(coef)"]),
            "hazard_ratio": float(s["exp(coef)"]), "hr_ci95": [float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])],
            "z": float(s["z"]), "p": float(s["p"]), **extra}


def fit_static(rows: pd.DataFrame, strata: bool, cluster: bool, label: str) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    df = rows[["exposure", "duration", "event", "c"]].copy()
    cph = CoxPHFitter()
    kw = {}
    if strata:
        kw["strata"] = ["c"]
    elif cluster:
        kw["cluster_col"] = "c"
    else:
        df = df.drop(columns="c")
    cph.fit(df, duration_col="duration", event_col="event", **kw)
    ph = proportional_hazard_test(cph, df, time_transform="rank")
    extra = {"n_rows": int(len(rows)), "n_events": int(rows["event"].sum()), "n_cascades": int(rows["c"].nunique()),
             "concordance": float(cph.concordance_index_),
             "schoenfeld_test": {"time_transform": "rank", "statistic": float(ph.summary.loc["exposure", "test_statistic"]),
                                 "p": float(ph.summary.loc["exposure", "p"])},
             "se_type": "robust, clustered by cascade" if cluster and not strata else "model-based",
             "strata": "cascade" if strata else "unstratified"}
    return cox_summary(cph, label, extra)


def fit_tv(df: pd.DataFrame, strata: bool, label: str) -> dict:
    from lifelines import CoxTimeVaryingFitter
    d = df.copy()
    d["id"] = np.arange(len(d))
    cols = ["id", "start", "stop", "exposure", "event", "weight"] + (["c"] if strata else [])
    ctv, used = None, None
    for step, pen in ((0.2, 0.0), (0.05, 0.0), (0.05, 0.01)):
        try:
            cand = CoxTimeVaryingFitter(penalizer=pen)
            cand.fit(d[cols], id_col="id", event_col="event", start_col="start", stop_col="stop",
                     weights_col="weight", strata=["c"] if strata else None, fit_options={"step_size": step})
        except Exception:
            continue
        s = cand.summary.loc["exposure"]
        if np.isfinite(s["se(coef)"]) and np.isfinite(s["p"]) and abs(s["coef"]) < 10:
            ctv, used = cand, {"newton_step_size": step, "penalizer": pen}
            break
    if ctv is None:
        raise RuntimeError(f"time-varying Cox did not converge: {label}")
    extra = {"n_rows": int(len(d)), "n_subject_weight": int(d["weight"].sum()),
             "n_events": int(d.loc[d["event"] == 1, "weight"].sum()), "n_cascades": int(d["c"].nunique()),
             "se_type": "model-based (frequency weights for collapsed never-exposed rows)",
             **used,
             "strata": "cascade" if strata else "unstratified",
             "frailty": "not fitted: no shared-frailty Cox implementation in lifelines/statsmodels; stratification by cascade used instead"}
    return cox_summary(ctv, label, extra)


def run(variant: str, alt_direction: bool = True) -> dict:
    seed_everything()
    sfx = suffix(variant)
    d = load_variant(variant)
    w = weighted_edges(build_edges(d["posts"], d["comments"]))
    G = Graph(w)
    casc = pd.read_csv(RESULTS / f"cascades{sfx}.csv")
    adopt = pd.read_parquet(WORK / f"adoptions{sfx}.parquet")
    meme = casc["cascade_type"] == "meme"
    repro = casc[(meme & casc["in_cap_n3to5"] & (casc["n"] >= 3)) | ~meme]
    repro_ids = (repro[repro["cascade_type"] == "meme"].sort_values(["size", "seed_time"], ascending=[False, True])["cascade_id"].tolist()
                 + repro.loc[repro["cascade_type"] != "meme", "cascade_id"].tolist())
    prim_ids = casc["cascade_id"].tolist()
    res = {"variant": variant, "graph_nodes": G.n, "graph_edges": int(len(w))}

    sets = {"primary_uncapped_n2to5": prim_ids}
    if variant == "main":
        sets["reproduction_capped_n3to5"] = repro_ids
    logit = {}
    for name, ids in sets.items():
        a = prepare_adoptions(adopt, G, ids)
        cells = exposure_cells(a, G, len(ids))
        logit[name] = fit_logit_cells(cells)
        logit[name]["n_cascades"] = len(ids)
        if name == "primary_uncapped_n2to5":
            bins = binned_rates(cells)
            bins.to_csv(RESULTS / f"figure3_binned_adoption{sfx}.csv", index=False)
            cf = logit[name]["coefficients"]
            kg = np.arange(0, 101)
            eta = cf["intercept"]["coef"] + cf["exposure"]["coef"] * kg + cf["exposure_squared"]["coef"] * kg ** 2
            pd.DataFrame({"exposure": kg, "logistic_predicted_rate": 1 / (1 + np.exp(-eta))}).to_csv(
                RESULTS / f"figure3_logistic_curve{sfx}.csv", index=False)
            res["binned_adoption"] = bins.to_dict("records")
        log.info("logit %s: %s", name, {k: round(v["coef"], 5) for k, v in logit[name]["coefficients"].items()})
    if alt_direction and variant == "main":
        Gr = Graph(w, reverse=True)
        a = prepare_adoptions(adopt, Gr, prim_ids)
        logit["primary_reader_direction"] = fit_logit_cells(exposure_cells(a, Gr, len(prim_ids)))
        logit["primary_reader_direction"]["note"] = ("exposure counted from agents the focal agent commented on "
                                                     "(successors), i.e. content it plausibly read")
    res["logistic"] = logit

    cox = {}
    rng = np.random.RandomState(SEED)
    rand_ids = list(rng.choice(prim_ids, size=min(N_COX, len(prim_ids)), replace=False))
    a_r = prepare_adoptions(adopt, G, rand_ids)
    srows = static_cox_rows(a_r, G, rand_ids, code_style=True)
    cox["random100_static_pooled_clustered"] = fit_static(srows, strata=False, cluster=True,
                                                          label="static exposure, pooled, SE clustered by cascade")
    cox["random100_static_stratified"] = fit_static(srows, strata=True, cluster=False,
                                                    label="static exposure, stratified by cascade")
    tv = tv_cox_rows(a_r, G)
    cox["random100_timevarying_pooled"] = fit_tv(tv, strata=False, label="time-varying cumulative exposure, pooled")
    cox["random100_timevarying_stratified"] = fit_tv(tv, strata=True, label="time-varying cumulative exposure, stratified by cascade")
    if variant == "main":
        top_ids = repro_ids[:N_COX]
        a_t = prepare_adoptions(adopt, G, top_ids)
        trows = static_cox_rows(a_t, G, top_ids, code_style=True)
        cox["reproduction_top100_static"] = fit_static(trows, strata=False, cluster=False,
                                                       label="earlier-run procedure: first 100 cascades of the capped list (largest), static exposure")
        cox["reproduction_top100_static_stratified"] = fit_static(trows, strata=True, cluster=False,
                                                                  label="earlier-run cascades, stratified by cascade")
    pd.DataFrame({"cascade_id": rand_ids}).to_csv(RESULTS / f"cox_sampled_cascades{sfx}.csv", index=False)
    for k_, v in cox.items():
        log.info("cox %s: HR=%.3f p=%.3g", k_, v["hazard_ratio"], v["p"])
    res["cox"] = cox
    rows = []
    for name, L in logit.items():
        for pred, cf in L["coefficients"].items():
            rows.append({"model": f"logistic[{name}]", "predictor": pred, "coefficient": cf["coef"],
                         "se": cf["se_cluster_cascade"], "se_type": "cluster(cascade)", "p_value": cf["p"]})
    for name, C in cox.items():
        rows.append({"model": f"cox[{name}]", "predictor": "exposure", "coefficient": C["coef"], "se": C["se"],
                     "se_type": C["se_type"], "p_value": C["p"], "hazard_ratio": C["hazard_ratio"]})
    pd.DataFrame(rows).to_csv(RESULTS / f"table_adoption_models{sfx}.csv", index=False)
    write_json(RESULTS / f"exposure_models{sfx}.json", res)
    if variant == "main":
        (RESULTS / "exposure_definition.md").write_text(EXPOSURE_DEFINITION)
    return res


EXPOSURE_DEFINITION = """# Exposure definition used in the code

Implemented in `analysis/s04_exposure.py` (`exposure_cells`, `static_cox_rows`), which reproduces
`DiffusionModeler.compute_exposures` in `src/molt_dynamics/rq2_diffusion.py` (verified cell-for-cell
against a direct port of that loop on 43 cascades).

## Network

A directed, static graph over every agent that sent or received at least one interaction in the window.
There is an edge `u -> v` if `u` wrote a top-level comment on a post by `v`, or replied to a comment by `v`
(`analysis/graphs.py::build_edges`, the rule in `dataset_loader._extract_interactions`). Self-interactions
are dropped. The graph pools the whole window: an edge exists if the interaction happened at any time,
before or after the cascade. Edge weights (interaction counts) are not used for exposure.

## Adoption

Agent `a` adopts cascade `c` at `t_a(c)`, the timestamp of its first post or comment containing the
cascade's n-gram (meme), identical code block (skill) or formatting pattern (behavioral).

## What counts as one exposure

For a focal agent `v` and cascade `c`, each **distinct in-neighbour** `u` of `v` (an agent that has
commented on or replied to `v`) that adopted `c` counts as one exposure. A neighbour counts once,
however many times it used the n-gram and however many interactions it has with `v`.

## Over what window

* If `v` adopted `c`: only neighbours that adopted **strictly before** `v` (`t_u(c) < t_v(c)`). Ties do not count.
* If `v` never adopted `c`: **every** adopting neighbour over the whole observation window counts, with no time limit.
* Units of analysis: every node of the graph x every cascade. Agents with no interactions are not in the
  risk set, and adopters outside the graph are dropped.

## Repeated exposures from the same source

A neighbour is counted at most once per cascade. Repeated adoptions, repeated posts and multiple
interactions between the same pair are ignored.

## Consequences

The rule is asymmetric. Adopters are credited with exposures up to their adoption time, while non-adopters
are credited with exposures from the whole window. With a static graph this mechanically makes
non-adopters look more exposed, which biases the logistic quadratic term and the static Cox hazard
ratio towards "saturating" (HR < 1). The edge direction means that exposure flows from commenters to the
author they commented on, not from authors to their readers.

## Variants reported alongside

* **Time-varying Cox** (`tv_cox_rows`): cumulative count of distinct in-neighbours that have adopted by
  time `t`, with the clock starting at the cascade seed. Every graph node is at risk until it adopts or until
  the cascade's last adoption. This removes the asymmetry.
* **Reader direction** (logistic sensitivity): `v` is exposed by agents whose content `v` commented on
  (out-neighbours), i.e. content `v` plausibly read.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args()
    run(a.variant)
