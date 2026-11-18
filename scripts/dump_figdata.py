from __future__ import annotations

import glob
import json
import sys

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

sys.path.insert(0, "scripts")
import run_rigor as R
from salvage_classifier import features, REGEX, DD

COLS = ["cov", "log_n", "log_gap_h", "hour_entropy", "active_hours", "round_tick_frac", "gap_skew"]


def kimjo_A(r, n):
    sp, sm = np.sqrt(n + 1), np.sqrt(n - 1)
    return (sp * r - sm) / ((sp - 2) * r + sm)


def corrected_burst(posts, comments):
    ev = pl.concat([posts.select(["agent_id", "created_at"]),
                    comments.select(["agent_id", "created_at"])]).drop_nulls().sort(["agent_id", "created_at"])
    g = (ev.group_by("agent_id").agg(gaps=pl.col("created_at").diff().dt.total_seconds().drop_nulls())
         .filter(pl.col("gaps").list.len() >= 4))
    rows = []
    for aid, gaps in zip(g["agent_id"], g["gaps"]):
        a = np.array([x for x in gaps if x and x > 0], float)
        if len(a) < 4:
            continue
        m = a.mean()
        if m <= 0:
            continue
        rows.append((str(aid), float(kimjo_A(a.std() / m, len(a)))))
    return pl.DataFrame(rows, schema=["agent_id", "A_kimjo"], orient="row")


def downsample(x, y, k=45):
    idx = np.linspace(0, len(x) - 1, min(k, len(x))).astype(int)
    return [round(float(v), 4) for v in np.asarray(x)[idx]], [round(float(v), 4) for v in np.asarray(y)[idx]]


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)
    ptext = posts.select(["agent_id", (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    disc = set(pl.concat([ptext, ctext]).filter(pl.col("t").str.contains(REGEX))["agent_id"].unique().to_list())

    feat = features(posts, comments)
    ag = pl.scan_parquet(sorted(glob.glob(DD + "/agents/*.parquet"))).select(["id", "owner_x_handle"]).collect().with_columns(
        pl.col("id").cast(pl.Utf8).alias("agent_id"))
    cb = corrected_burst(posts, comments)
    d = feat.join(ag, on="agent_id", how="left").join(cb, on="agent_id", how="left").with_columns([
        pl.col("agent_id").is_in(list(disc)).alias("self_ai"),
        (pl.col("owner_x_handle").is_not_null() & (pl.col("owner_x_handle").cast(pl.Utf8).str.len_chars() > 0)).alias("has_x"),
    ])
    Xall = np.nan_to_num(d.select(COLS).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    nev = np.expm1(d["log_n"].to_numpy())
    self_ai = d["self_ai"].to_numpy()
    has_x = d["has_x"].to_numpy()
    cov = d["cov"].to_numpy()
    akimjo = np.nan_to_num(d["A_kimjo"].to_numpy().astype(float), nan=0.0)

    out = {}

    edges = np.linspace(0, 4, 33)
    ctr = (edges[:-1] + edges[1:]) / 2
    auto_cov = np.clip(cov[self_ai], 0, 3.999)
    hum_cov = np.clip(cov[has_x & ~self_ai], 0, 3.999)
    hda, _ = np.histogram(auto_cov, bins=edges, density=True)
    hdh, _ = np.histogram(hum_cov, bins=edges, density=True)
    out["cov_density"] = {"centers": [round(float(c), 3) for c in ctr],
                          "autonomous": [round(float(v), 4) for v in hda],
                          "human": [round(float(v), 4) for v in hdh],
                          "mean_auto": round(float(auto_cov.mean()), 3), "mean_human": round(float(hum_cov.mean()), 3)}

    train = self_ai | (has_x & ~self_ai)
    Xtr, ytr = Xall[train], self_ai[train].astype(int)
    burst_tr = akimjo[train]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(lr, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    f_m, t_m, _ = roc_curve(ytr, oof)
    f_b, t_b, _ = roc_curve(ytr, burst_tr)
    fm, tm = downsample(f_m, t_m); fb, tb = downsample(f_b, t_b)
    out["roc"] = {"multi_fpr": fm, "multi_tpr": tm, "multi_auc": round(float(roc_auc_score(ytr, oof)), 3),
                  "burst_fpr": fb, "burst_tpr": tb, "burst_auc": round(float(roc_auc_score(ytr, burst_tr)), 3),
                  "burst_assumed_auc": round(float(roc_auc_score(ytr, -burst_tr)), 3)}

    def pop_prob(est):
        p = np.full(len(Xall), np.nan)
        p[train] = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        est.fit(Xtr, ytr)
        p[~train] = est.predict_proba(Xall[~train])[:, 1]
        return p
    p_gb = pop_prob(GradientBoostingClassifier(random_state=42))
    p_lr = pop_prob(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))
    thr = np.round(np.arange(0.2, 0.81, 0.05), 2)
    share = lambda p, t: float(nev[p >= t].sum() / nev.sum())
    out["mix_sweep"] = {"thresholds": [float(t) for t in thr],
                        "gbm_activity": [round(share(p_gb, t), 3) for t in thr],
                        "lr_activity": [round(share(p_lr, t), 3) for t in thr]}

    import powerlaw
    casc, _ = R._cascade_sizes(posts)
    sizes = casc["n_agents"].to_numpy().astype(float)
    sizes = sizes[sizes >= 2]
    fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
    xmin, alpha = float(fit.power_law.xmin), float(fit.power_law.alpha)
    mu, sigma = float(fit.lognormal.mu), float(fit.lognormal.sigma)
    xs = np.unique(np.round(np.logspace(np.log10(2), np.log10(sizes.max()), 22))).astype(float)
    ccdf = np.array([(sizes >= x).mean() for x in xs])
    anchor = float((sizes >= xmin).mean())
    xt = xs[xs >= xmin]
    pl_ccdf = anchor * (xt / xmin) ** (-(alpha - 1.0))
    lnd = stats.lognorm(s=sigma, scale=np.exp(mu))
    ln_ccdf = anchor * lnd.sf(xt) / lnd.sf(xmin)
    out["cascade"] = {"n": int(len(sizes)), "xmin": xmin, "alpha": round(alpha, 3),
                      "x": [int(v) for v in xs],
                      "empirical_ccdf": [round(float(v), 5) for v in ccdf],
                      "x_tail": [int(v) for v in xt],
                      "powerlaw_ccdf": [round(float(v), 6) for v in pl_ccdf],
                      "lognormal_ccdf": [round(float(v), 6) for v in ln_ccdf]}

    with open("results/figdata.json", "w") as f:
        json.dump(out, f, indent=2)
    print("cov means auto/human:", out["cov_density"]["mean_auto"], out["cov_density"]["mean_human"])
    print("ROC AUC multi(logistic)/burst(+)/burst(assumed-):", out["roc"]["multi_auc"], out["roc"]["burst_auc"], out["roc"]["burst_assumed_auc"])
    print("mix activity range:", min(out["mix_sweep"]["gbm_activity"]+out["mix_sweep"]["lr_activity"]),
          "to", max(out["mix_sweep"]["gbm_activity"]+out["mix_sweep"]["lr_activity"]))
    print("cascade n:", out["cascade"]["n"])


if __name__ == "__main__":
    main()
