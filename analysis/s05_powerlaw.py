"""Step 5: Clauset-style power-law fits of cascade sizes (distinct adopters).

Uses powerlaw.Fit(discrete=True) as molt_dynamics.rq2_diffusion.CascadeAnalyzer.test_power_law
did, and adds normalised likelihood-ratio tests and a parametric-bootstrap goodness of fit.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, SEED, get_logger, seed_everything, suffix, write_json

log = get_logger("s05_powerlaw")
warnings.filterwarnings("ignore")


def gof_pvalue(sizes: np.ndarray, fit, n_iter: int, rng) -> dict:
    import powerlaw
    xmin, alpha = fit.power_law.xmin, fit.power_law.alpha
    d_obs = float(fit.power_law.D)
    body = sizes[sizes < xmin]
    n = len(sizes)
    ptail = float((sizes >= xmin).mean())
    hits = 0
    for _ in range(n_iter):
        nt = rng.binomial(n, ptail)
        u = rng.uniform(size=nt)
        tail = np.floor((xmin - 0.5) * (1 - u) ** (-1 / (alpha - 1)) + 0.5)
        syn = np.concatenate([tail, rng.choice(body, n - nt, replace=True) if len(body) else []])
        f = powerlaw.Fit(syn, discrete=True, verbose=False)
        hits += int(f.power_law.D >= d_obs)
    return {"gof_p": hits / n_iter, "gof_iters": n_iter,
            "tail_sampler": "continuous power law rounded to integers (Clauset et al. 2009, App. D approximation)"}


def fit_sizes(sizes: np.ndarray, gof_iters: int = 0) -> dict:
    import powerlaw
    sizes = np.asarray(sizes, float)
    sizes = sizes[sizes > 0]
    fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
    a, s, xmin = fit.power_law.alpha, fit.power_law.sigma, fit.power_law.xmin
    out = {"n": int(len(sizes)), "alpha": float(a), "alpha_se": float(s),
           "alpha_ci95": [float(a - 1.96 * s), float(a + 1.96 * s)], "xmin": float(xmin),
           "ks_D": float(fit.power_law.D), "n_tail": int((sizes >= xmin).sum()),
           "tail_share": float((sizes >= xmin).mean()), "max": float(sizes.max()),
           "median": float(np.median(sizes))}
    for alt in ("exponential", "lognormal", "truncated_power_law"):
        R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
        out[f"vs_{alt}"] = {"normalized_llr": float(R), "p": float(p),
                            "favors": "power_law" if (R > 0 and p < 0.05) else (alt if (R < 0 and p < 0.05) else "neither")}
    out["lognormal_params"] = {"mu": float(fit.lognormal.mu), "sigma": float(fit.lognormal.sigma)}
    out["truncated_power_law_params"] = {"alpha": float(fit.truncated_power_law.alpha),
                                         "lambda": float(fit.truncated_power_law.parameter2)}
    if gof_iters:
        out["goodness_of_fit"] = gof_pvalue(sizes, fit, gof_iters, np.random.RandomState(SEED))
    return out, fit


def ccdf_rows(name: str, sizes: np.ndarray, fit) -> pd.DataFrame:
    x = np.sort(np.asarray(sizes, float))
    u, idx = np.unique(x, return_index=True)
    ccdf = 1 - idx / len(x)
    rows = pd.DataFrame({"set": name, "x": u, "empirical_ccdf": ccdf})
    tail = rows["x"] >= fit.power_law.xmin
    scale = float((x >= fit.power_law.xmin).mean())
    for dist in ("power_law", "lognormal", "truncated_power_law", "exponential"):
        rows[f"fit_{dist}"] = np.nan
        rows.loc[tail, f"fit_{dist}"] = scale * getattr(fit, dist).ccdf(rows.loc[tail, "x"].to_numpy())
    return rows


def size_sets(variant: str) -> dict[str, np.ndarray]:
    c = pd.read_csv(RESULTS / f"cascades{suffix(variant)}.csv")
    meme = c["cascade_type"] == "meme"
    other = ~meme
    sets = {
        "reproduction_capped_n3to5": c.loc[(meme & c["in_cap_n3to5"] & (c["n"] >= 3)) | other, "size"].to_numpy(),
        "uncapped_n3to5": c.loc[(meme & (c["n"] >= 3)) | other, "size"].to_numpy(),
        "capped_n2to5": c.loc[(meme & c["in_cap_n2to5"]) | other, "size"].to_numpy(),
        "primary_uncapped_n2to5": c["size"].to_numpy(),
        "meme_only_uncapped_n2to5": c.loc[meme, "size"].to_numpy(),
    }
    if variant == "main" and (RESULTS / "embedding_cascades.csv").exists():
        sets["embedding"] = pd.read_csv(RESULTS / "embedding_cascades.csv")["size"].to_numpy()
    return sets


def run(variant: str, gof_iters: int) -> dict:
    seed_everything()
    sfx = suffix(variant)
    res, curves = {}, []
    for name, sizes in size_sets(variant).items():
        g = gof_iters if name in ("reproduction_capped_n3to5", "embedding") else 0
        out, fit = fit_sizes(sizes, g)
        res[name] = out
        curves.append(ccdf_rows(name, sizes, fit))
        log.info("%s: n=%d alpha=%.3f xmin=%g D=%.4f", name, out["n"], out["alpha"], out["xmin"], out["ks_D"])
    ccdf = pd.concat(curves, ignore_index=True)
    ccdf.to_csv(RESULTS / f"figure2_ccdf{sfx}.csv", index=False)
    rows = []
    for name, r in res.items():
        for alt in ("exponential", "lognormal", "truncated_power_law"):
            rows.append({"set": name, "alternative": alt, "normalized_llr": r[f"vs_{alt}"]["normalized_llr"],
                         "p": r[f"vs_{alt}"]["p"], "favors": r[f"vs_{alt}"]["favors"]})
    pd.DataFrame(rows).to_csv(RESULTS / f"table_powerlaw_lr{sfx}.csv", index=False)
    pd.DataFrame([{"set": k, **{kk: v for kk, v in r.items() if not isinstance(v, (dict, list))},
                   "alpha_ci_lower": r["alpha_ci95"][0], "alpha_ci_upper": r["alpha_ci95"][1]}
                  for k, r in res.items()]).to_csv(RESULTS / f"table_powerlaw_fits{sfx}.csv", index=False)
    write_json(RESULTS / f"powerlaw{sfx}.json", {"variant": variant, "primary_set": "primary_uncapped_n2to5",
                                                 "reproduction_set": "reproduction_capped_n3to5", "fits": res})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--gof-iters", type=int, default=200)
    a = ap.parse_args()
    run(a.variant, a.gof_iters)
