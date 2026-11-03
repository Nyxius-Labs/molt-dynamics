"""Step 7: human-rating sample for Q and weight sensitivity of Q.

q_validation_sample.csv is the rating sheet; it is blinded (no Q, no stratum).
q_validation_key.csv maps sample_id to thread, stratum and Q. Once human_score is
filled in, rerunning this step adds Spearman and Pearson agreement to the outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, SEED, WORK, get_logger, read_json, seed_everything, write_json
from s06_collaboration import Q_CODE_WEIGHTS, Q_SPEC_WEIGHTS, SUCCESS_Q

log = get_logger("s07_q_validation")

N_SAMPLE = 100
N_STRATA = 5
N_WEIGHTS = 20


def stratified_sample(E: pd.DataFrame, q: str, rng) -> pd.DataFrame:
    E = E.copy()
    E["stratum"] = pd.qcut(E[q].rank(method="first"), N_STRATA, labels=False)
    if len(E) <= N_SAMPLE:
        return E
    per = {s: int(round(N_SAMPLE * len(g) / len(E))) for s, g in E.groupby("stratum")}
    while sum(per.values()) != N_SAMPLE:
        s = max(per, key=per.get) if sum(per.values()) > N_SAMPLE else min(per, key=per.get)
        per[s] += -1 if sum(per.values()) > N_SAMPLE else 1
    parts = [g.sample(per[s], random_state=rng) for s, g in E.groupby("stratum")]
    return pd.concat(parts)


def weight_sensitivity(E: pd.DataFrame, comps: dict, q: str, rng) -> dict:
    C = E[list(comps)].to_numpy(float)
    base = E[q].to_numpy()
    W = rng.dirichlet(np.ones(len(comps)), size=N_WEIGHTS)
    rows = []
    for i, w in enumerate(W):
        qw = C @ w
        rho = stats.spearmanr(base, qw).statistic
        tau = stats.kendalltau(base, qw).statistic
        flip = float(((qw >= SUCCESS_Q) != (base >= SUCCESS_Q)).mean())
        rows.append({"q": q, "draw": i, **{f"w_{k}": float(v) for k, v in zip(comps, w)},
                     "spearman_rho": float(rho), "kendall_tau": float(tau), "success_flip_share": flip,
                     "success_rate": float((qw >= SUCCESS_Q).mean())})
    df = pd.DataFrame(rows)
    return df, {"n_weight_vectors": N_WEIGHTS, "distribution": "Dirichlet(1,1,1,1), i.e. uniform on the simplex",
                "spearman_min": float(df["spearman_rho"].min()), "spearman_median": float(df["spearman_rho"].median()),
                "spearman_mean": float(df["spearman_rho"].mean()), "kendall_median": float(df["kendall_tau"].median()),
                "success_flip_share_median": float(df["success_flip_share"].median()),
                "success_flip_share_max": float(df["success_flip_share"].max()),
                "success_rate_range": [float(df["success_rate"].min()), float(df["success_rate"].max())],
                "original_success_rate": float((E[q] >= SUCCESS_Q).mean())}


def main() -> None:
    seed_everything()
    E = pd.read_csv(RESULTS / "collab_events.csv")
    txt = pd.read_parquet(WORK / "collab_events_text.parquet").rename(columns={"post_id": "thread_id"})
    rng = np.random.RandomState(SEED)
    S = stratified_sample(E, "Q_spec", rng)
    S = S.merge(txt[["thread_id", "title", "post_content", "solution"]], on="thread_id", how="left")
    S = S.sample(frac=1, random_state=SEED).reset_index(drop=True)
    S["sample_id"] = [f"Q{i + 1:03d}" for i in range(len(S))]

    sheet_path = RESULTS / "q_validation_sample.csv"
    prior = pd.read_csv(sheet_path, dtype=str) if sheet_path.exists() else None
    sheet = pd.DataFrame({
        "sample_id": S["sample_id"], "thread_id": S["thread_id"], "submolt": S["submolt"],
        "participants": S["participants"], "n_comments": S["n_comments"],
        "title": S["title"].fillna("").str.slice(0, 300),
        "post_excerpt": S["post_content"].fillna("").str.slice(0, 1500),
        "solution_comment": S["solution"].fillna("").str.slice(0, 3000),
        "human_score": "", "rater_notes": "",
    })
    if prior is not None and "human_score" in prior and prior["human_score"].notna().any():
        keep = prior.set_index("thread_id")[["human_score", "rater_notes"]]
        sheet = sheet.set_index("thread_id")
        sheet.update(keep)
        sheet = sheet.reset_index()[["sample_id", "thread_id", "submolt", "participants", "n_comments", "title",
                                     "post_excerpt", "solution_comment", "human_score", "rater_notes"]]
    sheet.to_csv(sheet_path, index=False)
    key = S[["sample_id", "thread_id", "stratum", "Q_spec", "Q_code", "code_presence", "comment_quality",
             "test_inclusion", "syntax_validity"]]
    key.to_csv(RESULTS / "q_validation_key.csv", index=False)

    rng = np.random.RandomState(SEED)
    ws_spec, sum_spec = weight_sensitivity(E, Q_SPEC_WEIGHTS, "Q_spec", rng)
    ws_code, sum_code = weight_sensitivity(E, Q_CODE_WEIGHTS, "Q_code", rng)
    pd.concat([ws_spec, ws_code]).to_csv(RESULTS / "q_weight_sensitivity.csv", index=False)

    rated = pd.to_numeric(sheet["human_score"], errors="coerce")
    out = {"sample": {"n_events": int(len(E)), "n_sampled": int(len(S)), "stratified_by": "Q_spec quintile (rank-based)",
                      "per_stratum": {int(k): int(v) for k, v in S["stratum"].value_counts().sort_index().items()},
                      "blinded": True, "rating_sheet": "results/q_validation_sample.csv",
                      "key": "results/q_validation_key.csv", "n_rated": int(rated.notna().sum())},
           "weight_sensitivity": {"Q_spec": sum_spec, "Q_code": sum_code},
           "q_spec_vs_q_code_spearman": float(stats.spearmanr(E["Q_spec"], E["Q_code"]).statistic)}
    if rated.notna().sum() >= 10:
        m = rated.notna()
        k = key.set_index("sample_id").loc[sheet.loc[m, "sample_id"]]
        out["agreement"] = {q: {"spearman": float(stats.spearmanr(rated[m], k[q]).statistic),
                                "pearson": float(stats.pearsonr(rated[m], k[q]).statistic)} for q in ("Q_spec", "Q_code")}
    write_json(RESULTS / "q_validation.json", out)
    log.info("sample %d events; Q_spec weight-sensitivity median rho %.3f", len(S), sum_spec["spearman_median"])


if __name__ == "__main__":
    main()
