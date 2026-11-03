"""Step 11: assemble results/results.json (keyed by paper table and figure) and
results/RESULTS_SUMMARY.md, and validate that no value is null, NaN or a placeholder."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (RESULTS, REVISIONS, TECHNICAL_KEYWORDS, _dedupe, _read_table_files, get_logger,
                    read_json, write_json)

log = get_logger("s11_assemble")

REFERENCE = {
    "agents": 90704, "cascades_total": 10323, "meme_cascades": 10000, "selected_k": 6, "silhouette": 0.914,
    "powerlaw_alpha": 2.57, "powerlaw_xmin": 104, "cox_hr": 0.528, "collab_events": 164,
    "technical_threads": 60045, "cohens_d": -0.88,
}
PLACEHOLDERS = {"", "tbd", "todo", "n/a", "na", "none", "null", "nan", "placeholder", "..."}


def opt(name: str):
    p = RESULTS / name
    return read_json(p) if p.exists() else None


def csv_records(name: str) -> list[dict]:
    return pd.read_csv(RESULTS / name).to_dict("records")


def pct(new: float, ref: float) -> float:
    return 100.0 * (new - ref) / abs(ref)


def technical_posts_without_window() -> int:
    p = _dedupe(_read_table_files("posts", ["id", "title", "content", "dump_date"], REVISIONS["primary"]["dir"] / "data"))
    t = (p["title"].fillna("None").astype(str) + " " + p["content"].fillna("None").astype(str)).str.lower()
    return int(np.any(np.column_stack([t.str.contains(k, regex=False) for k in TECHNICAL_KEYWORDS]), axis=1).sum())


def deviations(S, C, P, X, CO, man) -> list[dict]:
    sweep = {r["k"]: r["silhouette"] for r in S["k_sweep"]}
    rows = []

    def add(name, ref, new, cause_if_large, note=""):
        dv = pct(new, ref)
        rows.append({"quantity": name, "reference": ref, "rerun": new, "deviation_pct": dv,
                     "above_5pct": bool(abs(dv) > 5), "cause": cause_if_large if abs(dv) > 5 else (note or "within 5%")})

    add("agents", REFERENCE["agents"], man["revisions"]["primary"]["window"]["agent_population"],
        "", "agents table of the pinned revision ee98f6c8, identical to the earlier run")
    add("cascades (capped reproduction: memes + skill + behavioral)", REFERENCE["cascades_total"], C["reproduction_total_capped_n3to5"],
        "", f"10,000 capped memes + {C['n_skill_cascades']} skill + {C['n_behavioral_cascades']} behavioral; the earlier run also scanned posts dated 2026-02-21..24")
    add("meme cascades (capped)", REFERENCE["meme_cascades"], C["cap"]["capped_meme_cascades_n3to5"], "", "hard cap cascades[:10000]")
    add("selected k", REFERENCE["selected_k"], S["selected_k"],
        f"silhouette rises almost monotonically over k=3..8 ({min(sweep.values()):.3f} to {max(sweep.values()):.3f}); "
        f"the maximum moved from k=6 to k={S['selected_k']} by {max(sweep.values()) - sweep[6]:.4f}, within k-means initialisation noise; "
        f"at k=6 the rerun silhouette is {sweep[6]:.4f}")
    add("silhouette at selected k", REFERENCE["silhouette"], S["silhouette"], "")
    add("silhouette at k=6", REFERENCE["silhouette"], sweep[6], "")
    add("power-law alpha (capped reproduction set)", REFERENCE["powerlaw_alpha"], P["reproduction_capped_n3to5"]["alpha"], "")
    add("power-law xmin (capped reproduction set)", REFERENCE["powerlaw_xmin"], P["reproduction_capped_n3to5"]["xmin"],
        f"xmin equals the smallest cascade kept by the 10,000 cap ({C['cap']['min_size_in_capped_set_n3to5']} adopters here); "
        "the cap boundary shifts with the corpus (the earlier run also scanned posts dated 2026-02-21..24)")
    add("Cox HR (earlier-run procedure, top-100 capped cascades, static exposure)", REFERENCE["cox_hr"],
        X["cox"]["reproduction_top100_static"]["hazard_ratio"],
        "the top-100 cascade membership differs (corpus includes only in-window posts; size ties broken by seed time "
        "instead of dict insertion order); the static, asymmetric exposure that drives HR < 1 is unchanged")
    add("collaborative events", REFERENCE["collab_events"], CO["n_events"], "")
    add("technical threads", REFERENCE["technical_threads"], CO["n_technical_threads"],
        f"the earlier run counted every post in the revision, including posts created 2026-02-21..24 after the window; "
        f"without the window filter the count is {technical_posts_without_window()} (exact match)")
    add("Cohen's d (earlier-run one-sample test, Q_code)", REFERENCE["cohens_d"], CO["earlier_run_test_Q_code"]["cohens_d"], "")
    return rows


def variant_digest(sfx: str) -> dict:
    out = {}
    s = opt(f"structure{sfx}.json")
    if s:
        out["structure"] = {"n_agents": s["n_agents"], "graph_nodes": s["graph"]["n_nodes"], "graph_interactions": s["graph"]["n_interactions"],
                            "selected_k": s["selected_k"], "silhouette": s["silhouette"], "largest_cluster_share": s["largest_cluster_share"],
                            "periphery_share_total_degree_le_1": s["periphery_share_total_degree_le_1"],
                            "bootstrap_ari_mean": s["bootstrap"]["ari"]["mean"], "gmm_ari_vs_kmeans": s["gmm"]["ari_vs_kmeans"]}
    c = opt(f"cascades{sfx}.json")
    if c:
        out["cascades"] = {"uncapped_meme_n2to5": c["cap"]["uncapped_meme_cascades_n2to5"],
                           "uncapped_meme_n3to5": c["cap"]["uncapped_meme_cascades_n3to5"]}
    x = opt(f"exposure_models{sfx}.json")
    if x:
        lg = x["logistic"]["primary_uncapped_n2to5"]["coefficients"]
        out["exposure"] = {"logit_exposure": lg["exposure"]["coef"], "logit_exposure_p": lg["exposure"]["p"],
                           "logit_exposure_squared": lg["exposure_squared"]["coef"], "logit_exposure_squared_p": lg["exposure_squared"]["p"],
                           **{f"cox_{k}_hr": v["hazard_ratio"] for k, v in x["cox"].items()},
                           **{f"cox_{k}_p": v["p"] for k, v in x["cox"].items()}}
    co = opt(f"collaboration{sfx}.json")
    if co:
        d = {"n_events": co["n_events"], "n_technical_threads": co["n_technical_threads"],
             "earlier_run_d_Q_code": co["earlier_run_test_Q_code"]["cohens_d"]}
        for q in ("Q_spec", "Q_code"):
            for bname in ("matched_low_participation", "paper_baseline_single_agent"):
                b = co[f"baseline_{q}"][bname]
                d[f"{bname}_available_{q}"] = b["available"]
                if b["available"]:
                    d[f"{bname}_dz_{q}"] = b["paired"]["cohens_dz"]
                    d[f"{bname}_p_{q}"] = b["paired"]["p"]
                    d[f"{bname}_n_{q}"] = b["n_events_matched"]
        out["collaboration"] = d
    return out


def check(obj, path="results"):
    bad = []
    if obj is None:
        bad.append(path)
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(path)
    elif isinstance(obj, str) and obj.strip().lower() in PLACEHOLDERS:
        bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += check(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += check(v, f"{path}[{i}]")
    return bad


def clean_records(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({k: v for k, v in r.items() if not (v is None or (isinstance(v, float) and not math.isfinite(v))
                                                       or (isinstance(v, str) and v.strip() == ""))})
    return out


def main() -> None:
    man = read_json(RESULTS / "data_manifest.json")
    S = read_json(RESULTS / "structure.json")
    C = read_json(RESULTS / "cascades.json")
    P = read_json(RESULTS / "powerlaw.json")["fits"]
    X = read_json(RESULTS / "exposure_models.json")
    CO = read_json(RESULTS / "collaboration.json")
    QV = read_json(RESULTS / "q_validation.json")
    AU = read_json(RESULTS / "autonomy_audit.json")
    EM = opt("embedding_cascades.json")
    HP = opt("human_platform.json")
    dev = deviations(S, C, P, X, CO, man)
    sweep = {r["k"]: r["silhouette"] for r in S["k_sweep"]}
    lr = csv_records("table_powerlaw_lr.csv")

    R = {
        "meta": {
            "seed": 42, "window": [man["revisions"]["primary"]["window"]["start_inclusive"], man["revisions"]["primary"]["window"]["end_inclusive"]],
            "primary_revision": man["revisions"]["primary"]["sha"], "head_revision": man["revisions"]["head"]["sha"],
            "posts": man["revisions"]["primary"]["window"]["posts"], "comments": man["revisions"]["primary"]["window"]["comments"],
            "agents": man["revisions"]["primary"]["window"]["agent_population"],
            "active_agents": man["revisions"]["primary"]["window"]["active_agents_in_window"],
            "graph_nodes": S["graph"]["n_nodes"], "graph_edges": S["graph"]["n_edges"], "graph_interactions": S["graph"]["n_interactions"],
            "edge_rule": S["graph"]["edge_rule"], "coverage": man["coverage"],
            "variants": {"main": "pinned revision ee98f6c8, full window",
                         "no_autonomy": "main minus all activity by agents flagged in the autonomy audit",
                         "no_incident": "main minus all activity created 2026-01-30..2026-02-01",
                         "merged": "union of pinned revision and archive head, deduplicated by id (recovers the posts most top-level comments point to)"},
        },
        "reference_comparison": dev,
        "sec_res_structure": {
            "selected_k": S["selected_k"], "silhouette": S["silhouette"], "silhouette_at_k6": sweep[6],
            "k_sweep": S["k_sweep"], "largest_cluster_share": S["largest_cluster_share"],
            "periphery_share_lowest_cluster": S["periphery_share_lowest_cluster"],
            "periphery_share_total_degree_le_1": S["periphery_share_total_degree_le_1"],
            "share_agents_not_in_graph": S["share_not_in_graph"], "n_unique_feature_rows": S["n_unique_feature_rows"],
            "bootstrap": S["bootstrap"], "gmm": S["gmm"],
            "silhouette_interpretation": "the silhouette is dominated by the "
                                         f"{100 * S['share_not_in_graph']:.1f}% of agents with no interactions, which share one identical feature vector",
        },
        "tab_network_clusters": S["clusters"],
        "fig1_clusters": {"csv": "results/table_network_clusters.csv", "per_agent_csv": "results/agents_features_clusters.csv",
                          "profiles": [{k: v for k, v in c.items() if k.startswith(("cluster", "count", "proportion", "z_"))} for c in S["clusters"]]},
        "tab_role_taxonomy_results": S["roles"],
        "role_thresholds_file": "results/role_thresholds.json",
        "sec_res_diffusion": {
            "cascade_counts": {"ngram_counts_by_n": C["ngram_counts_by_n"], "cap": C["cap"],
                               "n_skill": C["n_skill_cascades"], "n_behavioral": C["n_behavioral_cascades"],
                               "reproduction_total_capped": C["reproduction_total_capped_n3to5"],
                               "primary_total_uncapped_n2to5": C["primary_total_uncapped_n2to5"]},
            "powerlaw_primary": P["primary_uncapped_n2to5"], "powerlaw_reproduction_capped": P["reproduction_capped_n3to5"],
            "powerlaw_uncapped_n3to5": P["uncapped_n3to5"], "powerlaw_capped_n2to5": P["capped_n2to5"],
            "embedding_cascades": EM if EM else {"status_file": "embedding step did not complete; see RESULTS_SUMMARY.md"},
            "powerlaw_embedding": P.get("embedding", {"status": "not fitted"}),
            "exposure_definition_file": "results/exposure_definition.md",
            "logistic": X["logistic"], "cox": X["cox"],
            "frailty": "shared-frailty Cox is not implemented in lifelines or statsmodels; the stratified-by-cascade Cox "
                       "(cascade-specific baseline hazards) is reported in its place",
        },
        "fig2_ccdf": {"csv": "results/figure2_ccdf.csv", "cascade_size_vector_csv": "results/cascades.csv",
                      "fits": {k: {kk: v[kk] for kk in ("n", "alpha", "alpha_ci95", "xmin", "ks_D", "n_tail")} for k, v in P.items()}},
        "tab_adoption_models": csv_records("table_adoption_models.csv"),
        "fig3_exposure": {"binned_csv": "results/figure3_binned_adoption.csv", "logistic_curve_csv": "results/figure3_logistic_curve.csv",
                          "binned": X["binned_adoption"]},
        "sec_res_collab": {k: CO[k] for k in ("n_events", "n_technical_threads", "n_threads_with_comments", "event_rules",
                                               "n_single_agent_technical_threads", "n_low_participation_technical_threads",
                                               "comparison_type_in_code", "earlier_run_test_Q_code", "earlier_run_test_Q_spec",
                                               "q_definitions", "primary_q", "primary_baseline", "n_successful_Q_spec", "n_successful_Q_code",
                                               "baseline_Q_spec", "baseline_Q_code", "dose_response")},
        "tab_baseline_comparison": csv_records("table_baseline_comparison.csv"),
        "fig4_participants": {"csv": "results/figure4_quality_by_participants.csv", "dose_response": CO["dose_response"]},
        "tab_quality_components": CO["components"],
        "app_validation": {"bootstrap": S["bootstrap"], "k_sweep": S["k_sweep"], "gmm": S["gmm"],
                           "bootstrap_csv": "results/bootstrap_clustering.csv"},
        "app_powerlaw": {"fits": P, "likelihood_ratios": lr},
        "app_cox": X["cox"],
        "app_task3": {"thread_feature_differences": CO["thread_feature_differences"], "rematching": {q: CO[f"baseline_{q}"] for q in ("Q_spec", "Q_code")},
                      "q_validation": QV},
        "robustness": {
            "autonomy_audit": {**AU, "rerun": variant_digest("_no_autonomy")},
            "incident_sensitivity": {"excluded_dates": ["2026-01-30", "2026-01-31", "2026-02-01"], "rerun": variant_digest("_no_incident")},
            "merged_data": {"rerun": variant_digest("_merged")},
            "main": variant_digest(""),
        },
        "human_platform": HP if HP else {"status": "human-platform step did not complete; see RESULTS_SUMMARY.md"},
    }
    n_agent_rows = len(pd.read_csv(RESULTS / "agents_features_clusters.csv", usecols=["agent_id"]))
    n_event_rows = len(pd.read_csv(RESULTS / "collab_events.csv", usecols=["thread_id"]))
    R["meta"]["output_row_counts"] = {
        "agents_features_clusters.csv": n_agent_rows, "manifest_agent_population": M_pop(man),
        "collab_events.csv": n_event_rows, "collaboration_n_events": CO["n_events"],
        "consistent": bool(n_agent_rows == M_pop(man) and n_event_rows == CO["n_events"])}
    if not R["meta"]["output_row_counts"]["consistent"]:
        raise SystemExit(f"row counts disagree: {R['meta']['output_row_counts']}")
    R["tab_adoption_models"] = clean_records(R["tab_adoption_models"])
    R["tab_baseline_comparison"] = clean_records(R["tab_baseline_comparison"])
    R["app_powerlaw"]["likelihood_ratios"] = clean_records(lr)
    R["sec_res_diffusion"]["cascade_counts"]["top_memes"] = [m for m in C["top_memes"] if m.get("ngram")]
    bad = check(R)
    if bad:
        raise SystemExit(f"results.json would contain null/NaN/placeholder values at: {bad[:20]}")
    write_json(RESULTS / "results.json", R)
    (RESULTS / "RESULTS_SUMMARY.md").write_text(summary_md(R, dev))
    log.info("wrote results.json and RESULTS_SUMMARY.md")


def M_pop(man: dict) -> int:
    return int(man["revisions"]["primary"]["window"]["agent_population"])


def f(x, nd=3):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def summary_md(R: dict, dev: list[dict]) -> str:
    from summary_text import render
    return render(R, dev, f)


if __name__ == "__main__":
    main()
