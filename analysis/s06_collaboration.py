"""Step 6: collaborative events, quality Q, matched baselines and dose-response.

Event identification and the as-implemented score (Q_code) follow
molt_dynamics.rq3_collaboration exactly. Q_spec implements the composite described in
the paper's Methods (code 0.30, comment quality 0.30, tests 0.20, ast syntax 0.20).
The earlier run never built a baseline: BaselineComparator.collect_individual_baselines
returns an empty DataFrame and compare_quality_distributions runs a one-sample t-test
of event Q against the constant 0.5. That test is reproduced, and the paper's matched
single-agent baseline is built here with a paired test on event-level differences.
"""

from __future__ import annotations

import argparse
import ast
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import (RESULTS, SEED, TECHNICAL_KEYWORDS, cohens_d_two_sample, get_logger, load_variant,
                    seed_everything, suffix, write_json)

log = get_logger("s06_collab")

MIN_AGENTS, MIN_COMMENTS, MIN_DURATION_MIN = 3, 5, 30
MAX_MATCHES = 5
SUCCESS_Q = 0.5
Q_SPEC_WEIGHTS = {"code_presence": 0.30, "comment_quality": 0.30, "test_inclusion": 0.20, "syntax_validity": 0.20}
Q_CODE_WEIGHTS = {"has_code": 0.30, "has_comments": 0.20, "has_tests": 0.30, "brackets_balanced": 0.20}
CODE_RE = re.compile(r"```|`[^`]+`")
COMMENT_RE = re.compile(r"#.*|//.*|/\*.*\*/")
TEST_RE = re.compile(r"test|assert|expect")
FENCE_RE = re.compile(r"```[^\n`]*\n?([\s\S]*?)```")
INLINE_RE = re.compile(r"`([^`\n]+)`")
LENGTH_SATURATION_WORDS = 50
_VADER = None


def vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def balanced(code: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in code:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return not stack


def q_code(text: str) -> dict:
    c = {"has_code": bool(CODE_RE.search(text)), "has_comments": bool(COMMENT_RE.search(text)),
         "has_tests": bool(TEST_RE.search(text.lower())), "brackets_balanced": balanced(text)}
    c = {k: float(v) for k, v in c.items()}
    c["Q_code"] = sum(Q_CODE_WEIGHTS[k] * c[k] for k in Q_CODE_WEIGHTS)
    return c


def parses(snippet: str) -> bool:
    try:
        ast.parse(snippet)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def q_spec(text: str) -> dict:
    blocks = FENCE_RE.findall(text) or INLINE_RE.findall(text)
    blocks = [b for b in blocks if b.strip()]
    s = (vader().polarity_scores(text)["compound"] + 1) / 2
    lw = min(1.0, len(text.split()) / LENGTH_SATURATION_WORDS)
    c = {"code_presence": float(bool(CODE_RE.search(text))),
         "comment_quality": (s + lw) / 2,
         "test_inclusion": float(bool(TEST_RE.search(text.lower()))),
         "syntax_validity": float(bool(blocks) and all(parses(b) for b in blocks)),
         "vader_component": s, "length_component": lw}
    c["Q_spec"] = sum(Q_SPEC_WEIGHTS[k] * c[k] for k in Q_SPEC_WEIGHTS)
    return c


def threads(d: dict) -> pd.DataFrame:
    P = d["posts"][["id", "agent_id", "submolt", "title", "content", "created_at"]].rename(
        columns={"id": "post_id", "agent_id": "post_author", "content": "post_content", "created_at": "post_time"})
    C = d["comments"][["post_id", "agent_id", "content", "created_at"]].sort_values(["post_id", "created_at"], kind="mergesort")
    C = C[C["post_id"].isin(set(P["post_id"]))]
    C["has_tick"] = C["content"].str.contains("`", regex=False)
    g = C.groupby("post_id", sort=False)
    T = pd.DataFrame({"n_comments": g.size(),
                      "commenters": g["agent_id"].agg(lambda s: set(s.dropna().astype(str))),
                      "cmin": g["created_at"].min(), "cmax": g["created_at"].max(),
                      "comment_text": g["content"].agg(" ".join)})
    last_tick = C[C["has_tick"]].groupby("post_id")["content"].last()
    last_any = g["content"].last()
    T["solution"] = last_tick.reindex(T.index).fillna(last_any)
    T = P.merge(T, left_on="post_id", right_index=True, how="inner")
    T["participants"] = [len(set(s) | {str(a)}) for s, a in zip(T["commenters"], T["post_author"])]
    T["duration_minutes"] = (np.maximum(T["cmax"], T["post_time"]) - np.minimum(T["cmin"], T["post_time"])).dt.total_seconds() / 60
    all_text = (T["title"] + " " + T["post_content"] + " " + T["comment_text"]).str.lower()
    hits = np.column_stack([all_text.str.contains(k, regex=False).to_numpy() for k in TECHNICAL_KEYWORDS])
    T["keyword_count"] = hits.sum(axis=1)
    T["technical"] = T["keyword_count"] > 0
    T["text_chars"] = all_text.str.len()
    return T.drop(columns=["comment_text", "commenters"]).reset_index(drop=True)


def score(T: pd.DataFrame) -> pd.DataFrame:
    qc = pd.DataFrame([q_code(s) for s in T["solution"]], index=T.index)
    qs = pd.DataFrame([q_spec(s) for s in T["solution"]], index=T.index)
    return pd.concat([T, qc, qs], axis=1)


def match(events: pd.DataFrame, pool: pd.DataFrame, keys: dict, rng) -> list[list[int]]:
    """For each event, up to MAX_MATCHES random pool rows satisfying all key constraints.

    keys maps column -> tolerance (0 = exact match)."""
    out = []
    for _, e in events.iterrows():
        m = pd.Series(True, index=pool.index)
        for col, tol in keys.items():
            if tol == 0:
                m &= pool[col] == e[col]
            else:
                m &= (pool[col] - e[col]).abs() <= tol
        idx = pool.index[m].to_numpy()
        if len(idx) > MAX_MATCHES:
            idx = rng.choice(idx, MAX_MATCHES, replace=False)
        out.append(sorted(int(i) for i in idx))
    return out


def compare(events: pd.DataFrame, pool: pd.DataFrame, matches: list[list[int]], q: str) -> dict:
    has = np.array([len(m) > 0 for m in matches])
    ev = events[has]
    base_mean = np.array([pool.loc[m, q].mean() for m in matches if m])
    diff = ev[q].to_numpy() - base_mean
    uniq = sorted({i for m in matches for i in m})
    bq = pool.loc[uniq, q].to_numpy() if uniq else np.array([])
    res = {"n_events": int(len(events)), "n_events_matched": int(has.sum()),
           "n_unique_baseline_threads": int(len(uniq)),
           "mean_matches_per_event": float(np.mean([len(m) for m in matches if m])) if has.any() else 0.0}
    if has.sum() >= 3 and len(uniq) >= 3:
        t, p = stats.ttest_rel(ev[q].to_numpy(), base_mean)
        tw, pw = stats.ttest_ind(ev[q].to_numpy(), bq, equal_var=False)
        res.update({
            "event_mean_Q": float(ev[q].mean()), "event_sd_Q": float(ev[q].std(ddof=1)),
            "baseline_mean_Q": float(bq.mean()), "baseline_sd_Q": float(bq.std(ddof=1)),
            "baseline_success_rate": float((bq >= SUCCESS_Q).mean()),
            "event_success_rate": float((ev[q] >= SUCCESS_Q).mean()),
            "paired": {"test": "paired t-test on event Q minus mean Q of its matches", "mean_difference": float(diff.mean()),
                       "sd_difference": float(diff.std(ddof=1)), "t": float(t), "df": int(len(diff) - 1), "p": float(p),
                       "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0},
            "two_sample": {"test": "Welch t-test, matched events vs unique matched baseline threads", "t": float(tw), "p": float(pw),
                           "cohens_d_pooled": cohens_d_two_sample(ev[q].to_numpy(), bq)},
        })
        res["available"] = True
    else:
        res["available"] = False
    return res


def run(variant: str) -> dict:
    seed_everything()
    sfx = suffix(variant)
    d = load_variant(variant)
    T = threads(d)
    ptxt = (d["posts"]["title"] + " " + d["posts"]["content"]).str.lower()
    n_technical_posts = int(np.any(np.column_stack([ptxt.str.contains(k, regex=False) for k in TECHNICAL_KEYWORDS]), axis=1).sum())
    is_event = (T["n_comments"] >= MIN_COMMENTS) & T["technical"] & (T["participants"] >= MIN_AGENTS) & (T["duration_minutes"] >= MIN_DURATION_MIN)
    single = T["technical"] & (T["participants"] == 1)
    cand = T[is_event | single | (T["technical"] & (T["n_comments"] >= 3))].copy()
    cand = score(cand)
    cand["is_event"] = is_event.reindex(cand.index)
    E = cand[cand["is_event"] & (cand["solution"].str.len() > 0)].copy()
    S = cand[single.reindex(cand.index)].copy()
    log.info("%d threads, %d technical posts, %d events, %d single-agent technical threads", len(T), n_technical_posts, len(E), len(S))

    out = {"variant": variant, "n_threads_with_comments": int(len(T)), "n_technical_threads": n_technical_posts,
           "technical_thread_definition": "post whose title+body contains a technical keyword (rq3 extract_technical_threads)",
           "n_events": int(len(E)), "event_rules": {"min_agents": MIN_AGENTS, "min_comments": MIN_COMMENTS,
                                                   "min_duration_minutes": MIN_DURATION_MIN,
                                                   "keyword_match": "case-insensitive substring of any TECHNICAL_KEYWORDS in title+post+comments"},
           "n_single_agent_technical_threads": int(len(S)),
           "q_definitions": {
               "Q_code": {"weights": Q_CODE_WEIGHTS, "applied_to": "solution comment (last comment containing a backtick, else last comment)",
                          "notes": "as implemented in rq3 SolutionAssessor.assess_code_solution; '//' matches URLs, syntax is bracket balance"},
               "Q_spec": {"weights": Q_SPEC_WEIGHTS, "applied_to": "same solution comment",
                          "comment_quality": f"mean of (VADER compound + 1)/2 and min(1, words/{LENGTH_SATURATION_WORDS})",
                          "syntax_validity": "1 if the solution has fenced code blocks (else inline code spans) and every one parses with ast.parse",
                          "notes": "paper Methods weights; the comment-quality operationalisation is defined here and needs author sign-off"}},
           "primary_q": "Q_spec", "success_threshold": SUCCESS_Q}

    for q in ("Q_spec", "Q_code"):
        x = E[q].to_numpy()
        t, p = stats.ttest_1samp(x, 0.5)
        out[f"earlier_run_test_{q}"] = {"test": "one-sample t-test of event Q against the constant 0.5 (rq3 compare_quality_distributions)",
                                        "n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t": float(t), "p": float(p),
                                        "cohens_d": float((x.mean() - 0.5) / x.std(ddof=1)),
                                        "success_rate_ge_0.5": float((x >= 0.5).mean()), "success_rate_gt_0.5_code": float((x > 0.5).mean())}
    out["comparison_type_in_code"] = ("neither paired nor two-sample: a one-sample t-test against the constant 0.5; "
                                      "collect_individual_baselines() returns an empty DataFrame, so no baseline threads were ever used")

    def matched(pool, keys):
        return match(E, pool, keys, np.random.RandomState(SEED))

    has_sol = cand["solution"].str.len() > 0
    low = cand[(~cand["is_event"]) & cand["technical"] & (cand["participants"] <= 2) & has_sol]
    nonevent = cand[(~cand["is_event"]) & cand["technical"] & has_sol]
    paper_m = matched(S, {"submolt": 0, "n_comments": 2})
    base_matches = matched(low, {"submolt": 0, "n_comments": 2})
    diff_matches = matched(low, {"submolt": 0, "n_comments": 2, "keyword_count": 1})
    part_matches = matched(nonevent, {"submolt": 0, "n_comments": 2, "keyword_count": 1, "participants": 1})
    out["n_low_participation_technical_threads"] = int(len(low))
    out["n_nonevent_technical_threads_with_comments"] = int(len(nonevent))
    out["primary_baseline"] = "matched_low_participation"
    paper_cmp = {}
    for q in ("Q_spec", "Q_code"):
        r = compare(E, S, paper_m, q)
        r["feasibility"] = ("estimable" if r["available"] else
                            "not estimable: fewer than 3 matched events or matched single-agent threads")
        paper_cmp[q] = r
    for q in ("Q_spec", "Q_code"):
        out[f"baseline_{q}"] = {
            "paper_baseline_single_agent": {
                "match": "single-agent technical threads, same submolt, comments +-2, up to 5 (paper Methods)",
                **paper_cmp[q]},
            "matched_low_participation": {
                "match": "technical non-event threads with 1-2 participants, same submolt, comments +-2, up to 5",
                **compare(E, low, base_matches, q)},
            "rematch_difficulty": {
                "match": "technical non-event threads with 1-2 participants, same submolt, comments +-2, keyword count +-1, up to 5",
                **compare(E, low, diff_matches, q)},
            "rematch_participants_and_difficulty": {
                "match": "technical non-event threads, same submolt, comments +-2, keyword count +-1, participants +-1, up to 5",
                **compare(E, nonevent, part_matches, q)},
        }

    E["log_participants"] = np.log(E["participants"])
    import statsmodels.formula.api as smf
    dose = {}
    for q in ("Q_spec", "Q_code"):
        m = smf.ols(f"{q} ~ log_participants", data=E).fit(cov_type="HC3")
        allt = cand[cand["technical"] & (cand["n_comments"] >= MIN_COMMENTS) & (cand["solution"].str.len() > 0)].copy()
        allt["log_participants"] = np.log(allt["participants"])
        m2 = smf.ols(f"{q} ~ log_participants", data=allt).fit(cov_type="HC3")
        dose[q] = {"events": {"slope": float(m.params["log_participants"]), "se_hc3": float(m.bse["log_participants"]),
                              "p": float(m.pvalues["log_participants"]), "intercept": float(m.params["Intercept"]),
                              "r2": float(m.rsquared), "n": int(m.nobs)},
                   "all_technical_threads_ge5_comments": {"slope": float(m2.params["log_participants"]),
                                                          "se_hc3": float(m2.bse["log_participants"]),
                                                          "p": float(m2.pvalues["log_participants"]),
                                                          "r2": float(m2.rsquared), "n": int(m2.nobs)}}
    out["dose_response"] = dose

    comp_rows = []
    succ = E["Q_spec"] >= SUCCESS_Q
    for c, wt in Q_SPEC_WEIGHTS.items():
        comp_rows.append({"q": "Q_spec", "component": c, "weight": wt, "successful_mean": float(E.loc[succ, c].mean()) if succ.any() else 0.0,
                          "unsuccessful_mean": float(E.loc[~succ, c].mean()) if (~succ).any() else 0.0,
                          "all_events_mean": float(E[c].mean())})
    succ_c = E["Q_code"] >= SUCCESS_Q
    for c, wt in Q_CODE_WEIGHTS.items():
        comp_rows.append({"q": "Q_code", "component": c, "weight": wt, "successful_mean": float(E.loc[succ_c, c].mean()) if succ_c.any() else 0.0,
                          "unsuccessful_mean": float(E.loc[~succ_c, c].mean()) if (~succ_c).any() else 0.0,
                          "all_events_mean": float(E[c].mean())})
    out["n_successful_Q_spec"] = int(succ.sum())
    out["n_successful_Q_code"] = int(succ_c.sum())
    out["components"] = comp_rows
    pd.DataFrame(comp_rows).to_csv(RESULTS / f"table_quality_components{sfx}.csv", index=False)

    feat = []
    for name, df in (("events", E), ("low_participation_baseline_matched", low.loc[sorted({i for m in base_matches for i in m})])):
        feat.append({"group": name, "n": int(len(df)), "mean_comments": float(df["n_comments"].mean()) if len(df) else 0.0,
                     "mean_keyword_count": float(df["keyword_count"].mean()) if len(df) else 0.0,
                     "mean_text_chars": float(df["text_chars"].mean()) if len(df) else 0.0,
                     "mean_participants": float(df["participants"].mean()) if len(df) else 0.0,
                     "mean_duration_minutes": float(df["duration_minutes"].mean()) if len(df) else 0.0})
    out["thread_feature_differences"] = feat
    pd.DataFrame(feat).to_csv(RESULTS / f"table_thread_features{sfx}.csv", index=False)

    E["n_baseline_matches"] = [len(m) for m in base_matches]
    E["matched_baseline_mean_Q_spec"] = [low.loc[m, "Q_spec"].mean() if m else np.nan for m in base_matches]
    E["matched_baseline_mean_Q_code"] = [low.loc[m, "Q_code"].mean() if m else np.nan for m in base_matches]
    E["baseline_thread_ids"] = [";".join(low.loc[m, "post_id"]) for m in base_matches]
    cols = ["post_id", "submolt", "post_author", "participants", "n_comments", "duration_minutes", "keyword_count",
            "Q_spec", "code_presence", "comment_quality", "test_inclusion", "syntax_validity",
            "Q_code", "has_code", "has_comments", "has_tests", "brackets_balanced",
            "n_baseline_matches", "matched_baseline_mean_Q_spec", "matched_baseline_mean_Q_code", "baseline_thread_ids"]
    E[cols].rename(columns={"post_id": "thread_id"}).to_csv(RESULTS / f"collab_events{sfx}.csv", index=False)
    E[["post_id", "title", "post_content", "solution", "Q_spec", "Q_code", "code_presence", "comment_quality",
       "test_inclusion", "syntax_validity", "participants", "n_comments"]].to_parquet(
        RESULTS.parent / "work" / f"collab_events_text{sfx}.parquet", index=False)
    pd.DataFrame({"participants": E["participants"], "Q_spec": E["Q_spec"], "Q_code": E["Q_code"]}).to_csv(
        RESULTS / f"figure4_quality_by_participants{sfx}.csv", index=False)

    b = out["baseline_Q_spec"]["matched_low_participation"]
    rows = [("collaborative events", out["n_events"]), ("technical threads", n_technical_posts)]
    if b["available"]:
        rows += [("events with >=1 match", b["n_events_matched"]), ("matched baseline threads", b["n_unique_baseline_threads"]),
                 ("event mean Q", b["event_mean_Q"]), ("baseline mean Q", b["baseline_mean_Q"]),
                 ("baseline SD Q", b["baseline_sd_Q"]), ("baseline success rate", b["baseline_success_rate"]),
                 ("paired t", b["paired"]["t"]), ("paired p", b["paired"]["p"]), ("Cohen's d_z", b["paired"]["cohens_dz"])]
    pd.DataFrame(rows, columns=["statistic", "value"]).to_csv(RESULTS / f"table_baseline_comparison{sfx}.csv", index=False)
    write_json(RESULTS / f"collaboration{sfx}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args()
    r = run(a.variant)
    for q in ("Q_spec", "Q_code"):
        log.info("%s earlier-run test: %s", q, r[f"earlier_run_test_{q}"])
        log.info("%s paper baseline: %s", q, {k: v for k, v in r[f"baseline_{q}"]["paper_baseline_single_agent"].items() if k != "match"})
