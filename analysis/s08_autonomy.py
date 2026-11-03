"""Step 8: autonomy audit. Flags agents whose activity carries human-driven signatures.

Rules (an agent is flagged if any rule fires):
  R1 cadence     >= 30 events over >= 2 days whose UTC hour-of-day histogram has a
                 circular run of >= 6 consecutive empty hours (a sleep gap; scheduled
                 agent loops post around the clock)
  R2 duplicates  >= 5 items and >= 50% of them are verbatim copies (lower-cased,
                 whitespace-collapsed) of another item by the same agent (copy-paste or
                 scripted posting rather than generated text)
  R3 burstiness  >= 10 events and Goh-Barabasi burstiness B = (sd - mean)/(sd + mean)
                 of inter-event gaps >= 0.5 (human-like bursts; periodic loops give B < 0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, get_logger, load_variant, seed_everything, write_json

log = get_logger("s08_autonomy")

RULES = {
    "R1_cadence": {"min_events": 30, "min_span_days": 2.0, "min_empty_hour_run": 6},
    "R2_duplicates": {"min_items": 5, "min_duplicate_share": 0.5},
    "R3_burstiness": {"min_events": 10, "min_B": 0.5},
}


def longest_circular_zero_run(h: np.ndarray) -> int:
    if not (h == 0).any():
        return 0
    if (h == 0).all():
        return 24
    z = np.concatenate([h, h]) == 0
    best = run = 0
    for v in z:
        run = run + 1 if v else 0
        best = max(best, run)
    return min(best, 24)


def main() -> None:
    seed_everything()
    d = load_variant("main")
    ev = pd.concat([d["posts"][["agent_id", "created_at", "content"]], d["comments"][["agent_id", "created_at", "content"]]])
    ev = ev.dropna(subset=["agent_id", "created_at"]).sort_values(["agent_id", "created_at"], kind="mergesort")
    ev["norm"] = ev["content"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    ev["hour"] = ev["created_at"].dt.hour
    g = ev.groupby("agent_id", sort=True)
    A = pd.DataFrame({"n_events": g.size(), "first": g["created_at"].min(), "last": g["created_at"].max()})
    A["span_days"] = (A["last"] - A["first"]).dt.total_seconds() / 86400

    hours = ev.groupby(["agent_id", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    A["max_empty_hour_run"] = [longest_circular_zero_run(r) for r in hours.reindex(A.index).to_numpy()]

    nonempty = ev[ev["norm"].str.len() > 0]
    dup = nonempty.groupby(["agent_id", "norm"]).size().rename("c").reset_index()
    dup_items = dup.assign(d=np.where(dup["c"] > 1, dup["c"], 0)).groupby("agent_id")[["c", "d"]].sum()
    A["n_text_items"] = dup_items["c"].reindex(A.index, fill_value=0)
    A["duplicate_share"] = (dup_items["d"] / dup_items["c"]).reindex(A.index).fillna(0.0)

    ev["gap"] = ev.groupby("agent_id")["created_at"].diff().dt.total_seconds()
    gs = ev.dropna(subset=["gap"]).groupby("agent_id")["gap"].agg(["mean", "std"])
    B = (gs["std"] - gs["mean"]) / (gs["std"] + gs["mean"])
    A["burstiness_B"] = B.reindex(A.index)
    A["gap_cov"] = (gs["std"] / gs["mean"]).reindex(A.index)

    r1, r2, r3 = RULES["R1_cadence"], RULES["R2_duplicates"], RULES["R3_burstiness"]
    A["R1_cadence"] = (A["n_events"] >= r1["min_events"]) & (A["span_days"] >= r1["min_span_days"]) & \
                      (A["max_empty_hour_run"] >= r1["min_empty_hour_run"])
    A["R2_duplicates"] = (A["n_text_items"] >= r2["min_items"]) & (A["duplicate_share"] >= r2["min_duplicate_share"])
    A["R3_burstiness"] = (A["n_events"] >= r3["min_events"]) & (A["burstiness_B"] >= r3["min_B"])
    A["flagged"] = A[["R1_cadence", "R2_duplicates", "R3_burstiness"]].any(axis=1)
    A = A.drop(columns=["first", "last"]).reset_index()
    A.to_csv(RESULTS / "autonomy_flagged_agents.csv", index=False)

    flagged = set(A.loc[A["flagged"], "agent_id"])
    n_items = len(ev)
    out = {
        "rules": RULES,
        "scored_agents": int(len(A)),
        "eligible": {"R1_cadence": int(((A["n_events"] >= r1["min_events"]) & (A["span_days"] >= r1["min_span_days"])).sum()),
                     "R2_duplicates": int((A["n_text_items"] >= r2["min_items"]).sum()),
                     "R3_burstiness": int((A["n_events"] >= r3["min_events"]).sum())},
        "flag_counts": {k: int(A[k].sum()) for k in ("R1_cadence", "R2_duplicates", "R3_burstiness")},
        "n_flagged": int(A["flagged"].sum()),
        "share_of_active_agents_flagged": float(A["flagged"].mean()),
        "share_of_items_by_flagged": float(ev["agent_id"].isin(flagged).sum() / n_items),
        "share_of_posts_by_flagged": float(d["posts"]["agent_id"].isin(flagged).mean()),
        "share_of_comments_by_flagged": float(d["comments"]["agent_id"].isin(flagged).mean()),
        "overlap": {"R1_and_R3": int((A["R1_cadence"] & A["R3_burstiness"]).sum()),
                    "R2_and_R3": int((A["R2_duplicates"] & A["R3_burstiness"]).sum()),
                    "R1_and_R2": int((A["R1_cadence"] & A["R2_duplicates"]).sum())},
        "burstiness_quantiles": {str(q): float(A["burstiness_B"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }
    write_json(RESULTS / "autonomy_audit.json", out)
    log.info("flagged %d of %d agents (%s)", out["n_flagged"], out["scored_agents"], out["flag_counts"])


if __name__ == "__main__":
    main()
