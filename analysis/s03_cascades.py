"""Step 3a: n-gram (meme), code-block (skill) and formatting (behavioral) cascades.

Follows molt_dynamics.rq2_diffusion.CascadeIdentifier. The earlier run called
identify_meme_cascades() with its defaults (n = 3..5, >= 10 distinct adopters) and the
method truncated the sorted list with `cascades[:10000]`; both the capped reproduction
and the uncapped extraction are produced here, for n = 3..5 and for n = 2..5.
"""

from __future__ import annotations

import argparse
import hashlib
import re

import numpy as np
import pandas as pd

from common import RESULTS, WORK, get_logger, load_variant, seed_everything, suffix, write_json
from ngram_engine import extract, recover_text

log = get_logger("s03_cascades")

MEME_MIN_ADOPTERS = 10
SKILL_MIN_ADOPTERS = 5
BEHAVIORAL_MIN_ADOPTERS = 5
CODE_CAP = 10000
CODE_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")
BEHAVIORAL_PATTERNS = {
    "emoji_heavy": r"[\U0001F600-\U0001F64F]{3,}",
    "bullet_list": r"(?m)^\s*[-*]\s+",
    "numbered_list": r"(?m)^\s*\d+\.\s+",
    "header_style": r"(?m)^#+\s+",
    "quote_style": r"(?m)^>\s+",
    "link_heavy": r"\[.*?\]\(.*?\)",
}


def items_table(d: dict) -> pd.DataFrame:
    p = d["posts"][["id", "agent_id", "content", "created_at"]].assign(kind="post")
    c = d["comments"][["id", "agent_id", "content", "created_at"]].assign(kind="comment")
    it = pd.concat([p, c], ignore_index=True)
    it = it[(it["content"].str.len() > 0) & it["created_at"].notna() & it["agent_id"].notna()]
    return it.reset_index(drop=True)


def first_adoptions(keys: pd.Series, it: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"key": keys.to_numpy(), "agent": it["agent_id"].to_numpy(),
                       "t": it["created_at"].to_numpy()})
    return df.sort_values("t", kind="mergesort").drop_duplicates(["key", "agent"]).reset_index(drop=True)


def skill_cascades(it: pd.DataFrame) -> pd.DataFrame:
    ex = it["content"].str.findall(CODE_RE)
    ex = ex[ex.str.len() > 0].explode()
    keys = ex.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])
    fa = first_adoptions(keys, it.loc[ex.index])
    sizes = fa.groupby("key")["agent"].transform("size")
    fa = fa[sizes >= SKILL_MIN_ADOPTERS]
    fa["cascade_id"] = "skill_" + fa["key"]
    return fa[["cascade_id", "agent", "t"]]


def behavioral_cascades(it: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, pat in BEHAVIORAL_PATTERNS.items():
        hit = it["content"].str.contains(pat, regex=True)
        sub = it[hit]
        fa = first_adoptions(pd.Series(name, index=sub.index), sub)
        if fa["agent"].nunique() >= BEHAVIORAL_MIN_ADOPTERS:
            fa["cascade_id"] = "behavioral_" + name
            out.append(fa[["cascade_id", "agent", "t"]])
    return pd.concat(out, ignore_index=True)


def summarize(adopt: pd.DataFrame, ctype: str) -> pd.DataFrame:
    a = adopt.sort_values(["cascade_id", "t"], kind="mergesort")
    g = a.groupby("cascade_id", sort=False)
    s = g.agg(size=("agent", "size"), seed_time=("t", "first"), seed_agent=("agent", "first"),
              last_time=("t", "last")).reset_index()
    s["duration_hours"] = (s["last_time"] - s["seed_time"]).dt.total_seconds() / 3600
    s["cascade_type"] = ctype
    return s.drop(columns="last_time")


def run(variant: str) -> dict:
    seed_everything()
    sfx = suffix(variant)
    d = load_variant(variant)
    it = items_table(d)
    agents_codes, agents_uni = pd.factorize(it["agent_id"], sort=True)
    tms = it["created_at"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    log.info("%d items (%d posts, %d comments)", len(it), (it["kind"] == "post").sum(), (it["kind"] == "comment").sum())

    ad, counts = extract(it["content"].tolist(), agents_codes.astype(np.int64), tms, ns=(2, 3, 4, 5),
                         min_keep=MEME_MIN_ADOPTERS, thresholds=(2, 5, 10))
    log.info("meme n-gram adoptions kept: %d rows", len(ad))
    ad["cascade_id"] = ad["n"].astype(str) + "g_" + pd.Series(ad["h"].to_numpy()).map("{:016x}".format)
    meme_ad = pd.DataFrame({"cascade_id": ad["cascade_id"].to_numpy(),
                            "agent": agents_uni[ad["agent"].to_numpy()],
                            "t": pd.to_datetime(ad["t"].to_numpy(), unit="ms"),
                            "item": ad["item"].to_numpy()})
    memes = summarize(meme_ad, "meme")
    first_item = meme_ad.sort_values(["cascade_id", "t"]).drop_duplicates("cascade_id").set_index("cascade_id")["item"]
    memes["n"] = memes["cascade_id"].str.slice(0, 1).astype(int)
    memes["seed_item"] = memes["cascade_id"].map(first_item)

    def capped(df):
        return df.sort_values(["size", "seed_time"], ascending=[False, True], kind="mergesort").head(CODE_CAP)

    m35 = memes[memes["n"] >= 3]
    cap35 = capped(m35)
    cap25 = capped(memes)
    memes["in_cap_n3to5"] = memes["cascade_id"].isin(set(cap35["cascade_id"]))
    memes["in_cap_n2to5"] = memes["cascade_id"].isin(set(cap25["cascade_id"]))

    texts = it["content"].to_numpy()
    show = set(cap35["cascade_id"]) | set(memes.nlargest(2000, "size")["cascade_id"])
    hcol = memes["cascade_id"].str.slice(3).map(lambda x: np.uint64(int(x, 16)))
    memes["ngram"] = ""
    sel = memes["cascade_id"].isin(show)
    memes.loc[sel, "ngram"] = [recover_text(texts[i], h, n) for i, h, n in
                               zip(memes.loc[sel, "seed_item"], hcol[sel], memes.loc[sel, "n"])]

    skill_ad = skill_cascades(it)
    beh_ad = behavioral_cascades(it)
    skills = summarize(skill_ad, "skill")
    behs = summarize(beh_ad, "behavioral")
    for df in (skills, behs):
        df["n"] = 0
        df["in_cap_n3to5"] = True
        df["in_cap_n2to5"] = True
        df["ngram"] = ""
    cols = ["cascade_id", "cascade_type", "n", "size", "seed_time", "seed_agent", "duration_hours",
            "in_cap_n3to5", "in_cap_n2to5", "ngram"]
    allc = pd.concat([memes[cols], skills[cols], behs[cols]], ignore_index=True)
    allc.to_csv(RESULTS / f"cascades{sfx}.csv", index=False)
    pd.concat([meme_ad.drop(columns="item"), skill_ad, beh_ad], ignore_index=True).to_parquet(
        WORK / f"adoptions{sfx}.parquet", index=False)

    n_all_35 = int(len(m35))
    res = {
        "variant": variant,
        "n_items": int(len(it)),
        "extraction": {"tokenizer": r"re.findall(r'\b\w+\b', text.lower())", "min_ngram_chars": 10,
                       "stopword_rule": "drop n-gram if more than half of its words are stopwords",
                       "meme_min_distinct_adopters": MEME_MIN_ADOPTERS,
                       "skill_min_distinct_adopters": SKILL_MIN_ADOPTERS,
                       "behavioral_min_distinct_adopters": BEHAVIORAL_MIN_ADOPTERS,
                       "texts": "post content and comment content (titles excluded, as in the code)"},
        "ngram_counts_by_n": counts,
        "cap": {
            "cap_in_code": "rq2_diffusion.py: cascades.sort(by distinct adopters, desc); cascades = cascades[:10000]",
            "code_default_ngram_range": [3, 5],
            "uncapped_meme_cascades_n3to5": n_all_35,
            "uncapped_meme_cascades_n2to5": int(len(memes)),
            "capped_meme_cascades_n3to5": int(len(cap35)),
            "cap_binding": bool(n_all_35 > CODE_CAP),
            "min_size_in_capped_set_n3to5": int(cap35["size"].min()),
            "min_size_in_capped_set_n2to5": int(cap25["size"].min()),
            "tie_break": "equal sizes ordered by earliest seed time (the code's order depended on dict insertion order)",
        },
        "n_skill_cascades": int(len(skills)),
        "n_behavioral_cascades": int(len(behs)),
        "reproduction_total_capped_n3to5": int(len(cap35) + len(skills) + len(behs)),
        "primary_total_uncapped_n2to5": int(len(memes) + len(skills) + len(behs)),
        "meme_size_summary_n2to5": memes["size"].describe().to_dict(),
        "top_memes": memes.nlargest(15, "size")[["cascade_id", "n", "size", "ngram"]].to_dict("records"),
    }
    write_json(RESULTS / f"cascades{sfx}.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args()
    r = run(a.variant)
    log.info("cap: %s", r["cap"])
    log.info("counts by n: %s", r["ngram_counts_by_n"])
