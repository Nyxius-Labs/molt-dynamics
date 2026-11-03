"""Vectorised n-gram cascade extraction.

Reproduces molt_dynamics.rq2_diffusion.CascadeIdentifier.identify_meme_cascades:
tokens are re.findall(r'\\b\\w+\\b', text.lower()); an n-gram is kept if its joined
length is >= 10 characters and at most half of its words are in the code's stopword
list; a cascade is an n-gram used by >= min_adopters distinct agents. Instead of a
Python dict of n-gram strings, n-grams are 64-bit polynomial hashes of 64-bit token
hashes (pandas.util.hash_array), so the full archive fits in memory.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"\b\w+\b")
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "are", "have",
    "was", "were", "been", "being", "would", "could", "should", "will",
    "can", "may", "might", "must", "shall", "need", "want", "like",
}
MIN_NGRAM_CHARS = 10
PRIME = np.uint64(0x100000001B3)
N_SALT = np.uint64(0x9E3779B97F4A7C15)


def tokenize(texts: list[str]):
    toks: list[str] = []
    lens = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        w = TOKEN_RE.findall(t.lower())
        lens[i] = len(w)
        toks.extend(w)
    arr = np.array(toks, dtype=object)
    th = pd.util.hash_array(arr, categorize=True) if len(arr) else np.zeros(0, np.uint64)
    tl = np.fromiter((len(x) for x in toks), dtype=np.int32, count=len(toks))
    ts = pd.Series(arr, dtype=object).isin(STOPWORDS).to_numpy() if len(arr) else np.zeros(0, bool)
    return th, tl, ts, lens


def ngram_positions(th, tl, ts, doc_lens, n):
    """Hash and doc index of every valid n-gram start position."""
    total = len(th)
    if total < n:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    doc = np.repeat(np.arange(len(doc_lens)), doc_lens)
    m = total - n + 1
    same = doc[:m] == doc[n - 1:n - 1 + m]
    cl = np.concatenate([[0], np.cumsum(tl, dtype=np.int64)])
    cs = np.concatenate([[0], np.cumsum(ts, dtype=np.int64)])
    chars = cl[n:n + m] - cl[:m] + (n - 1)
    stops = cs[n:n + m] - cs[:m]
    ok = same & (chars >= MIN_NGRAM_CHARS) & ~(stops > n * 0.5)
    with np.errstate(over="ignore"):
        h = th[:m].copy()
        for j in range(1, n):
            h = h * PRIME + th[j:j + m]
        h = h ^ (N_SALT * np.uint64(n))
    idx = np.nonzero(ok)[0]
    starts = np.concatenate([[0], np.cumsum(doc_lens)])[:-1]
    return h[idx], doc[idx], idx - starts[doc[idx]]


def _chunk_job(args):
    texts, base, ns = args
    th, tl, ts, lens = tokenize(texts)
    out = {}
    for n in ns:
        h, d, _ = ngram_positions(th, tl, ts, lens, n)
        pair = np.unique(np.rec.fromarrays([h, d.astype(np.int64)], names="h,d"))
        out[n] = (pair["h"].astype(np.uint64), pair["d"].astype(np.int64) + base)
    return out


def extract(texts: list[str], agent: np.ndarray, tms: np.ndarray, ns=(2, 3, 4, 5),
            min_keep: int = 10, thresholds=(2, 5, 10), n_jobs: int = 8, chunk: int = 20000):
    """Return (adoptions, counts).

    adoptions: DataFrame of first adoption per (n-gram, agent) for n-grams with
    >= min_keep distinct agents: columns h, n, agent, t, item.
    counts: number of distinct n-grams reaching each adopter threshold, per n.
    """
    jobs = [(texts[s:s + chunk], s, ns) for s in range(0, len(texts), chunk)]
    per_n = {n: ([], []) for n in ns}
    with ProcessPoolExecutor(n_jobs) as ex:
        for res in ex.map(_chunk_job, jobs):
            for n in ns:
                per_n[n][0].append(res[n][0])
                per_n[n][1].append(res[n][1])
    frames, counts = [], {}
    for n in ns:
        h = np.concatenate(per_n[n][0])
        it = np.concatenate(per_n[n][1])
        per_n[n] = None
        a = agent[it]
        t = tms[it]
        o = np.lexsort((it, t, a, h))
        h, a, t, it = h[o], a[o], t[o], it[o]
        first = np.ones(len(h), bool)
        first[1:] = (h[1:] != h[:-1]) | (a[1:] != a[:-1])
        h, a, t, it = h[first], a[first], t[first], it[first]
        bound = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
        sizes = np.diff(bound)
        counts[n] = {f">={k}": int((sizes >= k).sum()) for k in thresholds}
        counts[n]["occurrence_pairs"] = int(len(h))
        keep_grp = sizes >= min_keep
        keep = np.repeat(keep_grp, sizes)
        frames.append(pd.DataFrame({"h": h[keep], "n": np.int8(n), "agent": a[keep], "t": t[keep],
                                    "item": it[keep]}))
        del h, a, t, it
    return pd.concat(frames, ignore_index=True), counts


def recover_text(text: str, h: np.uint64, n: int) -> str:
    th, tl, ts, lens = tokenize([text])
    hh, _, pos = ngram_positions(th, tl, ts, lens, n)
    hit = np.flatnonzero(hh == h)
    if not len(hit):
        return ""
    words = TOKEN_RE.findall(text.lower())
    p = int(pos[hit[0]])
    return " ".join(words[p:p + n])
