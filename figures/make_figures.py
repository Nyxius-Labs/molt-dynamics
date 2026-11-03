"""Draw the paper's four figures from the CSV/JSON outputs in results/.

Usage: python figures/make_figures.py [RESULTS_DIR] [OUT_DIR]
Defaults: ../results relative to the repo root, and figures/ next to this script.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1_clusters():
    t = pd.read_csv(RES / "table_network_clusters.csv").sort_values("cluster")
    feats = ["in_degree", "out_degree", "betweenness", "clustering_coefficient", "pagerank"]
    labels = ["In-degree", "Out-degree", "Betweenness", "Clustering", "PageRank"]
    z = t[[f"z_{f}" for f in feats]].to_numpy()

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    y = np.arange(len(t))
    a.barh(y, t["count"], color=BLUE, height=0.6)
    a.set_xscale("log")
    a.set_yticks(y, [f"Cluster {c}" for c in t["cluster"]])
    a.invert_yaxis()
    a.set_xlabel("Agents (log scale)")
    a.grid(axis="y", visible=False)
    for yi, (n, p) in enumerate(zip(t["count"], t["proportion"])):
        a.text(n * 1.25, yi, f"{n:,} ({p:.1%})" if p >= 0.001 else f"{n:,}", va="center",
               fontsize=7, color=INK2)
    a.set_xlim(0.7, 1e7)
    a.set_title("(a) Cluster sizes", loc="left", color=INK)

    im = b.imshow(np.clip(z, 0, 10), cmap=SEQ, vmin=0, vmax=10, aspect="auto")
    b.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    b.set_yticks(y, [f"{c}" for c in t["cluster"]])
    b.set_ylabel("Cluster")
    b.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            b.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                   color="#ffffff" if v > 5 else INK)
    cb = fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    cb.set_label("Mean z-score (clipped at 10)", color=INK2)
    cb.outline.set_linewidth(0.4)
    b.set_title("(b) Standardized feature means", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig1_clusters")


def fig2_ccdf():
    d = pd.read_csv(RES / "figure2_ccdf.csv")
    fits = json.load(open(RES / "results.json"))["fig2_ccdf"]["fits"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    panels = [
        ("primary_uncapped_n2to5", "(a) All cascades, uncapped ($n$=2..5)"),
        ("reproduction_capped_n3to5", "(b) 10,000-cascade cap ($n$=3..5)"),
    ]
    for ax, (s, title) in zip(axes, panels):
        x = d[d.set == s].sort_values("x")
        ax.plot(x.x, x.empirical_ccdf, "o", ms=2.2, mfc="none", mec=MUTED, mew=0.6,
                label="Empirical", zorder=2)
        tail = x.dropna(subset=["fit_power_law"])
        f = fits[s]
        ax.plot(tail.x, tail.fit_power_law, color=BLUE, lw=1.6,
                label=f"Power law ($\\alpha$={f['alpha']:.2f})", zorder=3)
        ax.plot(tail.x, tail.fit_lognormal, color=ORANGE, lw=1.6, ls="--", label="Lognormal", zorder=3)
        ax.plot(tail.x, tail.fit_truncated_power_law, color=AQUA, lw=1.6, ls=":",
                label="Truncated power law", zorder=3)
        ax.axvline(f["xmin"], color=MUTED, lw=0.6, ls="-", zorder=1)
        ax.text(f["xmin"] * 1.1, 2e-6, f"$x_{{\\min}}$={f['xmin']:.0f}", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cascade size (distinct adopters)")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(1e-6, 1.5)
    axes[0].set_ylabel("P(size $\\geq$ x)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig2_ccdf")


def fig3_exposure():
    b = pd.read_csv(RES / "figure3_binned_adoption.csv")
    b = b[b.exposure_bin.astype(str).str.isdigit()].copy()
    b["e"] = b.exposure_bin.astype(int)
    b = b[b.n_agent_cascade_pairs >= 1000]
    lc = pd.read_csv(RES / "figure3_logistic_curve.csv")
    lc = lc[lc.exposure <= b.e.max()]
    cox = json.load(open(RES / "results.json"))["app_cox"]

    fig, (a, c) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    a.errorbar(b.e, b.adoption_rate * 100,
               yerr=[(b.adoption_rate - b.wilson_lower) * 100, (b.wilson_upper - b.adoption_rate) * 100],
               fmt="o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=0.8, label="Observed (95% Wilson)", zorder=3)
    a.plot(lc.exposure, lc.logistic_predicted_rate * 100, color=ORANGE, lw=1.6, ls="--",
           label="Quadratic logistic fit", zorder=2)
    a.set_xlabel("Adopting in-neighbors (static count)")
    a.set_ylabel("Adoption rate (%)")
    a.set_title("(a) Adoption by exposure", loc="left", color=INK)
    a.legend(loc="upper left")

    rows = [
        ("random100_static_pooled_clustered", "Static, pooled"),
        ("random100_static_stratified", "Static, stratified"),
        ("random100_timevarying_pooled", "Time-varying, pooled"),
        ("random100_timevarying_stratified", "Time-varying, stratified"),
    ]
    for i, (k, lab) in enumerate(rows):
        r = cox[k]
        lo, hi = r["hr_ci95"]
        col = AQUA if "timevarying" in k else ORANGE
        mk = "s" if "timevarying" in k else "D"
        c.errorbar(r["hazard_ratio"], i, xerr=[[r["hazard_ratio"] - lo], [hi - r["hazard_ratio"]]],
                   fmt=mk, ms=4.5, color=col, elinewidth=1.0, capsize=2)
        c.text(r["hazard_ratio"] * (0.55 if "timevarying" in k else 1), i - 0.3, f"{r['hazard_ratio']:.3f}",
               ha="right" if "timevarying" in k else "center", fontsize=7, color=INK2)
    c.axvline(1, color=INK2, lw=0.8)
    c.set_xscale("log")
    c.set_xlim(1e-3, 4)
    c.set_yticks(range(len(rows)), [r[1] for r in rows])
    c.invert_yaxis()
    c.set_ylim(3.6, -0.7)
    c.set_xlabel("Hazard ratio per exposure (log)")
    c.grid(axis="y", visible=False)
    c.set_title("(b) Cox models, 100 random cascades", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig3_exposure")


def fig4_participants():
    d = pd.read_csv(RES / "figure4_quality_by_participants.csv")
    dr = json.load(open(RES / "results.json"))["fig4_participants"]["dose_response"]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (q, col, title) in zip(axes, [("Q_spec", BLUE, "(a) $Q_{\\mathrm{spec}}$"),
                                          ("Q_code", ORANGE, "(b) $Q_{\\mathrm{code}}$")]):
        jit = np.exp(rng.uniform(-0.06, 0.06, len(d)))
        ax.scatter(d.participants * jit, d[q], s=10, color=col, alpha=0.55, edgecolor="white", linewidth=0.4)
        e = dr[q]["events"]
        xs = np.linspace(d.participants.min(), d.participants.max(), 100)
        ax.plot(xs, e["intercept"] + e["slope"] * np.log(xs), color=INK, lw=1.4,
                label=f"OLS slope {e['slope']:.3f} (p = {e['p']:.3f})")
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=":")
        ax.text(3.0, 0.515, "success threshold", ha="left", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("Participants (log scale)")
        ax.set_title(title, loc="left", color=INK)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.03, 1.12)
    axes[0].set_ylabel("Quality score")
    fig.tight_layout()
    save(fig, "fig4_participants")


if __name__ == "__main__":
    fig1_clusters()
    fig2_ccdf()
    fig3_exposure()
    fig4_participants()
    print("wrote", sorted(p.name for p in OUT.glob("fig*")))
