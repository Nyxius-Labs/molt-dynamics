# Emergent, Steered, or Neither? A Validity Audit of the MoltBook Agent Society

Code and results for the paper presented at the KDD 2026 Workshop on Evaluation and Trustworthiness of Agentic AI ([PDF](https://kdd-eval-workshop.github.io/agenticai-evaluation-kdd2026/assets/papers/52_Emergent_Steered_or_Neither.pdf)).

## Data

The analysis uses the [MoltBook Observatory Archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive) maintained by Sushant Gautam and Michael A. Riegler, restricted to 28 January to 20 February 2026 (1,500,385 posts, 224,210 comments, 161,911 agents).

```bash
git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
pip install -r requirements.txt
```

`dcsbm_null.py` also needs graph-tool (`conda install -c conda-forge graph-tool`), and `llm_judge.py` needs vLLM and a GPU.

## Main pipeline

`run_rigor.py` builds the directed reply network and computes every structural, diffusion and collaboration result in the paper, including the configuration-model nulls, Leiden modularity, the cascade fits with the Clauset goodness-of-fit test, the exposure hazard models and the matched collaboration comparison.

```bash
python scripts/run_rigor.py --data-dir moltbook-observatory-archive/data --out results/rigor_final.json --shuffle-iters 30
python scripts/run_rigor.py --data-dir moltbook-observatory-archive/data --out results/rigor_excl_final.json --shuffle-iters 30 --exclude-dates 2026-01-31,2026-02-09
python scripts/run_rigor.py --data-dir moltbook-observatory-archive/data --start 2026-03-01 --end 2026-03-21 --out results/replication_march.json --null-iters 50 --shuffle-iters 25 --gof-iters 50
```

## Further analyses

| Script | Output in `results/` | Paper section |
|---|---|---|
| `subwindows.py` | `subwindow_invariance.json` | Structure, weekly sub-windows |
| `replicate_aisilab.py` | `replication_aisilab.json` | Structure, second collection |
| `dcsbm_null.py` | `dcsbm_null.json` | Structure, degree-corrected block-model null |
| `equivalence.py` | `equivalence.json` | Diffusion, equivalence test on the exposure hazard |
| `robustness_panel.py` | `robustness_panel.json` | Collaboration, matched comparison with bootstrap and power checks |
| `make_calibration_set.py`, `llm_judge.py` | `judge_threads.parquet`, `llm_judge_scores.parquet`, `judge_validation.json`, `intercoder_reliability.json` | Collaboration, language-model judge |
| `collab_cem.py`, `collab_doseresponse.py`, `collab_robustness.py`, `check_human_slope.py` | `collab_*.json` | Collaboration, matched and continuous designs |
| `shutdown_validation.py`, `diag_shutdown.py` | `shutdown_validation*.json` | Autonomy, platform-shutdown validation |
| `salvage_labels.py`, `validate_selfdisclosure.py` | `salvage_labels.json`, `selfdisclosure_sample.json` | Autonomy, self-disclosure and ownership labels |
| `salvage_classifier.py`, `verify_polarity.py`, `burstiness_corrected.py` | `salvage_classifier.json`, `verify_polarity.json`, `burstiness_corrected.json` | Autonomy, multi-feature classifier and corrected burstiness |
| `salvage_mix.py` | `salvage_mix.json` | Autonomy, share of activity across thresholds |
| `dump_figdata.py` | `figdata.json` | Figures |

`results/human_labels.csv` holds the hand codes used to validate the judge. Every script takes `--help`.

## License

MIT
