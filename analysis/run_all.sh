#!/usr/bin/env bash
# Reproduce every result in results/ from the raw archive.
#
# Usage: analysis/run_all.sh            (uses .venv/bin/python by default)
#        PY=/path/to/python analysis/run_all.sh
#        CLEAN=1 analysis/run_all.sh    (also deletes cached intermediates in work/, incl. embeddings)
#
# Raw downloads (data_raw/, moltbook-observatory-archive/) are skipped when already present.
# Runtime on 8 CPU cores without a GPU: about 2.5 hours, most of it embedding ~520k texts.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

step() { echo "=== $(date +%H:%M:%S) $*"; "$PY" "$@"; }

rm -rf ../results
mkdir -p ../results ../work
if [[ "${CLEAN:-0}" == "1" ]]; then rm -rf ../work/*; fi

step s01_data.py
step s08_autonomy.py

for v in main no_autonomy no_incident; do
  step s02_structure.py --variant "$v"
  step s03_cascades.py --variant "$v"
done
step s03b_embed_cascades.py
step s05_powerlaw.py --variant main --gof-iters 200
for v in main no_autonomy no_incident; do
  step s04_exposure.py --variant "$v"
  step s06_collaboration.py --variant "$v"
done
step s07_q_validation.py

step s02_structure.py --variant merged
step s06_collaboration.py --variant merged

step s10_human_platform.py
step s11_assemble.py
echo "=== $(date +%H:%M:%S) done"
