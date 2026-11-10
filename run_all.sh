#!/usr/bin/env bash
# Usage: bash run_all.sh /path/to/folder_with_dataset1-4_csv [SEEDS] ["extra flags for ALL runs"]
# Example: bash run_all.sh archive 3 "--file-classes impersonation"
set -e
DATA=$1; SEEDS=${2:-3}; EXTRA=${3:-}
python inspect_data.py --root "$DATA" --out results $EXTRA                 # 1) check parsing + rule labels FIRST
R="python run_experiments.py --root $DATA --seeds $SEEDS $EXTRA"
$R --out results/full
$R --out results/abl_nopseudo --pseudo none
$R --out results/abl_raw      --features raw
$R --out results/baseline     --features raw --pseudo none
$R --out results/split_random --split random
python make_plots.py --main results/full --ablations results/abl_nopseudo results/abl_raw results/baseline \
                     --split-random results/split_random --out overleaf     # 2) figures/ + tables/ for the paper
echo "Copy overleaf/figures and overleaf/tables into the Overleaf project, then recompile."
