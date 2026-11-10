# OpenCAN-X — explainable open-set intrusion detection for CAN (CAN_HCRL_OTIDS)

## Input format (as described by you)
`dataset1.csv` attack-free, `dataset2.csv` DoS (ID 0x000), `dataset3.csv` fuzzy (random IDs/data), `dataset4.csv` impersonation (ID 0x164).
Header `TS,ID1,ID0,LEN,DLC0..DLC5` (payload columns are read generically: 8 bytes after `LEN`, missing bytes = 0, ragged rows allowed;
the combined layout `TS,ID1,DLC0..DLC7` also parses). Hex vs decimal is auto-detected (`--hex yes|no` to force).
The shuffled/SMOTE combined files are NOT used for training: windows, temporal splits and per-ID periodicity need time order.
Labels are derived by rule: DoS -> ID==0; Impersonation -> ID==0x164; Fuzzy -> ID not in the legitimate-ID set
(IDs seen >= `--legit-min` times in attack-free TRAINING traffic); windows need >= `--min-atk` (3) attack messages.

## Steps
    pip install -r requirements.txt
    python inspect_data.py --root DATA            # row counts vs published, hex/timestamp checks, rule-label stats
    bash run_all.sh DATA 3                        # all experiments + figures/tables (or run commands one by one)
Quick trial first:  `python run_experiments.py --root DATA --max-rows 300000 --seeds 1 --epochs 3 --out results/try`
Smoke test without data: `python -m opencan.mock --out mock_data && python run_experiments.py --root mock_data --seeds 1 --epochs 2 --epoch-size 3000 --pseudo-msgs 8000 --fit-max 5000 --out mock_res`

## Outputs -> paper
`make_plots.py` writes `overleaf/figures/fig_*.pdf` and `overleaf/tables/tab_*.tex` with the exact names used in `main.tex`.

## Design summary
features.py CAN-aware features (periodicity residual, payload drift, ID novelty; fitted on attack-free train only) · pseudo.py mutation
operators + operator hold-out · model.py 3-head CNN + SE + temporal attention pooling (K+1 outputs) · openset.py MSP/Energy/Mahalanobis/
Reject-class, fusion, Mondrian conformal thresholds · explain.py deletion/insertion + localisation vs message-level ground truth.

## Notes
* Tested end-to-end only on mock data in this format; NOT run on the real 4.6M-message files. Watch the warnings from `inspect_data.py`.
* GPU is used automatically if available. Peak RAM is a few GB; use `--max-rows` if memory is tight.
* If TS is not in seconds the unit is auto-detected (`--ts-scale` to override); pseudo-unknown rates depend on it.

## Troubleshooting (from a real run)
* `ValueError: Input contains NaN` / a class with 0 windows: the labelling rules found no attack messages for some class. The runner now stops
  with a message and per-class window counts. Run `inspect_data.py`, then if a rule does not fit use file-level labels, e.g.
  `--file-classes fuzzy impersonation` (pass the same flag to run_all.sh commands / make runs comparable).
* In the real data the fuzzy file only uses legitimate IDs, so fuzzy messages are detected by payload novelty (random bytes on a known ID);
  impersonation messages (ID 0x164) are confirmed by payload/timing anomalies. Thresholds are calibrated on attack-free validation traffic.
* If a rule labels <10% of a file's windows (fuzzy/impersonation), the code automatically switches that class to FILE-LEVEL labels and prints a NOTE
  (disable with `--no-auto-fallback`). On the real data this happens for impersonation.
* `normal` having ~470 fewer rows than the published count is usually harmless; compare with `wc -l dataset1.csv` (minus the header).
