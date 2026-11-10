"""Run this FIRST. Verifies parsing (hex/decimal, timestamp unit), row counts vs. the published counts, and the rule-based labels.
   python inspect_data.py --root /path/to/csvs [--combined /path/to/CAN_HCRL_OTIDS_UB.csv]"""
import argparse, json, os
import numpy as np
from opencan.data import prepare, read_stream, rule_labels, CLASSES, EXPECTED, IMP_ID

p = argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--hex", default="auto"); p.add_argument("--max-rows", type=int, default=None)
p.add_argument("--file-classes", nargs="*", default=[]); p.add_argument("--no-auto-fallback", action="store_true"); p.add_argument("--combined", default=None); p.add_argument("--out", default="results"); a = p.parse_args()
ds = prepare(a.root, hex_mode=a.hex, max_rows=a.max_rows, feat_mode="raw", file_classes=a.file_classes, auto_fallback=not a.no_auto_fallback); R = ds.report
print("\nSanity checks"); ok = True
for c in CLASSES:
    r = R[c]; frac = r["n_atk"] / max(r["n"], 1)
    print(f"  {c:14s} rows={r['n']:>9,}/{r['expected']:>9,}  rule-attack={r['n_atk']:>9,} ({frac:6.1%})  ID0x000={r['id0']:,}  ID0x164(hex)={r['id164_hex']:,} (dec164={r['id164_dec']:,})")
    print(f"  {'':14s} top IDs (hex:count): " + ", ".join(f"{i:03x}:{n:,}" for i, n in r["top_ids"][:6]))
if a.max_rows is None:
    for c in CLASSES:
        if R[c]["n"] != EXPECTED[c]: print(f"  WARNING: {c} has {R[c]['n']:,} rows, expected {EXPECTED[c]:,}"); ok = False
if R["dos"]["id0"] < 0.1 * R["dos"]["n"]: print("  WARNING: few ID 0x000 messages in dos file -> check --hex / column layout"); ok = False
if R["impersonation"]["id164_hex"] < R["impersonation"]["id164_dec"]: print("  WARNING: ID 164 (decimal) is more frequent than 0x164 in impersonation file -> try --hex no"); ok = False
G = R["_global"]; print("  label thresholds (calibrated on attack-free validation traffic):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in G["label_thresholds"].items()}, "| file-level classes:", G["file_classes"])
if "fuzzy" not in G["file_classes"] and R["fuzzy"]["n_atk"] < 0.05 * R["fuzzy"]["n"]:
    print(f"  WARNING: only {R['fuzzy']['n_atk']:,} rule-labelled fuzzy messages ({R['fuzzy']['n_atk'] / R['fuzzy']['n']:.1%}) -> the payload-novelty rule finds almost nothing. Use --file-classes fuzzy"); ok = False
fr = R["impersonation"]["imp_flag_rate"]
for c_ in ("fuzzy", "impersonation"):
    if c_ in G["file_classes"]: print(f"  {c_}: FILE-LEVEL labels in use (rules flagged {R[c_]['rule_n_atk']:,} of {R[c_]['n']:,} messages)")
if "impersonation" not in G["file_classes"] and fr is not None:
    print(f"  impersonation: {fr:.1%} of ID-0x164 messages pass the anomaly confirmation (benign calibration rate ~{G['label_thresholds']['fpr_imp']:.0%}-{2 * G['label_thresholds']['fpr_imp']:.0%})")
    if fr < 0.05: print("  WARNING: 0x164 messages in the impersonation file look statistically like benign ones -> use --file-classes impersonation"); ok = False
print("  ts scale:", R["_global"]["ts_scale"], "| legit IDs:", R["_global"]["legit_ids"], "| checks:", "OK" if ok else "REVIEW WARNINGS")
if a.combined:
    s = read_stream(a.combined, "no", a.max_rows); ids = s["ids"]; n = len(ids)
    lab = np.where(ids == 0, 1, np.where(ids == IMP_ID, 3, np.where(~ds.stats.legit[ids], 2, 0)))
    print(f"\nRule labels on {os.path.basename(a.combined)} (n={n:,}) [combined file has NO time order -> not used for windowing]")
    for k, c in enumerate(CLASSES): print(f"  {c:14s} {int((lab == k).sum()):>9,}  (published total {EXPECTED[c]:,})")
    print("  Note: 0x164 also occurs in attack-free traffic, so the impersonation rule over-counts on shuffled data.")
os.makedirs(a.out, exist_ok=True); json.dump(R, open(os.path.join(a.out, "data_stats.json"), "w"), indent=1); print(f"\nsaved {a.out}/data_stats.json")
