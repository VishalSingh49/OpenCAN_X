"""Creates every figure (PDF) and LaTeX table used by the paper. Output names are fixed (see FIG / TAB).
python make_plots.py --main results/full --ablations results/abl_nopseudo results/abl_raw results/baseline \
                     --split-random results/split_random --out overleaf"""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "savefig.bbox": "tight", "font.family": "serif", "figure.dpi": 150})
FIG = dict(class_dist="fig_class_distribution", id_dist="fig_id_distribution", train="fig_training_curves", cm="fig_confusion_matrix",
           roc="fig_roc_curves", scoredist="fig_score_distributions", auroc="fig_openset_auroc", calib="fig_calibration",
           faith="fig_faithfulness_curves", loc="fig_localization", ablation="fig_ablation", split="fig_split_comparison", attn="fig_attention_example")
SC = ["MSP", "Energy", "Mahalanobis", "RejectClass", "Fusion-min", "Fusion-mean"]; HOLD = ["dos", "fuzzy", "impersonation"]
RANK = ["attention", "gradxinput", "occlusion", "random", "reversed_attention"]; PRIM = "Fusion-min"
ms = lambda v: (np.mean(v), np.std(v)) if len(v) else (np.nan, np.nan)
fmt = lambda v, d=3: "--" if not len(v) else f"{np.mean(v):.{d}f}$\\pm${np.std(v):.{d}f}"


def load(d):
    p = os.path.join(d, "results.json"); return json.load(open(p)) if os.path.exists(p) else []


def opens(R, h, s, m): return [r["open"][s][m] for r in R if r["holdout"] == h and s in r["open"]]


def save(fig, out, key): fig.savefig(os.path.join(out, "figures", FIG[key] + ".pdf")); plt.close(fig); print("  wrote", FIG[key] + ".pdf")


def f_class_dist(D, out):
    cs = ["normal", "dos", "fuzzy", "impersonation"]; x = np.arange(4); fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(x - .2, [D[c]["n"] - D[c]["n_atk"] for c in cs], .4, label="benign messages"); ax.bar(x + .2, [D[c]["n_atk"] for c in cs], .4, label="rule-labelled attack messages")
    ax.scatter(x, [D[c]["expected"] for c in cs], marker="_", s=300, color="k", label="published file size"); ax.set_yscale("log")
    ax.set_xticks(x, ["Attack-free", "DoS", "Fuzzy", "Imperson."]); ax.set_ylabel("messages"); ax.legend(fontsize=6); save(fig, out, "class_dist")


def f_id_dist(D, out):
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.8), sharex=True)
    for ax, c, t in zip(axs.ravel(), ["normal", "dos", "fuzzy", "impersonation"], ["Attack-free", "DoS", "Fuzzy", "Impersonation"]):
        h = np.array(D[c]["id_hist"]); ax.bar(np.arange(64) * 32, h, width=32, align="edge"); ax.set_yscale("log"); ax.set_title(t, fontsize=7)
    fig.supxlabel("CAN arbitration ID (binned)", fontsize=7); fig.supylabel("messages", fontsize=7); save(fig, out, "id_dist")


def f_train(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(3.6, 2.2))
    for h in HOLD + ["none"]:
        rs = [r for r in R if r["holdout"] == h]
        if not rs: continue
        L = max(len(r["history"]) for r in rs)
        for ax, k in zip(axs, ["train_loss", "val_macro_f1"]):
            M = np.full((len(rs), L), np.nan)
            for i, r in enumerate(rs): M[i, :len(r["history"])] = [e[k] for e in r["history"]]
            ax.plot(np.arange(1, L + 1), np.nanmean(M, 0), label=h)
    axs[0].set_ylabel("training loss"); axs[1].set_ylabel("validation macro-F1"); [a.set_xlabel("epoch") for a in axs]; axs[1].legend(fontsize=6); save(fig, out, "train")


def f_cm(R, out):
    rs = [r for r in R if r["holdout"] == "none"]
    if not rs: return
    C = np.mean([np.array(r["closed"]["confusion"], float) / np.maximum(np.sum(r["closed"]["confusion"], 1, keepdims=True), 1) for r in rs], 0)
    fig, ax = plt.subplots(figsize=(2.8, 2.6)); ax.imshow(C, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    names = ["Normal", "DoS", "Fuzzy", "Imperson."]; ax.set_xticks(range(4), names, rotation=30); ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] > .5 else "k", fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); save(fig, out, "cm")


def _npz(d, h): p = os.path.join(d, "scores", f"{h}_s0.npz"); return np.load(p) if os.path.exists(p) else None


def f_roc(d, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, h in zip(axs, HOLD):
        z = _npz(d, h)
        if z is None: continue
        for s in SC:
            if f"sk_{s}" not in z: continue
            sk, su = z[f"sk_{s}"], z[f"su_{s}"]; f, t, _ = roc_curve(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su]); ax.plot(f, t, label=s, lw=1)
        ax.plot([0, 1], [0, 1], "k:", lw=.6); ax.set_title(f"unknown = {h}", fontsize=7); ax.set_xlabel("FPR (unknown accepted)")
    axs[0].set_ylabel("TPR (known accepted)"); axs[0].legend(fontsize=5); save(fig, out, "roc")


def f_scoredist(d, out):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.4))
    for j, h in enumerate(HOLD):
        z = _npz(d, h)
        for i, s in enumerate(["MSP", PRIM]):
            ax = axs[i, j]
            if z is None or f"sk_{s}" not in z: continue
            allv = np.r_[z[f"sk_{s}"], z[f"su_{s}"]]; b = np.linspace(*np.percentile(allv, [1, 99]), 40)
            ax.hist(np.clip(z[f"sk_{s}"], b[0], b[-1]), b, alpha=.6, density=True, label="known"); ax.hist(np.clip(z[f"su_{s}"], b[0], b[-1]), b, alpha=.6, density=True, label="unknown")
            ax.set_title(f"{s} | unknown={h}", fontsize=6); ax.set_yticks([])
    axs[0, 0].legend(fontsize=6); save(fig, out, "scoredist")


def f_auroc(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4)); w = .8 / len(SC)
    for ax, m, yl in zip(axs, ["auroc", "fpr95"], ["AUROC (known vs unknown)", "FPR@95%TPR"]):
        for k, s in enumerate(SC):
            v = [ms(opens(R, h, s, m)) for h in HOLD]; ax.bar(np.arange(3) + k * w, [a for a, _ in v], w, yerr=[b for _, b in v], label=s, capsize=1.5)
        ax.set_xticks(np.arange(3) + .4 - w / 2, HOLD); ax.set_ylabel(yl)
    axs[0].axhline(.5, color="k", ls=":", lw=.6); axs[0].legend(fontsize=5, ncol=2); save(fig, out, "auroc")


def f_calib(R, out):
    al = ["0.01", "0.02", "0.05", "0.1", "0.2"]; a = [float(x) for x in al]; fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for kind in ("global", "mondrian"):
        v = [np.mean([r["conformal"][x][kind]["known_reject"] for r in R if r["conformal"]]) for x in al]; axs[0].plot(a, v, "o-", label=kind, ms=3)
    axs[0].plot(a, a, "k:", label="nominal $\\alpha$"); axs[0].set_xlabel("target false-alarm rate $\\alpha$"); axs[0].set_ylabel("achieved known-reject rate"); axs[0].legend(fontsize=6)
    for h in HOLD:
        rs = [r for r in R if r["holdout"] == h and r["conformal"]]
        if rs: axs[1].plot(a, [np.mean([r["conformal"][x]["mondrian"]["unk_recall"] for r in rs]) for x in al], "o-", label=h, ms=3)
    axs[1].set_xlabel("target false-alarm rate $\\alpha$"); axs[1].set_ylabel("unknown recall"); axs[1].legend(fontsize=6); save(fig, out, "calib")


def f_faith(R, out):
    rs = [r for r in R if "class" in r["explain"]]; fr = np.linspace(0, 1, 11); fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for ax, k, t in zip(axs, ["del_curve", "ins_curve"], ["deletion", "insertion"]):
        for n in RANK: ax.plot(fr, np.mean([r["explain"]["class"][n][k] for r in rs], 0), label=n.replace("_", " "), lw=1)
        ax.set_xlabel(f"fraction of messages {'deleted' if k[0] == 'd' else 'inserted'}"); ax.set_ylabel("prob. of predicted class"); ax.set_title(t, fontsize=7)
    axs[0].legend(fontsize=5); save(fig, out, "faith")


def f_loc(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.3), sharey=True)
    for ax, task, t in zip(axs, ["class", "unknown"], ["known-attack classification", "unknown-alert (Mahalanobis)"]):
        v = [[r["explain"][task][n]["lift"] for r in R if n in r["explain"].get(task, {}) and "lift" in r["explain"][task][n]] for n in RANK]
        ax.bar(range(len(RANK)), [np.mean(x) if x else 0 for x in v], yerr=[np.std(x) if x else 0 for x in v], capsize=2); ax.axhline(1, color="k", ls=":", lw=.6)
        ax.set_xticks(range(len(RANK)), [n.replace("_", "\n") for n in RANK], fontsize=5); ax.set_title(t, fontsize=7)
    axs[0].set_ylabel("R-precision lift over chance"); save(fig, out, "loc")


def f_ablation(variants, out):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4)); names = list(variants); w = .8 / 3
    for k, h in enumerate(HOLD):
        axs[0].bar(np.arange(len(names)) + k * w, [np.mean(opens(variants[n], h, PRIM, "auroc")) if opens(variants[n], h, PRIM, "auroc") else 0 for n in names], w, label=h)
    axs[0].set_xticks(np.arange(len(names)) + w, names, rotation=20, fontsize=5); axs[0].set_ylabel(f"AUROC ({PRIM})"); axs[0].legend(fontsize=6)
    axs[1].bar(range(len(names)), [np.mean([r["closed"]["macro_f1"] for r in variants[n]]) if variants[n] else 0 for n in names], color="C3")
    axs[1].set_xticks(range(len(names)), names, rotation=20, fontsize=5); axs[1].set_ylabel("closed-set macro-F1"); save(fig, out, "ablation")


def f_split(a, b, out):
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.2)); lab = ["temporal (ours)", "random (paper-style)"]
    axs[0].bar(lab, [np.mean([r["closed"]["acc"] for r in X]) for X in (a, b)], color=["C0", "C1"]); axs[0].set_ylabel("closed-set accuracy")
    axs[1].bar(lab, [np.mean([x for h in HOLD for x in opens(X, h, PRIM, "auroc")]) for X in (a, b)], color=["C0", "C1"]); axs[1].set_ylabel(f"mean AUROC ({PRIM})")
    for ax in axs: ax.tick_params(axis="x", labelsize=5)
    save(fig, out, "split")


def f_attn(d, out):
    for h in ["impersonation", "fuzzy", "dos"]:
        z = _npz(d, h)
        if z is not None and "ex_unknown_attention" in z: break
    else: return
    T = len(z["ex_unknown_mask"]); x = np.arange(T); fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.6), sharex=True)
    for ax, k, t in zip(axs, ["attention", "occlusion"], ["head-averaged attention", "occlusion evidence (Mahalanobis drop)"]):
        v = z[f"ex_unknown_{k}"]; ax.bar(x, v, color=np.where(z["ex_unknown_mask"], "tab:red", "tab:gray")); ax.set_ylabel(t, fontsize=5)
    axs[1].set_xlabel("message position in window (red = injected message)"); axs[0].set_title(f"unknown = {h}", fontsize=7); save(fig, out, "attn")


def T(rows, cols, fn, out):
    with open(os.path.join(out, "tables", fn + ".tex"), "w") as f:
        f.write("\\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n")
        f.write("\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote", fn + ".tex")


def tables(R, D, variants, Rr, out):
    nm = dict(normal="Attack-free", dos="DoS", fuzzy="Fuzzy", impersonation="Impersonation")
    if D: T([[nm[c], f"{D[c]['n']:,}", f"{D[c]['expected']:,}", f"{D[c]['n_atk']:,}", f"{100 * D[c]['n_atk'] / max(D[c]['n'], 1):.1f}"] for c in nm],
            ["Stream", "Rows read", "Published", "Attack msgs", "\\%"], "tab_dataset", out)
    if R:
        T([[("none (all known)" if h == "none" else f"hide {h}"), fmt([r["closed"]["acc"] for r in R if r["holdout"] == h]), fmt([r["closed"]["macro_f1"] for r in R if r["holdout"] == h])] for h in HOLD + ["none"]],
          ["Setting", "Accuracy", "Macro-F1"], "tab_closed", out)
        rows = []
        for h in HOLD:
            for m, lab, best in (("auroc", "AUROC", max), ("fpr95", "FPR@95", min), ("open_f1", "Open-F1", max)):
                mu = [np.mean(opens(R, h, s, m)) if opens(R, h, s, m) else np.nan for s in SC]; b = best(np.nan_to_num(mu, nan=-1 if best is max else 9))
                rows.append([f"{h}: {lab}" if m == "auroc" else f"\\quad {lab}"] + [("\\textbf{%s}" if abs(mu[i] - b) < 1e-12 else "%s") % fmt(opens(R, h, s, m)) for i, s in enumerate(SC)])
        T(rows, ["Hidden attack / metric"] + [s.replace("RejectClass", "Reject") for s in SC], "tab_openset", out)
        rows = []
        for n in RANK:
            g = lambda t, k: [r["explain"][t][n][k] for r in R if n in r["explain"].get(t, {}) and k in r["explain"][t][n]]
            rows.append([n.replace("_", " "), fmt(g("class", "del_auc")), fmt(g("class", "ins_auc")), fmt(g("class", "lift"), 2), fmt(g("unknown", "lift"), 2), fmt(g("unknown", "ap"), 2)])
        T(rows, ["Ranking", "Del. AUC", "Ins. AUC", "Lift (class)", "Lift (unk.)", "AP (unk.)"], "tab_localization", out)
    if variants:
        rows = []
        for n, V in variants.items():
            au = [x for h in HOLD for x in opens(V, h, PRIM, "auroc")]; fp = [x for h in HOLD for x in opens(V, h, PRIM, "fpr95")]; fo = [x for h in HOLD for x in opens(V, h, PRIM, "open_f1")]
            ms_ = [x for h in HOLD for x in opens(V, h, "MSP", "auroc")]
            rows.append([n.replace("_", "\\_"), fmt([r["closed"]["macro_f1"] for r in V]), fmt(au), fmt(fp), fmt(fo), fmt(ms_)])
        T(rows, ["Variant", "Closed macro-F1", f"AUROC ({PRIM})", f"FPR@95 ({PRIM})", f"Open-F1 ({PRIM})", "AUROC (MSP)"], "tab_ablation", out)
    if R and Rr:
        T([[lab, fmt([r["closed"]["acc"] for r in X]), fmt([r["closed"]["macro_f1"] for r in X]), fmt([x for h in HOLD for x in opens(X, h, PRIM, "auroc")])] for lab, X in (("temporal", R), ("random", Rr))],
          ["Split", "Accuracy", "Macro-F1", f"AUROC ({PRIM})"], "tab_split", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--main", required=True); p.add_argument("--ablations", nargs="*", default=[]); p.add_argument("--split-random", default=None); p.add_argument("--out", default="overleaf")
    a = p.parse_args(); os.makedirs(os.path.join(a.out, "figures"), exist_ok=True); os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    R = load(a.main); D = json.load(open(os.path.join(a.main, "data_stats.json"))) if os.path.exists(os.path.join(a.main, "data_stats.json")) else None
    variants = {"full": R}; variants.update({os.path.basename(d.rstrip("/")): load(d) for d in a.ablations}); Rr = load(a.split_random) if a.split_random else []
    jobs = [(lambda: f_class_dist(D, a.out), D), (lambda: f_id_dist(D, a.out), D), (lambda: f_train(R, a.out), R), (lambda: f_cm(R, a.out), R), (lambda: f_roc(a.main, a.out), R),
            (lambda: f_scoredist(a.main, a.out), R), (lambda: f_auroc(R, a.out), R), (lambda: f_calib(R, a.out), R), (lambda: f_faith(R, a.out), R), (lambda: f_loc(R, a.out), R),
            (lambda: f_ablation(variants, a.out), len(variants) > 1), (lambda: f_split(R, Rr, a.out), R and Rr), (lambda: f_attn(a.main, a.out), R)]
    for fn, ok in jobs:
        try:
            if ok: fn()
        except Exception as e: print("  skipped a figure:", type(e).__name__, e)
    tables(R, D, variants if len(variants) > 1 else {}, Rr, a.out)
