"""Leave-one-attack-out open-set evaluation of OpenCAN-X on CAN_HCRL_OTIDS (dataset1..4.csv).
Example:  python run_experiments.py --root /path/to/csvs --out results/full --seeds 3"""
import argparse, copy, json, math, os, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score
from opencan.data import prepare, CLASSES
from opencan.features import featurize
from opencan.pseudo import allowed_ops, synthesize
from opencan.model import AIMHCNN
from opencan.openset import Mahalanobis, base_scores, Fusion, conformal_thresholds
from opencan.metrics import closed_set, open_set
from opencan.explain import TorchMaha, explain_run

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2]
PRIMARY = "Fusion-min"


def gather(banks, bank, starts, ar):
    X = np.empty((len(starts), len(ar), banks[0][0].shape[1]), np.float32)
    for b, (f, _) in enumerate(banks):
        s = bank == b
        if s.any(): X[s] = f[starts[s][:, None] + ar]
    return X


@torch.no_grad()
def infer(model, banks, bank, starts, ar, dev, bs=2048):
    model.eval(); L, Z = [], []
    for i in range(0, len(starts), bs):
        lg, z, _ = model(torch.as_tensor(gather(banks, bank[i:i + bs], starts[i:i + bs], ar), device=dev))
        L.append(lg.float().cpu()); Z.append(z.float().cpu())
    return torch.cat(L).numpy(), torch.cat(Z).numpy()


def build_pseudo(ds, ops, a, rng):
    """Pseudo-unknown windows from mutated NORMAL TRAINING traffic (operators tied to the held-out attack are excluded)."""
    n_tr = int(0.7 * ds.meta["normal"]["n"]); raw = {k: ds.raw["normal"][k][:n_tr] for k in ("ts", "ids", "pay", "length")}
    fs, ms, ss, off = [], [], [], 0
    for op, (t, i, p, l, m) in synthesize(raw, ops, a.pseudo_msgs, rng):
        Fm = ds.norm(featurize(t, i, p, l, ds.stats, ds.args["feat_mode"]))
        cs = np.concatenate([[0], np.cumsum(m)]); s = np.arange(0, len(Fm) - ds.W + 1, ds.W // 2)
        keep = (cs[s + ds.W] - cs[s]) >= ds.args["min_atk"]
        fs.append(Fm); ms.append(m); ss.append(off + s[keep]); off += len(Fm)
    return np.concatenate(fs), np.concatenate(ms), np.concatenate(ss)


def train(model, banks, tr, va, K, a, dev, rng, ar):
    ts_, tb, ty = tr; vs, vb, vy = va
    w = (1.0 / np.maximum(np.bincount(ty), 1)) ** 0.5; p = w[ty]; p /= p.sum()          # sqrt-inverse-frequency balanced sampling
    spe = math.ceil(a.epoch_size / a.bs)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe)
    hist, best, state, bad = [], -1.0, None, 0
    for ep in range(a.epochs):
        model.train(); idx = rng.choice(len(ty), a.epoch_size, p=p); tot = 0.0
        for i in range(0, len(idx), a.bs):
            b = idx[i:i + a.bs]; x = torch.as_tensor(gather(banks, tb[b], ts_[b], ar), device=dev); y = torch.as_tensor(ty[b], device=dev)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x)[0], y); loss.backward(); opt.step(); sch.step(); tot += float(loss.detach()) * len(b)
        lg, _ = infer(model, banks, vb, vs, ar, dev)
        f1 = float(f1_score(vy, lg[:, :K].argmax(1), average="macro", labels=list(range(K)), zero_division=0))
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(idx), val_macro_f1=f1))
        print(f"    ep{ep + 1:02d} loss={hist[-1]['train_loss']:.4f} val_macroF1={f1:.4f}", flush=True)
        if f1 > best: best, state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(state); return model.eval(), hist


def run(a):
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu")); os.makedirs(os.path.join(a.out, "scores"), exist_ok=True)
    ds = prepare(a.root, a.window, a.hex, a.max_rows, a.ts_scale, a.features, a.legit_min, a.min_atk, a.label_mode, file_classes=a.file_classes, auto_fallback=not a.no_auto_fallback)
    json.dump(ds.report, open(os.path.join(a.out, "data_stats.json"), "w"), indent=1)
    ar, recs = ds.ar, []; banks0 = [(ds.feats, ds.mask)]
    for seed in range(a.seeds):
        sp = ds.make_splits(a.split, seed, train_stride=a.train_stride)
        print(f"seed {seed}: windows " + ", ".join(f"{k}={len(v[1]):,}" for k, v in sp.items()), flush=True)
        for hold in a.holdouts + ["none"]:
            t0 = time.time(); rng = np.random.default_rng(1000 * seed + 7); torch.manual_seed(seed)
            known = [c for c in CLASSES if c != hold]; K = len(known); lut = np.full(len(CLASSES), -1)
            for i, c in enumerate(known): lut[CLASSES.index(c)] = i
            def sel(s): y = lut[sp[s][1]]; m = y >= 0; return sp[s][0][m], y[m]
            trs, ytr = sel("train"); vas, yva = sel("val"); tes, yte = sel("test")
            cnt = {s_: np.bincount(y_, minlength=K) for s_, y_ in (("train", ytr), ("val", yva), ("test", yte))}
            print("  windows per class (train/val/test): " + ", ".join(f"{known[k]}={cnt['train'][k]:,}/{cnt['val'][k]:,}/{cnt['test'][k]:,}" for k in range(K)), flush=True)
            bad = [known[k] for k in range(K) if cnt["train"][k] < 50 or cnt["val"][k] < 10 or cnt["test"][k] < 10]
            if bad: raise SystemExit(f"\nERROR: class(es) {bad} have (almost) no windows, so the labelling rules found no attack messages for them.\n"
                                     "Run `python inspect_data.py --root <data>` and check the rule-label counts. If the rules do not fit a class, use file-level labels for it, "
                                     "e.g. `--file-classes fuzzy impersonation`.")
            banks, bank_tr, st_tr, y_tr, n_pw, ops = banks0, np.zeros(len(trs), int), trs, ytr, 0, []
            if a.pseudo == "ops":
                ops = allowed_ops(hold); pf, pm, ps = build_pseudo(ds, ops, a, rng); n_pw = len(ps)
                if n_pw: banks = banks0 + [(pf, pm)]; bank_tr = np.r_[bank_tr, np.ones(n_pw, int)]; st_tr = np.r_[trs, ps]; y_tr = np.r_[ytr, np.full(n_pw, K)]
            n_out = K + 1 if n_pw else K
            print(f"  hold-out={hold}: known={known} ops={ops} pseudo_windows={n_pw:,}", flush=True)
            model = AIMHCNN(n_out, ds.feats.shape[-1], drop=a.dropout).to(dev)
            model, hist = train(model, banks, (st_tr, bank_tr, y_tr), (vas, np.zeros(len(vas), int), yva), K, a, dev, rng, ar)
            fit = rng.choice(len(trs), min(a.fit_max, len(trs)), replace=False)
            for k_ in range(K): fit = np.union1d(fit, np.flatnonzero(ytr == k_)[:2000])
            _, ztr = infer(model, banks0, np.zeros(len(fit), int), trs[fit], ar, dev); maha = Mahalanobis().fit(ztr, ytr[fit], K)
            z0 = lambda s: np.zeros(len(s), int)
            lva, zva = infer(model, banks0, z0(vas), vas, ar, dev); lte, zte = infer(model, banks0, z0(tes), tes, ar, dev)
            S_v, S_k = base_scores(lva, zva, K, maha), base_scores(lte, zte, K, maha)
            fu = Fusion([c for c in ("Energy", "Mahalanobis", "RejectClass") if c in S_v]).fit(S_v); S_v.update(fu(S_v)); S_k.update(fu(S_k))
            pv, pk = lva[:, :K].argmax(1), lte[:, :K].argmax(1)
            rec = dict(seed=seed, holdout=hold, known=known, ops=ops, n_pseudo=n_pw, history=hist, closed=closed_set(pk, yte, K), open={}, conformal={})
            ex_save, npz = {}, dict(yk=yte, pk=pk)
            ku = np.flatnonzero(sp["test"][1] == CLASSES.index(hold)) if hold in CLASSES else np.zeros(0, int)
            if hold != "none" and len(ku):
                us = sp["test"][0][ku]; lu, zu = infer(model, banks0, z0(us), us, ar, dev); S_u = base_scores(lu, zu, K, maha); S_u.update(fu(S_u)); pu = lu[:, :K].argmax(1)
                rec["n_unknown"] = int(len(us)); rec["unk_assignment"] = np.bincount(pu, minlength=K).tolist(); npz["pu"] = pu
                for sn in S_k:
                    g, m = conformal_thresholds(S_v[sn], pv, 0.05, K, False), conformal_thresholds(S_v[sn], pv, 0.05, K, True)
                    rec["open"][sn] = open_set(S_k[sn], S_u[sn], pk, yte, pu, g, K); rec["open"][sn]["mondrian"] = open_set(S_k[sn], S_u[sn], pk, yte, pu, m, K)
                    npz[f"sk_{sn}"], npz[f"su_{sn}"] = S_k[sn].astype(np.float32), S_u[sn].astype(np.float32)
                for al in ALPHAS:
                    rec["conformal"][str(al)] = {}
                    for kind, md in (("global", False), ("mondrian", True)):
                        th = conformal_thresholds(S_v[PRIMARY], pv, al, K, md)
                        rec["conformal"][str(al)][kind] = dict(known_reject=float((S_k[PRIMARY] <= th[pk]).mean()), unk_recall=float((S_u[PRIMARY] <= th[pu]).mean()))
                rs = np.random.default_rng(seed); pick = rs.choice(len(us), min(300, len(us)), replace=False); Xu, Mu = ds.gather(us[pick]), ds.gather_mask(us[pick])
            else: Xu, Mu = np.zeros((0, ds.W, ds.feats.shape[-1]), np.float32), np.zeros((0, ds.W), bool)
            ai = np.flatnonzero(yte > 0); ai = np.random.default_rng(seed).choice(ai, min(600, len(ai)), replace=False) if len(ai) else ai
            rec["explain"], ex_save = explain_run(model, K, TorchMaha(maha, dev), ds.gather(tes[ai]), ds.gather_mask(tes[ai]), Xu, Mu, dev)
            for t, d in ex_save.items():
                for k, v in d.items(): npz[f"ex_{t}_{k}"] = np.asarray(v)
            np.savez_compressed(os.path.join(a.out, "scores", f"{hold}_s{seed}.npz"), **npz)
            rec["time_s"] = time.time() - t0; recs.append(rec); json.dump(recs, open(os.path.join(a.out, "results.json"), "w"), indent=1)
            o = rec["open"].get(PRIMARY, {}); print(f"  -> acc={rec['closed']['acc']:.4f} macroF1={rec['closed']['macro_f1']:.4f}" + (f" | {PRIMARY} AUROC={o['auroc']:.3f} MSP AUROC={rec['open']['MSP']['auroc']:.3f}" if o else "") + f" ({rec['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder with dataset1..4.csv"); p.add_argument("--out", default="results/full")
    p.add_argument("--window", type=int, default=32); p.add_argument("--split", choices=["temporal", "random"], default="temporal")
    p.add_argument("--features", choices=["can", "raw"], default="can"); p.add_argument("--pseudo", choices=["ops", "none"], default="ops")
    p.add_argument("--holdouts", nargs="+", default=["dos", "fuzzy", "impersonation"]); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--epoch-size", dest="epoch_size", type=int, default=60000)
    p.add_argument("--bs", type=int, default=256); p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--patience", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3); p.add_argument("--train-stride", dest="train_stride", type=int, default=16)
    p.add_argument("--min-atk", dest="min_atk", type=int, default=3); p.add_argument("--legit-min", dest="legit_min", type=int, default=50)
    p.add_argument("--label-mode", dest="label_mode", choices=["rule", "file"], default="rule")
    p.add_argument("--file-classes", dest="file_classes", nargs="*", default=[], choices=["dos", "fuzzy", "impersonation"], help="use file-level labels for these classes")
    p.add_argument("--no-auto-fallback", dest="no_auto_fallback", action="store_true", help="do not switch to file-level labels when the rules flag <2%% of a file")
    p.add_argument("--hex", choices=["auto", "yes", "no"], default="auto"); p.add_argument("--ts-scale", dest="ts_scale", type=float, default=None)
    p.add_argument("--max-rows", dest="max_rows", type=int, default=None, help="debug: rows per file")
    p.add_argument("--pseudo-msgs", dest="pseudo_msgs", type=int, default=40000); p.add_argument("--fit-max", dest="fit_max", type=int, default=60000)
    p.add_argument("--device", default=None); run(p.parse_args())
