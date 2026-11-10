"""Loading of the CAN_HCRL_OTIDS per-file CSVs (dataset1..4.csv), rule-based message labelling,
leakage-aware windowing/splitting, and robust normalisation."""
import os, re
import numpy as np, pandas as pd
from .features import CANStats, featurize, timing_residual, NID

CLASSES = ["normal", "dos", "fuzzy", "impersonation"]
FILES = dict(normal="dataset1.csv", dos="dataset2.csv", fuzzy="dataset3.csv", impersonation="dataset4.csv")
KEYS = dict(normal=("attack_free", "free", "dataset1"), dos=("dos", "dataset2"), fuzzy=("fuzz", "dataset3"),
            impersonation=("imperson", "spoof", "dataset4"))
EXPECTED = dict(normal=2_369_868, dos=656_579, fuzzy=591_990, impersonation=995_472)
IMP_ID = 0x164


def _hex(v):
    try: return int(v, 16)
    except ValueError: return 0


def _decode(col, hexa):
    col = col.fillna("0").astype(str).str.strip().replace("", "0")
    if not hexa: return pd.to_numeric(col, errors="coerce").fillna(0).to_numpy(np.int64)
    mp = {v: _hex(v) for v in col.unique()}
    return col.map(mp).to_numpy(np.int64)


def _looks_hex(col): return bool(col.dropna().astype(str).str.contains(r"[a-fA-F]").any())


def read_stream(path, hex_mode="auto", max_rows=None, chunksize=1_000_000):
    """Reads one CAN CSV. Handles the header layouts  TS,ID1,ID0,LEN,DLC0..  and  TS,ID1,DLC0..DLC7  (ragged rows allowed)."""
    with open(path) as f: head = f.readline().strip().split(",")
    try: float(head[0]); has_h = False
    except ValueError: has_h = True
    up = [h.strip().upper() for h in head] if has_h else []
    id_i = up.index("ID1") if "ID1" in up else 1
    len_i = up.index("LEN") if "LEN" in up else None
    idc = [i for i, n in enumerate(up) if re.fullmatch(r"ID\d*", n)]
    pay0 = (len_i + 1) if len_i is not None else ((max(idc) + 1) if idc else 4)
    rd = pd.read_csv(path, header=None, skiprows=1 if has_h else 0, names=list(range(pay0 + 10)), dtype=str,
                     index_col=False, chunksize=chunksize, nrows=max_rows, on_bad_lines="skip")
    out, hx_id, hx_pay = [], None, None
    for ch in rd:
        ts = pd.to_numeric(ch[0], errors="coerce"); ch = ch[ts.notna().to_numpy()]; ts = ts[ts.notna()].to_numpy(float)
        if hx_id is None:
            hx_id = hex_mode == "yes" or (hex_mode == "auto" and _looks_hex(ch[id_i]))
            hx_pay = hex_mode == "yes" or (hex_mode == "auto" and _looks_hex(pd.concat([ch[pay0 + j] for j in range(8)])))
        ids = np.clip(_decode(ch[id_i], hx_id), 0, NID - 1)
        pay = np.clip(np.stack([_decode(ch[pay0 + j], hx_pay) for j in range(8)], 1), 0, 255).astype(np.uint8)
        ln = np.clip(pd.to_numeric(ch[len_i], errors="coerce").fillna(8).to_numpy(), 0, 8).astype(np.uint8) if len_i is not None \
            else np.full(len(ts), 8, np.uint8)
        out.append((ts, ids.astype(np.int32), pay, ln))
    ts, ids, pay, ln = (np.concatenate(z) for z in zip(*out))
    if np.any(np.diff(ts) < 0):
        o = np.argsort(ts, kind="stable"); ts, ids, pay, ln = ts[o], ids[o], pay[o], ln[o]
    return dict(ts=ts, ids=ids, pay=pay, length=ln, hex_id=bool(hx_id), hex_pay=bool(hx_pay))


def _find(root, cls):
    names = {f.lower(): f for f in os.listdir(root)}
    if FILES[cls].lower() in names: return os.path.join(root, names[FILES[cls].lower()])
    for low, f in sorted(names.items()):
        if low.endswith(".csv") and any(k in low for k in KEYS[cls]): return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV for class '{cls}' in {root} (expected {FILES[cls]})")


def load_streams(root, hex_mode="auto", max_rows=None):
    return {c: read_stream(_find(root, c), hex_mode, max_rows) for c in CLASSES}


def auto_ts_scale(ts):
    d = np.diff(ts); d = d[d > 0]; med = float(np.median(d)) if len(d) else 1e-3
    for s in (1.0, 1e-3, 1e-6, 1e-9):
        if 1e-5 <= med * s <= 5e-2: return s
    return 1.0


def _q(u, fpr):
    """Smallest q in 1..8 with P(unseen >= q) <= fpr on attack-free validation traffic (9 = never fires)."""
    for q in range(1, 9):
        if (u >= q).mean() <= fpr: return q
    return 9


def calibrate_labels(S, st, fpr_fuzzy=0.005, fpr_imp=0.01):
    """Thresholds for the anomaly cues, calibrated on attack-free VALIDATION traffic (70-85% of the normal stream) so that
    the false-positive rate of the labelling rules on benign messages is controlled."""
    n = S["normal"]; N = len(n["ts"]); sl = slice(int(0.7 * N), int(0.85 * N))
    ids, un = n["ids"][sl], st.unseen(n["ids"][sl], n["pay"][sl])
    res = np.abs(timing_residual(n["ts"], n["ids"], st)[0][sl]); mi = ids == IMP_ID
    return dict(fuzzy_q=_q(un[st.legit[ids]], fpr_fuzzy), imp_q=_q(un[mi], fpr_imp) if mi.sum() > 100 else 9,
                imp_r=float(np.quantile(res[mi], 1 - fpr_imp)) if mi.sum() > 100 else float("inf"), fpr_fuzzy=fpr_fuzzy, fpr_imp=fpr_imp)


def rule_labels(cls, ids, pay, res, st, thr, file_classes=()):
    """Message-level ground truth from the dataset description, confirmed by anomaly cues where the ID alone is not decisive.
    DoS: ID 0x000.  Fuzzy: unseen ID OR payload with unseen byte values for that ID (random data on a legitimate ID).
    Impersonation: ID 0x164 AND (payload novelty OR off-period timing).  Normal: none.  Classes in `file_classes` use file-level labels."""
    if cls == "normal": return np.zeros(len(ids), bool)
    if cls in file_classes: return np.ones(len(ids), bool)
    if cls == "dos": return ids == 0
    un = st.unseen(ids, pay)
    if cls == "fuzzy": return (~st.legit[ids]) | (un >= thr["fuzzy_q"])
    return (ids == IMP_ID) & ((un >= thr["imp_q"]) | (np.abs(res) > thr["imp_r"]))


class RobustTanh:
    """Outlier-robust tanh squashing (median/MAD), inspired by the Tanh-based normalisation of Gupta et al.; train-only fit."""
    def fit(self, X):
        med = np.median(X, 0); mad = 1.4826 * np.median(np.abs(X - med), 0)
        self.med = med.astype(np.float32)
        self.s = (3 * np.maximum.reduce([mad, 0.1 * X.std(0), np.full(X.shape[1], 1e-6)])).astype(np.float32)
        return self
    def apply_(self, X, chunk=500_000):
        for i in range(0, len(X), chunk): X[i:i + chunk] = np.tanh((X[i:i + chunk] - self.med) / self.s)
        return X
    def __call__(self, X): return np.tanh((X - self.med) / self.s).astype(np.float32)


class Dataset:
    """Message-level feature array + lazily gathered windows (no window copies are stored)."""
    def __init__(self, S, st, norm, feats, mask, meta, W, args, report):
        self.raw, self.stats, self.norm, self.feats, self.mask = S, st, norm, feats, mask
        self.meta, self.W, self.args, self.report, self.names = meta, W, args, report, CLASSES
        self.csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        self.ar = np.arange(W)

    def _label(self, starts, ci):
        cnt = self.csum[starts + self.W] - self.csum[starts]
        return np.where(cnt >= self.args["min_atk"], ci, np.where(cnt == 0, 0, -1)).astype(np.int8)

    def make_splits(self, mode="temporal", seed=0, ratios=(0.7, 0.15), train_stride=None):
        W = self.W; ts_ = train_stride or W // 2; rng = np.random.default_rng(seed)
        sp = {k: ([], []) for k in ("train", "val", "test")}
        for ci, c in enumerate(CLASSES):
            o, n = self.meta[c]["offset"], self.meta[c]["n"]; a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
            def win(lo, hi, stride): return lo + stride * np.arange(max((hi - lo - W) // stride + 1, 0)) if hi - lo >= W else np.zeros(0, np.int64)
            if mode == "temporal":      # contiguous blocks separated by a W-message guard gap
                parts = {"train": win(o, o + a, ts_), "val": win(o + a + W, o + b, W), "test": win(o + b + W, o + n, W)}
            else:                       # paper-style random split of non-overlapping windows
                s = win(o, o + n, W); s = s[rng.permutation(len(s))]; i, j = int(ratios[0] * len(s)), int((ratios[0] + ratios[1]) * len(s))
                parts = {"train": s[:i], "val": s[i:j], "test": s[j:]}
            for k, s in parts.items():
                sp[k][0].append(s); sp[k][1].append(self._label(s, ci))
        out = {}
        for k, (ss, yy) in sp.items():
            s, y = np.concatenate(ss), np.concatenate(yy); keep = y >= 0; out[k] = (s[keep], y[keep].astype(np.int64))
        return out

    def gather(self, starts): return self.feats[starts[:, None] + self.ar]
    def gather_mask(self, starts): return self.mask[starts[:, None] + self.ar]


def prepare(root, W=32, hex_mode="auto", max_rows=None, ts_scale=None, feat_mode="can", legit_min=50, min_atk=3,
            label_mode="rule", train_frac=0.7, verbose=True, file_classes=(), auto_fallback=True, min_rule_frac=0.10):
    S = load_streams(root, hex_mode, max_rows)
    scale = ts_scale or auto_ts_scale(S["normal"]["ts"])
    for s in S.values(): s["ts"] = s["ts"] * scale
    ntr = int(train_frac * len(S["normal"]["ts"]))
    st = CANStats().fit(S["normal"]["ts"][:ntr], S["normal"]["ids"][:ntr], S["normal"]["pay"][:ntr], legit_min)
    fc = set(CLASSES[1:]) if label_mode == "file" else set(file_classes); thr = calibrate_labels(S, st)
    fl, ml, meta, report, off = [], [], {}, {}, 0
    for c in CLASSES:
        s = S[c]; res = timing_residual(s["ts"], s["ids"], st)[0]; atk = rule_labels(c, s["ids"], s["pay"], res, st, thr, fc)
        rule_n = int(atk.sum()); imp_rate = float(atk[s["ids"] == IMP_ID].mean()) if c == "impersonation" and (s["ids"] == IMP_ID).any() else None
        nw = len(atk) // W; wf = float((atk[:nw * W].reshape(nw, W).sum(1) >= min_atk).mean()) if nw else 0.0
        if auto_fallback and c in ("fuzzy", "impersonation") and c not in fc and wf < min_rule_frac:
            print(f"  NOTE: the rules flag {rule_n:,} messages in the {c} file, which labels only {wf:.1%} of its windows as attack -> using FILE-LEVEL "
                  f"labels for this class (attack messages are not distinguishable per message). Disable with --no-auto-fallback.", flush=True)
            atk = np.ones(len(atk), bool); fc.add(c)
        fl.append(featurize(s["ts"], s["ids"], s["pay"], s["length"], st, feat_mode)); ml.append(atk)
        cnt = np.bincount(s["ids"], minlength=NID); top = np.argsort(-cnt)[:15]
        report[c] = dict(n=len(atk), n_atk=int(atk.sum()), rule_n_atk=rule_n, expected=EXPECTED[c], hex_id=s["hex_id"], hex_pay=s["hex_pay"],
                         top_ids=[[int(i), int(cnt[i])] for i in top if cnt[i] > 0],
                         id_hist=np.histogram(s["ids"], bins=64, range=(0, NID))[0].tolist(),
                         id0=int(cnt[0]), id164_hex=int(cnt[IMP_ID]), id164_dec=int(cnt[164]), file_level=(c in fc),
                         imp_flag_rate=imp_rate)
        meta[c] = dict(offset=off, n=len(atk)); off += len(atk)
    feats, mask = np.concatenate(fl), np.concatenate(ml); del fl
    sub = [feats[m["offset"]: m["offset"] + int(train_frac * m["n"])][:: max(1, int(train_frac * m["n"]) // 200_000)] for m in meta.values()]
    norm = RobustTanh().fit(np.concatenate(sub)); norm.apply_(feats)
    report["_global"] = dict(ts_scale=scale, label_mode=label_mode, file_classes=sorted(fc), label_thresholds=thr, feat_mode=feat_mode, n_features=int(feats.shape[1]), legit_ids=int(st.legit.sum()))
    if verbose:
        print(f"ts scale={scale:g}  hex(id/payload)={S['normal']['hex_id']}/{S['normal']['hex_pay']}  legit IDs={int(st.legit.sum())}")
        for c in CLASSES: print(f"  {c:14s} n={report[c]['n']:>9,} (expected {EXPECTED[c]:>9,})  rule-attack msgs={report[c]['n_atk']:>9,}")
    args = dict(min_atk=min_atk, feat_mode=feat_mode)
    return Dataset(S, st, norm, feats, mask, meta, W, args, report)
