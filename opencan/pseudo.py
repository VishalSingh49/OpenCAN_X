"""Leakage-controlled pseudo-unknown synthesis: protocol-level mutation operators applied to NORMAL TRAINING traffic.
Operators tied to a held-out attack family are excluded (operator hold-out), so the detector never sees a proxy of it."""
import numpy as np

ALL_OPS = ["flood", "randinj", "masq", "silence", "replay"]
LINK = {"dos": "flood", "fuzzy": "randinj", "impersonation": "masq"}      # operator <-> attack family it resembles


def allowed_ops(holdout):
    return [o for o in ALL_OPS if holdout in (None, "none", "extra") or o != LINK.get(holdout)]


def _active(t0, t1, rng):
    out, t = [], t0 + rng.uniform(0.05, 0.3)
    while t < t1:
        L = rng.uniform(0.2, 1.0); out.append((t, min(t1, t + L))); t += L + rng.uniform(0.1, 0.5)
    return out


def _times(act, rate):
    if not act: return np.zeros(0)
    return np.concatenate([np.arange(s, e, 1.0 / rate) for s, e in act])


def _inact(t, act):
    m = np.zeros(len(t), bool)
    for s, e in act: m |= (t >= s) & (t < e)
    return m


def _merge(B, N):
    t, i, p, l, m = (np.concatenate([a, b]) for a, b in zip(B, N)); o = np.argsort(t, kind="stable")
    return t[o], i[o], p[o], l[o], m[o]


def _new(t, i, p, k): return (t, np.broadcast_to(i, (k,)).astype(np.int32), p, np.full(k, 8, np.uint8), np.ones(k, bool))


def op_flood(B, rng):            # generic high-rate flooding with ONE random ID (not the DoS ID 0x000)
    t = B[0]; act = _active(t[0], t[-1], rng); tt = _times(act, rng.uniform(500, 3000)); k = len(tt)
    return _merge(B, _new(tt, rng.integers(1, 2048), np.tile(rng.integers(0, 256, 8, dtype=np.uint8), (k, 1)), k))


def op_randinj(B, rng):          # random IDs and payloads at a moderate rate
    t = B[0]; act = _active(t[0], t[-1], rng); tt = _times(act, rng.uniform(100, 800)); k = len(tt)
    return _merge(B, _new(tt, rng.integers(0, 2048, k), rng.integers(0, 256, (k, 8), dtype=np.uint8), k))


def op_masq(B, rng):             # silence a legitimate ECU and inject its ID faster with perturbed payloads
    t, i, p, l, m = B; ids, c = np.unique(i, return_counts=True); cand = ids[c >= 100]
    cid = rng.choice(cand); mt = t[i == cid]; per = float(np.median(np.diff(mt)))
    act = _active(t[0], t[-1], rng); rem = _inact(t, act) & (i == cid); keep = ~rem
    B = (t[keep], i[keep], p[keep], l[keep], m[keep]); tt = _times(act, rng.uniform(1.5, 3.0) / per); k = len(tt)
    src = p[i == cid]; pay = src[rng.integers(0, len(src), k)].copy()
    pos = rng.integers(0, 8, (k, 2)); pay[np.arange(k)[:, None], pos] ^= rng.integers(1, 256, (k, 2)).astype(np.uint8)
    return _merge(B, _new(tt, cid, pay, k))


def op_silence(B, rng):          # suspend 1-3 ECUs; the first message after each gap is the anomalous one
    t, i, p, l, m = B; ids, c = np.unique(i, return_counts=True); sel = rng.choice(ids[c >= 100], size=min(3, int((c >= 100).sum())), replace=False)
    act = _active(t[0], t[-1], rng); keep = ~(_inact(t, act) & np.isin(i, sel))
    t, i, p, l, m = t[keep], i[keep], p[keep], l[keep], m[keep].copy()
    for cid in sel:
        idx = np.flatnonzero(i == cid)
        for s, e in act:
            q = np.searchsorted(t[idx], e)
            if q < len(idx): m[idx[q]] = True
    return t, i, p, l, m


def op_replay(B, rng):           # replay an earlier block of traffic at double speed inside active intervals
    t, i, p, l, m = B; n = len(t) // 5; act = _active(t[n], t[-1], rng); parts = []
    for s, e in act:
        nt = s + (t[:n] - t[0]) * 0.5; ok = nt < e
        parts.append((nt[ok], i[:n][ok], p[:n][ok], l[:n][ok], np.ones(int(ok.sum()), bool)))
    if not parts: return B
    return _merge(B, tuple(np.concatenate(z) for z in zip(*parts)))


OPS = dict(flood=op_flood, randinj=op_randinj, masq=op_masq, silence=op_silence, replay=op_replay)


def synthesize(raw, ops, n_msgs, rng):
    """raw: dict(ts, ids, pay, length) of the normal TRAINING segment. Returns [(op, (ts, ids, pay, length, mask))]."""
    n = len(raw["ts"]); out = []
    for op in ops:
        lo = int(rng.integers(0, max(1, n - n_msgs))); sl = slice(lo, min(n, lo + n_msgs))
        B = (raw["ts"][sl].copy(), raw["ids"][sl].copy(), raw["pay"][sl].copy(), raw["length"][sl].copy(), np.zeros(sl.stop - sl.start, bool))
        out.append((op, OPS[op](B, rng)))
    return out
