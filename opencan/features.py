"""CAN-aware message features. All reference statistics are fitted on the NORMAL TRAINING segment only."""
import numpy as np, pandas as pd

NID = 2048                      # 11-bit arbitration IDs
EPS = 1e-6
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
RAW_DIM, CAN_DIM = 11, 18
FEATURE_NAMES = ["log_dt", "log_dt_same_id", "period_residual", "id_novel", "id_norm", "id_prior", "len"] + \
                [f"b{i}" for i in range(8)] + ["hamming_drift", "byte_drift", "byte_uniqueness"]


def prev_same_id(ids):
    """Index of the previous message with the same ID (-1 if none). Requires time-sorted input."""
    order = np.argsort(ids, kind="stable"); sid = ids[order]
    same = sid[1:] == sid[:-1]
    prev = np.full(len(ids), -1, np.int64)
    prev[order[1:][same]] = order[:-1][same]
    return prev


class CANStats:
    """Legitimate-ID set, per-ID period and ID prior learned from attack-free training traffic."""
    def fit(self, ts, ids, pay, min_count=50):
        cnt = np.bincount(ids, minlength=NID).astype(np.float64)
        self.seen = np.zeros((NID, 8, 256), bool)                       # (ID, byte position, value) seen in attack-free training
        self.seen[ids[:, None], np.arange(8)[None], pay] = True
        self.legit = cnt >= min_count
        self.prior = (np.log1p(cnt) / np.log1p(cnt.max())).astype(np.float32)
        prev = prev_same_id(ids); has = prev >= 0
        d = pd.DataFrame({"id": ids[has], "dt": ts[has] - ts[prev[has]]})
        g = d.groupby("id").dt; med, n = g.median(), g.size()
        self.period = np.full(NID, np.nan)
        keep = n >= 5
        self.period[med.index[keep].to_numpy()] = med[keep].to_numpy()
        return self


def _unseen(self, ids, pay):
    """Number of payload bytes (0..8) whose value was never observed for that (ID, position) in attack-free training traffic."""
    return (~self.seen[ids[:, None], np.arange(8)[None], pay]).sum(1)
CANStats.unseen = _unseen


def timing_residual(ts, ids, st):
    """Log ratio of the same-ID inter-arrival time to the ID's normal period (0 for unseen IDs / first occurrences)."""
    prev = prev_same_id(ids); has = prev >= 0
    dts = np.zeros(len(ts)); dts[has] = ts[has] - ts[prev[has]]
    per = st.period[ids]; ok = has & ~np.isnan(per)
    res = np.zeros(len(ts)); res[ok] = np.log((dts[ok] + EPS) / (per[ok] + EPS))
    return np.clip(res, -8, 8), dts, prev, has


def featurize(ts, ids, pay, length, st, mode="can"):
    """ts: seconds (sorted), ids: int, pay: (N,8) uint8, length: (N,) -> float32 (N, F)."""
    N = len(ts)
    dt = np.clip(np.diff(ts, prepend=ts[0]), 0, None)
    log_dt = np.log1p(np.clip(dt * 1e3, 0, 1e4))
    base = [log_dt, ids / (NID - 1.0), length / 8.0]
    if mode == "raw":
        return np.column_stack(base + [pay / 255.0]).astype(np.float32)
    res, dts, prev, has = timing_residual(ts, ids, st)
    ham = np.zeros(N); ham[has] = POP[pay[has] ^ pay[prev[has]]].sum(1) / 64.0
    drift = np.zeros(N); drift[has] = np.abs(pay[has].astype(np.int16) - pay[prev[has]]).mean(1) / 255.0
    uniq = ((np.diff(np.sort(pay, 1), axis=1) != 0).sum(1) + 1) / 8.0
    cols = [log_dt, np.log1p(np.clip(dts * 1e3, 0, 1e4)), res, (~st.legit[ids]).astype(float), ids / (NID - 1.0),
            st.prior[ids], length / 8.0] + [pay[:, j] / 255.0 for j in range(8)] + [ham, drift, uniq]
    return np.column_stack(cols).astype(np.float32)
