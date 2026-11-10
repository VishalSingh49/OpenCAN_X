"""Writes SMALL mock files in the real dataset1..4.csv layout (hex strings) for smoke-testing the code. Not real data."""
import os, argparse
import numpy as np, pandas as pd

def _veh(rng, n=24):
    ids = rng.choice(np.arange(0x100, 0x7F0), n, replace=False); ids[0] = 0x164
    return ids, rng.choice([0.002, 0.005, 0.01, 0.02], n), rng.integers(0, 256, (n, 8)), rng.choice([0, 1, 2], (n, 8), p=[.4, .2, .4])

def _normal(veh, n, rng):
    ids, per, st, kd = veh; T = n / np.sum(1 / per); parts = []
    for j in range(len(ids)):
        k = int(T / per[j]) + 2; t = np.cumsum(per[j] * (1 + rng.normal(0, .02, k))) + rng.uniform(0, per[j])
        p = np.zeros((k, 8))
        for b in range(8):
            p[:, b] = st[j, b] if kd[j, b] == 0 else ((np.arange(k) + st[j, b]) % 256 if kd[j, b] == 1 else np.clip(st[j, b] + np.cumsum(rng.normal(0, 2, k)), 0, 255))
        parts.append(pd.DataFrame(np.column_stack([t, np.full(k, ids[j]), p]), columns=["ts", "id"] + [f"b{i}" for i in range(8)]))
    return pd.concat(parts).sort_values("ts").reset_index(drop=True).iloc[:n]

def make(out, n=40000, seed=0):
    rng = np.random.default_rng(seed); veh = _veh(rng); os.makedirs(out, exist_ok=True)
    for ci, (cls, fn) in enumerate(zip(["normal", "dos", "fuzzy", "impersonation"], ["dataset1", "dataset2", "dataset3", "dataset4"])):
        df = _normal(veh, n, rng); T = df.ts.iloc[-1]
        if cls != "normal":
            act, t = [], 0.1
            while t < T: L = rng.uniform(0.2, 0.6); act.append((t, min(T, t + L))); t += L + rng.uniform(0.1, 0.4)
            if cls == "impersonation":
                for s, e in act: df = df[~((df.id == 0x164) & (df.ts >= s) & (df.ts < e))]
            rate = dict(dos=2000, fuzzy=500, impersonation=1 / (veh[1][0] / 2))[cls]
            tt = np.sort(np.concatenate([np.arange(s, e, 1 / rate) for s, e in act])); k = len(tt)
            nid = dict(dos=np.zeros(k), fuzzy=rng.choice(veh[0], k), impersonation=np.full(k, 0x164))[cls]
            pay = np.clip(rng.normal(220, 10, (k, 8)), 0, 255) if cls == "impersonation" else (np.zeros((k, 8)) if cls == "dos" else rng.integers(0, 256, (k, 8)))
            df = pd.concat([df, pd.DataFrame(np.column_stack([tt, nid, pay]), columns=df.columns)]).sort_values("ts")
        w = 6 if ci < 2 else 8                      # header has 6 payload columns; some files carry 8 (ragged rows)
        rows = [f"{r.ts:.6f},{int(r.id):04x},000,8," + ",".join(f"{int(r[f'b{i}']):02x}" for i in range(w)) for _, r in df.iterrows()] if len(df) < 5000 else None
        if rows is None:
            ids = df.id.astype(int).map("{:04x}".format).to_numpy(); ts = df.ts.map("{:.6f}".format).to_numpy()
            bs = [df[f"b{i}"].astype(int).map("{:02x}".format).to_numpy() for i in range(w)]
            rows = [",".join(x) for x in zip(ts, ids, ["000"] * len(df), ["8"] * len(df), *bs)]
        with open(os.path.join(out, fn + ".csv"), "w") as f:
            f.write("TS,ID1,ID0,LEN," + ",".join(f"DLC{i}" for i in range(6)) + "\n"); f.write("\n".join(rows) + "\n")

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--out", default="mock_data"); a.add_argument("--n", type=int, default=40000); a = a.parse_args(); make(a.out, a.n)
