"""Explanation evaluation against rule-derived MESSAGE-LEVEL ground truth.
Two tasks: (i) why this class (evidence = log-prob of predicted class), (ii) where is the unknown (evidence = Mahalanobis distance)."""
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score

trapz = getattr(np, "trapezoid", None) or np.trapz
FRACS = np.linspace(0, 1, 11)


class TorchMaha:
    def __init__(self, maha, dev):
        self.mu, self.P = torch.as_tensor(maha.mu, device=dev), torch.as_tensor(maha.P, device=dev)
    def d2(self, z):
        d = z[:, None, :] - self.mu[None]; return torch.einsum("nkd,de,nke->nk", d, self.P, d).min(1).values


def targets(model, K, tm):
    cls = lambda X, c: F.log_softmax(model(X)[0][:, :K], 1).gather(1, c[:, None]).squeeze(1)
    unk = lambda X, c: tm.d2(model(X)[1])
    return cls, unk


def rank_scores(model, X, c, target):
    N, T, _ = X.shape
    with torch.no_grad(): att = model(X)[2].mean(1); base = target(X, c)
    x = X.clone().requires_grad_(True); target(x, c).sum().backward(); grad = (x.grad * x).abs().sum(-1).detach()
    occ = []
    with torch.no_grad():
        for t in range(T):
            X2 = X.clone(); X2[:, t] = 0; occ.append(base - target(X2, c))
    return {"attention": att, "gradxinput": grad, "occlusion": torch.stack(occ, 1), "random": torch.rand(N, T, device=X.device),
            "reversed_attention": -att}


def _chunks(model, X, c, target, bs=256):
    acc = {}
    for i in range(0, len(X), bs):
        r = rank_scores(model, X[i:i + bs], c[i:i + bs], target)
        for k, v in r.items(): acc.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def localization(R, M):
    """R: {name: (N,T) scores}; M: (N,T) bool injected-message mask. Reports R-precision, pointing game, AP and lift over chance."""
    k = M.sum(1); sel = (k > 0) & (k < M.shape[1]); out = {}
    if sel.sum() < 10: return out
    m, kk = M[sel], k[sel]; chance = float((kk / M.shape[1]).mean())
    for name, s in R.items():
        s = s[sel]; rank = np.argsort(np.argsort(-s, 1), 1)
        rp = float(((rank < kk[:, None]) & m).sum(1).__truediv__(kk).mean())
        out[name] = dict(r_precision=rp, lift=rp / chance, chance=chance, pointing=float(m[np.arange(len(m)), np.argmax(s, 1)].mean()),
                         ap=float(np.mean([average_precision_score(m[i], s[i]) for i in range(len(m))])), n=int(sel.sum()))
    return out


@torch.no_grad()
def _prob(model, X, c, K, bs=512):
    return torch.cat([F.softmax(model(X[i:i + bs])[0][:, :K], 1).gather(1, c[i:i + bs, None]).squeeze(1) for i in range(0, len(X), bs)])


def del_ins(model, X, c, R, K):
    """Deletion/insertion curves for each ranking (messages replaced by the median message = 0 after normalisation)."""
    N, T, _ = X.shape; res = {}
    for name, s in R.items():
        rank = torch.as_tensor(np.argsort(np.argsort(-s, 1), 1), device=X.device); d, n = [], []; z = torch.zeros_like(X)
        for f in FRACS:
            m = (rank < int(round(f * T))).unsqueeze(-1)
            d.append(float(_prob(model, torch.where(m, z, X), c, K).mean())); n.append(float(_prob(model, torch.where(m, X, z), c, K).mean()))
        res[name] = dict(del_auc=float(trapz(d, FRACS)), ins_auc=float(trapz(n, FRACS)), del_curve=d, ins_curve=n)
    return res


def explain_run(model, K, tm, Xc, Mc, Xu, Mu, dev, n_cls=500, n_unk=300):
    """Xc/Mc: windows+masks of KNOWN attack classes; Xu/Mu: windows+masks of the held-out (unknown) attack. Returns (metrics, example)."""
    model.eval(); cls_t, unk_t = targets(model, K, tm); out, ex = {}, {}
    if len(Xc):
        X = torch.as_tensor(Xc[:n_cls], device=dev)
        with torch.no_grad(): c = model(X)[0][:, :K].argmax(1)
        R = _chunks(model, X, c, cls_t); out["class"] = del_ins(model, X, c, R, K)
        for k, v in localization(R, Mc[:n_cls]).items(): out["class"][k].update(v)
        part = np.flatnonzero((Mc[:n_cls].sum(1) > 0) & (Mc[:n_cls].sum(1) < Mc.shape[1]))
        if len(part):
            i = int(part[0]); ex["class"] = dict(attention=R["attention"][i], occlusion=R["occlusion"][i], mask=Mc[i])
    if tm is not None and len(Xu):
        X = torch.as_tensor(Xu[:n_unk], device=dev); R = _chunks(model, X, torch.zeros(len(X), dtype=torch.long, device=dev), unk_t)
        out["unknown"] = localization(R, Mu[:n_unk])
        part = np.flatnonzero((Mu[:n_unk].sum(1) > 0) & (Mu[:n_unk].sum(1) < Mu.shape[1]))
        if len(part):
            i = int(part[0]); ex["unknown"] = dict(attention=R["attention"][i], occlusion=R["occlusion"][i], mask=Mu[i])
    return out, ex
