"""Open-set scores (higher = more 'known'), robust score fusion and (Mondrian) conformal rejection thresholds."""
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf


class Mahalanobis:
    """Class-conditional Gaussians with a tied Ledoit-Wolf covariance in embedding space (fitted on KNOWN training windows)."""
    def fit(self, emb, y, K):
        miss = [c for c in range(K) if not np.any(y == c)]
        if miss: raise ValueError(f"Mahalanobis fit: no training embeddings for class index(es) {miss}")
        self.mu = np.stack([emb[y == c].mean(0) for c in range(K)]).astype(np.float32)
        self.P = LedoitWolf().fit(emb - self.mu[y]).precision_.astype(np.float32); return self
    def d2(self, emb):
        d = emb[:, None, :] - self.mu[None]; return np.einsum("nkd,de,nke->nk", d, self.P, d).min(1)
    def __call__(self, emb): return -self.d2(emb)


def base_scores(logits, emb, K, maha):
    kn = logits[:, :K]
    out = {"MSP": softmax(kn, 1).max(1), "Energy": logsumexp(kn, 1), "Mahalanobis": maha(emb)}
    if logits.shape[1] > K: out["RejectClass"] = logsumexp(kn, 1) - logits[:, K]        # log-odds known vs pseudo-unknown
    return out


class Fusion:
    """Robust-z-normalise component scores on VALIDATION KNOWN data; fuse by min (conservative) or mean."""
    def __init__(self, comps): self.comps = comps
    def fit(self, val):
        self.med = {c: np.median(val[c]) for c in self.comps}
        self.sc = {c: 1.4826 * np.median(np.abs(val[c] - self.med[c])) + 1e-9 for c in self.comps}; return self
    def __call__(self, sc):
        z = np.stack([(sc[c] - self.med[c]) / self.sc[c] for c in self.comps], 1)
        return {"Fusion-min": z.min(1), "Fusion-mean": z.mean(1)}


def conformal_thresholds(scores, pred, alpha, K, mondrian=True, min_n=50):
    """Reject when score <= threshold[pred]. Split-conformal quantile with finite-sample correction; per-class (Mondrian) optional."""
    def q(s): return np.sort(s)[max(int(np.floor(alpha * (len(s) + 1))) - 1, 0)]
    th = np.full(K, q(scores))
    if mondrian:
        for c in range(K):
            s = scores[pred == c]
            if len(s) >= min_n: th[c] = q(s)
    return th
