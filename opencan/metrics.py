import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, accuracy_score, confusion_matrix


def closed_set(pred, y, K):
    return dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
                per_class_f1=f1_score(y, pred, average=None, labels=list(range(K)), zero_division=0).tolist(),
                confusion=confusion_matrix(y, pred, labels=list(range(K))).tolist())


def open_set(sk, su, pk, yk, pu, th, K):
    """sk/su: knownness scores of known/unknown test windows; th: per-class thresholds (reject when score <= th[pred])."""
    s = np.r_[sk, su]; known = np.r_[np.ones(len(sk)), np.zeros(len(su))]; fpr, tpr, _ = roc_curve(known, s)
    rk, ru = sk <= th[pk], su <= th[pu]
    f1 = f1_score(np.r_[yk, np.full(len(pu), K)], np.r_[np.where(rk, K, pk), np.where(ru, K, pu)], labels=list(range(K + 1)), average="macro", zero_division=0)
    return dict(auroc=float(roc_auc_score(known, s)), aupr_unk=float(average_precision_score(1 - known, -s)),
                fpr95=float(fpr[np.searchsorted(tpr, 0.95)]), open_f1=float(f1), unk_recall=float(ru.mean()), known_reject=float(rk.mean()))
