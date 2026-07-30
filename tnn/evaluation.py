"""
Metrikler ve istatistiksel karsilastirma.

Neden ozel bir t-testi?
-----------------------
Tekrarli k-katli capraz dogrulamada foldlar bagimsiz DEGILDIR (egitim
kumeleri buyuk olcude ortusur). Duz eslesmis t-testi bu yuzden p-degerlerini
asiri iyimser (kucuk) verir. Nadeau & Bengio (2003) "corrected resampled
t-test" bu korelasyonu varyans terimine ekleyerek duzeltir; Bouckaert &
Frank (2004) tekrarli CV icin bunu onerir. Ayrica dagilim varsayimi
gerektirmeyen Wilcoxon isaretli sira testini de raporluyoruz.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)


METRIC_NAMES = ["accuracy", "balanced_accuracy", "f1_macro",
                "f1_weighted", "kappa", "auc"]


def compute_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        if n_classes == 2:
            out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                       average="macro")
    except Exception:
        out["auc"] = np.nan
    return out


def per_class_f1(y_true, y_pred, n_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None,
                    labels=list(range(n_classes)), zero_division=0)


def aggregate_confusion(y_true_list, y_pred_list, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for yt, yp in zip(y_true_list, y_pred_list):
        cm += confusion_matrix(yt, yp, labels=list(range(n_classes)))
    return cm


# --------------------------------------------------------------------------
def corrected_paired_ttest(diffs, n_train: int, n_test: int):
    """
    Nadeau-Bengio duzeltilmis eslesmis t-testi.

    diffs : fold bazli fark dizisi (yontem A - yontem B)
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    k = len(d)
    if k < 2:
        return np.nan, np.nan
    mean, var = d.mean(), d.var(ddof=1)
    if var <= 0:
        return (np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0)), \
               (0.0 if mean != 0 else 1.0)
    corr = 1.0 / k + float(n_test) / float(max(n_train, 1))
    t = mean / np.sqrt(var * corr)
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=k - 1))
    return float(t), float(p)


def wilcoxon_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.allclose(a, b):
        return np.nan, np.nan
    try:
        s, p = stats.wilcoxon(a, b)
        return float(s), float(p)
    except Exception:
        return np.nan, np.nan


def cohens_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


def holm_bonferroni(pvals):
    """Holm-Bonferroni duzeltilmis p degerleri."""
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        running = max(running, min(adj, 1.0))
        out[i] = running
    return out


def mean_ci(values, alpha: float = 0.05):
    """Ortalama ve %95 t-guven araligi."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, np.nan
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(len(v))
    h = se * stats.t.ppf(1 - alpha / 2, len(v) - 1)
    return float(m), float(m - h), float(m + h)
