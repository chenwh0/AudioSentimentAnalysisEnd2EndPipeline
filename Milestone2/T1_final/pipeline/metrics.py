import numpy as np, pandas as pd, jiwer
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr
from .common import LABELS, normalize_text

def corpus_wer(refs, hyps, norm=normalize_text):
    r = [norm(x) for x in refs]; h = [norm(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x != ""]
    return float(jiwer.wer([r[i] for i in keep], [h[i] for i in keep])), len(r) - len(keep)

def per_clip_wer(refs, hyps, norm=normalize_text):
    out = []
    for a, b in zip(refs, hyps):
        ra, hb = norm(a), norm(b)
        out.append(np.nan if ra == "" else float(jiwer.wer(ra, hb if hb else "")))
    return np.array(out)

def classification(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str); y_pred = pd.Series(y_pred).astype(str)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    row = {"macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
           "accuracy": float(accuracy_score(y_true, y_pred))}
    for i, L in enumerate(LABELS):
        row[f"P_{L}"] = float(p[i]); row[f"R_{L}"] = float(r[i]); row[f"F1_{L}"] = float(f[i]); row[f"n_{L}"] = int(s[i])
    return row

def _encode(y_true, y_pred):
    li = {L: i for i, L in enumerate(LABELS)}
    t = np.array([li.get(str(x), -1) for x in y_true]); p = np.array([li.get(str(x), -1) for x in y_pred])
    assert (t >= 0).all(), "unknown true label"
    p = np.where(p < 0, 3, p)   # unmapped prediction counts as wrong for every class
    return t, p

def _macro_f1_from_counts(cnt):
    """cnt: (B, 3, 4) confusion counts [true, pred]; macro-F1 over the 3 labels per resample."""
    f1s = []
    for c in range(3):
        tp = cnt[:, c, c]; fp = cnt[:, :, c].sum(1) - tp; fn = cnt[:, c, :].sum(1) - tp
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)

def _counts(t, p, idx):
    B, n = idx.shape
    k = (t * 4 + p)[idx] + 12 * np.arange(B)[:, None]
    return np.bincount(k.ravel(), minlength=12 * B).reshape(B, 3, 4)

def bootstrap_macro_f1(y_true, y_pred, B=10000, seed=8001):
    t, p = _encode(y_true, y_pred); n = len(t)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B, n))
    f = _macro_f1_from_counts(_counts(t, p, idx))
    return float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))

def paired_gap_ci(y_true, pred_a, pred_b, B=10000, seed=8001):
    """95% CI of macro-F1(a) - macro-F1(b) with the same resampled clips on both sides."""
    t, pa = _encode(y_true, pred_a); _, pb = _encode(y_true, pred_b); n = len(t)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B, n))
    g = _macro_f1_from_counts(_counts(t, pa, idx)) - _macro_f1_from_counts(_counts(t, pb, idx))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))

def rank_agreement(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    pe = float(pearsonr(x, y)[0]) if len(x) > 2 else float("nan")
    sp = float(spearmanr(x, y)[0]) if len(x) > 2 else float("nan")
    same = bool(np.array_equal(np.argsort(-x), np.argsort(-y)))
    return {"pearson": pe, "spearman": sp, "same_ranking": same}
