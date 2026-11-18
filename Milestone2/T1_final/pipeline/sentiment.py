import os, time, json
import numpy as np, pandas as pd, torch
from transformers import pipeline
from .common import LABELS

FIVE_TO_THREE = {"very negative": "Negative", "negative": "Negative", "neutral": "Neutral", "positive": "Positive", "very positive": "Positive"}

def get_clf(name, revision, device=0):
    return pipeline("text-classification", model=name, revision=revision, device=device, top_k=None)

def score_texts(name, revision, texts, keys, cache_path, device=0, batch_size=32, log_path=None):
    """Full probability vector per text, cached by key. Columns: key, raw_label, raw_score, p_<label>..."""
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path).set_index("key")
        if all(k in cached.index for k in keys): return cached.loc[keys].reset_index()
        todo = [(k, t) for k, t in zip(keys, texts) if k not in cached.index]
    else:
        cached = None; todo = list(zip(keys, texts))
    clf = get_clf(name, revision, device)
    t0 = time.time(); rows = []
    tx = [("" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)) for _, t in todo]
    tx = [t if t.strip() else "." for t in tx]   # empty transcripts still get a score vector (and fail A1 downstream)
    outs = clf(tx, batch_size=batch_size, truncation=True, max_length=512)
    for (k, _), o in zip(todo, outs):
        r = {"key": k}
        best = max(o, key=lambda d: d["score"])
        r["raw_label"] = best["label"]; r["raw_score"] = best["score"]
        for d in o: r[f"p_{d['label'].lower().replace(' ', '_')}"] = d["score"]
        rows.append(r)
    new = pd.DataFrame(rows).set_index("key")
    allc = pd.concat([cached, new]) if cached is not None else new
    allc.reset_index().to_csv(cache_path, index=False)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": f"sentiment:{name}", "event": "end", "pid": os.getpid(),
                                "note": f"{len(todo)} texts, {time.time()-t0:.0f}s, cache {os.path.basename(cache_path)}"}) + "\n")
    del clf; torch.cuda.empty_cache()
    return allc.loc[keys].reset_index()

def map_predictions(kind, scores, tau=None):
    """Guide mappings. kind: three (Cardiff as-is), five (tabularisai collapsed), binary (Neutral below tau)."""
    raw = scores["raw_label"].astype(str); sc = scores["raw_score"].astype(float)
    if kind == "three":
        return raw.str.capitalize()
    if kind == "five":
        return raw.str.lower().map(FIVE_TO_THREE)
    if kind == "binary":
        assert tau is not None
        return pd.Series(np.where(sc.values < tau, "Neutral", raw.str.capitalize().values), index=scores.index)
    raise ValueError(kind)

def tune_tau(kind, dev_scores, dev_labels, grid):
    """tau in grid maximizing dev macro-F1 (first maximum wins, as in the notebook). None for non-binary models."""
    from sklearn.metrics import f1_score
    if kind != "binary": return None, None
    best = None
    for g in grid:
        f = f1_score(dev_labels, map_predictions(kind, dev_scores, g), average="macro", labels=LABELS)
        if best is None or f > best[1]: best = (g, f)
    return best
