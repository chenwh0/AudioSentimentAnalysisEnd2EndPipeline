"""Step 8: results table from saved harness test CSVs. Usage: python final_results.py base fp10 [fp20 ...]
Bootstrap: 10,000 resamples of clip indices, default_rng(8001), same indices for every model (paired);
corpus WER recomputed from summed errors / summed ref words. Cardiff (CPU) macro-F1 on transcripts, 3 classes as-is."""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_edits as AE
W = AE.W; RES = f"{W}/results"
ckpts = sys.argv[1:]
dfs = {c: pd.read_csv(f"{RES}/eval_{c}_test.csv").sort_values("original_index").reset_index(drop=True) for c in ckpts}
for c in ckpts:
    assert (dfs[c]["original_index"].values == dfs["base"]["original_index"].values).all()
n = len(dfs["base"])
rng = np.random.default_rng(8001)
IDX = rng.integers(0, n, size=(10000, n))
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)
def cardiff(texts):
    out = clf([str(t) if str(t).strip() else " " for t in texts], batch_size=32, truncation=True)
    return [o["label"].capitalize() for o in out]
labels = ["Negative", "Neutral", "Positive"]
def boot_f1(y, p):
    m = {l: k for k, l in enumerate(labels)}
    yi = np.array([m[v] for v in y]); pi = np.array([m.get(v, 1) for v in p])
    code = yi * 3 + pi
    out = np.empty(len(IDX))
    for b, i in enumerate(IDX):
        cm = np.bincount(code[i], minlength=9).reshape(3, 3)
        tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
        f = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        out[b] = f.mean()
    return out
rows, cache = [], {}
for c in ckpts:
    d = dfs[c]
    eg, ng = d["err_gold"].values, d["nref_gold"].values
    bw = eg[IDX].sum(1) / ng[IDX].sum(1)
    pred = cardiff(d["transcript"].fillna("").tolist())
    d["cardiff_pred"] = pred
    f1 = f1_score(d["label"], pred, labels=labels, average="macro")
    cache[c] = (bw, pred)
    rows.append({"checkpoint": c, "n_changed_weights": AE.count_changed(c), "test_wer_gold": eg.sum() / ng.sum(),
                 "wer_ci_lo": np.quantile(bw, 0.025), "wer_ci_hi": np.quantile(bw, 0.975),
                 "wer_pseudo": d["err_pseudo"].sum() / d["nref_pseudo"].sum(), "long": int(d["long"].sum()),
                 "short": int(d["short"].sum()), "A1": d["A1"].mean(), "A2": d["A2"].mean(), "cardiff_macro_f1": f1})
base_bw, base_pred = cache["base"]
for r in rows:
    c = r["checkpoint"]; bw, pred = cache[c]
    diff = bw - base_bw
    r["delta_wer_vs_base"] = r["test_wer_gold"] - rows[0]["test_wer_gold"]
    r["delta_ci_lo"], r["delta_ci_hi"] = np.quantile(diff, 0.025), np.quantile(diff, 0.975)
    y = dfs[c]["label"].values
    fb = boot_f1(y, pred)
    r["f1_ci_lo"], r["f1_ci_hi"] = np.quantile(fb, 0.025), np.quantile(fb, 0.975)
    fdiff = fb - boot_f1(y, base_pred)
    r["delta_f1_ci_lo"], r["delta_f1_ci_hi"] = np.quantile(fdiff, 0.025), np.quantile(fdiff, 0.975)
    dfs[c][["file", "original_index", "label", "transcript", "cardiff_pred"]].to_csv(f"{RES}/cardiff_{c}_test.csv", index=False)
tab = pd.DataFrame(rows)
tab.to_csv(f"{RES}/wsp_results.csv", index=False)
print(tab.round(4).to_string())
