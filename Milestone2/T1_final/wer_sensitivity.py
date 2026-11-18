#!/usr/bin/env python
"""Sensitivity check for the SLUE span/WAV duration mismatch: corpus WER vs gold on the span-matched test clips
(|SLUE span - WAV duration| <= 0.1 s) versus the mismatched ones. CPU only, from saved transcripts."""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.common import load_config, repo_root, short, normalize_text
from pipeline.metrics import corpus_wer
cfg = load_config(); root = repo_root(); RD = os.path.join(root, cfg["results_dir"]); CD = os.path.join(root, cfg["cache_dir"])
j = pd.read_csv(os.path.join(RD, "gold_join.csv")); j = j[j["slice"] == "test"].set_index("file")
test = pd.read_csv(os.path.join(root, cfg["slices_dir"], "test.csv")).set_index("file")
matched = j["dur_delta"].abs() <= 0.1
rows = []
cands = {n: pd.read_csv(os.path.join(CD, f"transcripts_{short(n)}.csv")).set_index("file")["transcript"] for n in cfg["asr_models"]}
cands["openai/whisper-large-v3"] = test[cfg["pseudo_ref_col"]]
for name, tr in cands.items():
    h = tr.reindex(j.index).fillna(""); g = j["gold_text"].fillna("")
    for lab, m in [("span_matched", matched), ("span_mismatched", ~matched), ("all", pd.Series(True, index=j.index))]:
        w, _ = corpus_wer(g[m].tolist(), h[m].tolist()); rows.append({"asr": name, "subset": lab, "n": int(m.sum()), "wer_gold": w})
out = pd.DataFrame(rows); out.to_csv(os.path.join(RD, "wer_test_spanmatched.csv"), index=False); print(out.pivot(index="asr", columns="subset", values="wer_gold").round(4).to_string())
