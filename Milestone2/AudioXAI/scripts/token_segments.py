"""Per-token top-segment and mel-band statistics from the saved AudioXAI .npz arrays (CPU only).
Definitions (shared with figures/render_xai_tokens.py):
  token time      = token_ts[keep[k]+1]
  M2 time profile = sum_b clip(m2[k,b,t],0) over 100 ms patches; top segment = argmax t, centre (t+0.5)*0.1 s
  M2 band mass    = sum_t clip(m2,0) per band (8 bands of 10 mel bins), normalized per token
  M1 time profile = sum_mel |m1[k]| summed into 100 ms patches; top segment = argmax; band mass = |m1| into 8 bands
  distance        = |top-segment centre - token time| in seconds
Outputs results/token_segments/tokens.csv (one row per token) and summary.json (per model, bootstrap over clips)."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import WhisperTokenizer

X = Path(__file__).resolve().parents[1]
RUNS = {"whisper-tiny": X / "results/npz", "whisper-base": X / "results/whisper-base/npz", "aventiq": X / "results/aventiq/npz"}
SAMPLE = list(pd.read_csv(X / "results/test_sample_ids.csv")["file"])
tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
agg = pd.read_csv(X / "results/aggregate.csv").set_index("file")
GRP = pd.read_csv(X / "results/token_segments/groups.csv").set_index("clip")["group"].to_dict()
PT = 10

def to_patches(prof):
    nt = int(np.ceil(len(prof) / PT))
    return np.pad(prof, (0, nt * PT - len(prof))).reshape(nt, PT).sum(1)

rows = []
for model, d in RUNS.items():
    for c in SAMPLE:
        p = d / f"{c}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        keep, ts, seq = z["keep"], z["token_ts"], z["seq"]
        m1 = np.abs(z["m1"].astype(np.float32))
        pm2 = d / f"{c}_m2.npz"
        m2 = np.clip(np.load(pm2)["m2"].astype(np.float32), 0, None) if pm2.exists() else None
        cls = agg.loc[c, "degradation_class"] if c in agg.index else ""
        for k in range(len(keep)):
            t_tok = float(ts[keep[k] + 1])
            r = dict(model=model, clip=c, group=GRP.get(c, "background"), k=k, token=tok.decode([int(seq[keep[k] + 1])]), t_tok=t_tok, degradation_class=cls)
            prof1 = to_patches(m1[k].sum(0)); b1 = m1[k].reshape(8, 10, -1).sum((1, 2))
            if prof1.sum() > 0:
                r["m1_top_t"] = int(prof1.argmax()); r["m1_dist"] = abs((r["m1_top_t"] + 0.5) * 0.1 - t_tok)
                r["m1_top_band"] = int(b1.argmax())
                for b in range(8): r[f"m1_band{b}"] = float(b1[b] / b1.sum())
            if m2 is not None and m2[k].sum() > 0:
                prof2 = m2[k].sum(0); b2 = m2[k].sum(1)
                r["m2_top_t"] = int(prof2.argmax()); r["m2_dist"] = abs((r["m2_top_t"] + 0.5) * 0.1 - t_tok)
                r["m2_top_band"] = int(b2.argmax())
                r["m2_top_share"] = float(prof2.max() / prof2.sum())
                for b in range(8): r[f"m2_band{b}"] = float(b2[b] / b2.sum())
            rows.append(r)
df = pd.DataFrame(rows)
out = X / "results/token_segments"; out.mkdir(exist_ok=True)
df.to_csv(out / "tokens.csv", index=False)

rng = np.random.default_rng(8001)
def ci_clips(sub, col, stat=np.median, B=10000):
    per = sub.groupby("clip")[col].apply(stat).dropna().values
    if len(per) == 0: return None
    idx = rng.integers(0, len(per), (B, len(per)))
    boots = np.nanmean(per[idx], 1)
    return dict(n_clips=int(len(per)), n_tokens=int(sub[col].notna().sum()), mean_of_clip_medians=float(per.mean()),
                ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
S = {"groups": "hard = 10 clips with the lowest mean whisper-tiny token log-prob among the 50; mixed = 10 stratified clips (confidence x length x clean/degraded, default_rng(8001)); background = the other 30; all50 = every clip (background numbers, see groups.csv)", "definitions": __doc__, "sample": "50-clip stratified test sample shared by the three models (results/test_sample_ids.csv); M2 exists for all 50 on whisper-tiny and for 5 representative clips on whisper-base and AventIQ"}
for (model, grp), sub in list(df.groupby(["model", "group"])) + [((m_, "all50"), g_) for m_, g_ in df.groupby("model")]:
    m = {"n_clips": int(sub["clip"].nunique()), "n_tokens": int(len(sub))}
    for meth in ("m1", "m2"):
        if f"{meth}_dist" not in sub or sub[f"{meth}_dist"].notna().sum() == 0: continue
        s2 = sub[sub[f"{meth}_dist"].notna()]
        m[f"{meth}_dist_median_s"] = ci_clips(s2, f"{meth}_dist")
        m[f"{meth}_within_0.3s"] = ci_clips(s2.assign(w=(s2[f"{meth}_dist"] <= 0.3).astype(float)), "w", np.mean)
        m[f"{meth}_within_0.5s"] = ci_clips(s2.assign(w=(s2[f"{meth}_dist"] <= 0.5).astype(float)), "w", np.mean)
        m[f"{meth}_band_mass_mean"] = [float(s2[f"{meth}_band{b}"].mean()) for b in range(8)]
        m[f"{meth}_top_band_hist"] = [int((s2[f"{meth}_top_band"] == b).sum()) for b in range(8)]
        if meth == "m2": m["m2_top_share_median"] = ci_clips(s2, "m2_top_share")
    both = sub.dropna(subset=["m1_dist", "m2_dist"]) if "m2_dist" in sub else sub.iloc[0:0]
    if len(both):
        m["m1_m2_same_top_segment"] = ci_clips(both.assign(w=(both.m1_top_t == both.m2_top_t).astype(float)), "w", np.mean)
    S.setdefault(model, {})[grp] = m
json.dump(S, open(out / "summary.json", "w"), indent=1)
for model in RUNS:
    for grp, m in S.get(model, {}).items():
        for meth in ("m1", "m2"):
            if f"{meth}_dist_median_s" in m:
                x = m[f"{meth}_dist_median_s"]; w = m[f"{meth}_within_0.3s"]
                print(f"{model:13} {grp:10} {meth}: clips {x['n_clips']:2d} tokens {x['n_tokens']:4d} median {x['mean_of_clip_medians']:.3f} [{x['ci'][0]:.3f},{x['ci'][1]:.3f}] within0.3 {w['mean_of_clip_medians']:.3f} [{w['ci'][0]:.3f},{w['ci'][1]:.3f}] bands {[round(v,2) for v in m[f'{meth}_band_mass_mean']]}")
