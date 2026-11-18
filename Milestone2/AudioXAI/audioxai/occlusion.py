"""M2: spectrogram-patch occlusion in the spirit of SPES (arXiv:2411.01710).
Each 100 ms x 10-mel-bin patch (1/8 of the mel range) is set to the clip's log-mel floor, one at a time; the score is the
drop in each generated token's teacher-forced log-prob. Model-agnostic over the log-mel input."""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import core, paths
from .faithfulness import PF, PT


def m2_occlusion(res, batch=32):
    _, m = core.asr()
    feats, _ = core.features(res["wav"])
    seq = torch.as_tensor(res["seq"], device=core.DEV)
    keep = list(res["keep"])
    nf = res["n_frames"]
    nt = int(np.ceil(nf / PT)); nb = 80 // PF
    fill = float(feats[0, :, :nf].min())
    idx = [(b, t) for b in range(nb) for t in range(nt)]
    out = np.zeros((len(keep), nb, nt), np.float32)
    with torch.no_grad():
        lp0 = core.teacher_forced_logp(m, feats, seq, keep)[0]
        for i in range(0, len(idx), batch):
            chunk = idx[i:i + batch]
            x = feats.expand(len(chunk), -1, -1).clone()
            for j, (b, t) in enumerate(chunk):
                x[j, b * PF:(b + 1) * PF, t * PT:min((t + 1) * PT, nf)] = fill
            lp = core.teacher_forced_logp(m, x, seq, keep)
            d = (lp0[None] - lp).float().cpu().numpy()  # (B,K)
            for j, (b, t) in enumerate(chunk):
                out[:, b, t] = d[j]
    return out


def m1_m2_agreement(res, m2):
    a = np.abs(res["maps"]).sum(1)  # (K, nf)
    nf = a.shape[1]; nt = m2.shape[2]
    a = np.pad(a, ((0, 0), (0, nt * PT - nf))).reshape(len(a), nt, PT).sum(-1)
    b = m2.sum(1)
    rhos = [spearmanr(a[k], b[k]).statistic for k in range(len(a)) if np.std(b[k]) > 0 and np.std(a[k]) > 0]
    return float(np.median(rhos)) if rhos else np.nan, len(rhos)


def run(slice_="test", n=20):
    from .scan import out_dir
    od = out_dir(slice_)
    agg = pd.read_csv(od / "aggregate.csv")
    # positives and negatives first, then neutrals, in sample order
    order = list(agg[agg.label != "Neutral"].file) + list(agg[agg.label == "Neutral"].file)
    rows_f = od / "m2_rows.jsonl"
    done = {json.loads(l)["file"] for l in open(rows_f)} if rows_f.exists() else set()
    for c in order[:n]:
        if c in done:
            continue
        res = core.explain_clip(c)  # deterministic recompute (same seed) to get maps and wav
        m2 = m2_occlusion(res)
        np.savez_compressed(od / "npz" / f"{c}_m2.npz", m2=m2.astype(np.float16))
        rho, k = m1_m2_agreement(res, m2)
        from .core import flagged_mask
        flag = flagged_mask(res["aadd"], res["n_seg"])
        row = dict(file=c, label=res["meta"]["label"], m1_m2_rho_time_median=rho, n_tokens_rho=k)
        if flag is not None and flag.any():
            prof = np.clip(m2, 0, None).sum(1)
            prof = prof / np.maximum(prof.sum(1, keepdims=True), 1e-12)
            prof = prof.mean(0)  # (nt) at 100 ms
            f100 = np.array([flag[t * 5:(t + 1) * 5].mean() if t * 5 < len(flag) else 0 for t in range(len(prof))])
            ff = f100.mean()
            row["focus_ratio_m2"] = float((prof * f100).sum() / ff) if ff > 0 else np.nan
        with open(rows_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("m2", c, row, flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(rows_f)]).drop_duplicates("file", keep="last")
    df.to_csv(od / "m2.csv", index=False)
    return df


def segment_replacement(clip, win_s=0.5, hop_s=0.25, batch=16):
    """Item 7: replace sliding 500 ms windows (250 ms stride) with the clip's own noise floor (its lowest-RMS 100 ms
    frame, tiled), re-transcribe with whisper-tiny, re-classify with Cardiff, record the change in P(gold label)."""
    proc, m = core.asr()
    wav, meta = core.load_audio(clip)
    fr = int(0.1 * core.SR)
    frames = [wav[i:i + fr] for i in range(0, len(wav) - fr + 1, fr)]
    floor = frames[int(np.argmin([np.sqrt((f ** 2).mean()) for f in frames]))]
    W, H = int(win_s * core.SR), int(hop_s * core.SR)
    starts = list(range(0, max(len(wav) - W, 0) + 1, H))
    variants = []
    for s in starts:
        w = wav.copy(); seg = np.tile(floor, W // fr + 1)[:min(W, len(wav) - s)]
        w[s:s + len(seg)] = seg
        variants.append(w)
    texts = []
    for i in range(0, len(variants), batch):
        f = proc.feature_extractor(variants[i:i + batch], sampling_rate=core.SR, return_tensors="pt",
                                   return_attention_mask=True)
        with torch.no_grad():
            ids = m.generate(f.input_features.to(core.DEV), attention_mask=f.attention_mask.to(core.DEV),
                             language="en", task="transcribe")
        texts += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    feats, am = core.features(wav)
    seq, _ = core.transcribe(feats, am)
    base_text = proc.tokenizer.decode(seq, skip_special_tokens=True).strip()
    p = core.sentiment_predict([base_text] + [t or " " for t in texts])
    li = core.LABELS3.index(meta["label"])
    dp = p[0, li] - p[1:, li]  # positive = removing the window lowers P(gold)
    return dict(starts=np.array(starts) / core.SR, dp=dp, texts=texts, base_text=base_text, p0=float(p[0, li]))


def run_e2e(slice_="test", n=10):
    from .scan import out_dir
    od = out_dir(slice_)
    agg = pd.read_csv(od / "aggregate.csv")
    order = list(agg[agg.label != "Neutral"].file) + list(agg[agg.label == "Neutral"].file)
    rows = []
    for c in order[:n]:
        r = segment_replacement(c)
        np.savez_compressed(od / "npz" / f"{c}_e2e.npz", starts=r["starts"], dp=r["dp"].astype(np.float32))
        aadd, _ = core.aadd_frames(c)
        row = dict(file=c, p_gold_full=r["p0"], max_drop=float(r["dp"].max()), mean_abs_change=float(np.abs(r["dp"]).mean()),
                   n_windows=len(r["dp"]), frac_windows_change_gt_0p1=float((np.abs(r["dp"]) > 0.1).mean()))
        if aadd is not None:
            fl = (aadd >= 0.5).any(1)
            wf = np.array([fl[int(s / 0.02):int((s + 0.5) / 0.02)].mean() if int(s / 0.02) < len(fl) else 0 for s in r["starts"]])
            row["dp_flagged_windows"] = float(np.abs(r["dp"][wf >= 0.5]).mean()) if (wf >= 0.5).any() else np.nan
            row["dp_clean_windows"] = float(np.abs(r["dp"][wf < 0.5]).mean()) if (wf < 0.5).any() else np.nan
        rows.append(row); print("e2e", row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(od / "e2e_segments.csv", index=False)
    return df
