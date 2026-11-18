"""Clip selection, batch scanning (resumable) and the dataset summary with H1-H3 verdicts."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, paths
from .metrics import bootstrap, clip_metrics, pointbiserial


def test_sample(n=50, min_per=3, seed=8001):
    if not paths.TEST_UNLOCKED.exists():
        raise PermissionError("TEST_UNLOCKED missing")
    _, test, _ = core.slices()
    t = test.sort_values("original_index").reset_index(drop=True)
    counts = t.label.value_counts().sort_index()
    quota = counts / counts.sum() * n
    k = np.floor(quota).astype(int).clip(lower=min_per)
    while k.sum() < n:  # largest remainder among labels
        rem = (quota - k).sort_values(ascending=False)
        k[rem.index[0]] += 1
    while k.sum() > n:
        rem = (quota - k).sort_values()
        for lab in rem.index:
            if k[lab] > min_per:
                k[lab] -= 1; break
    rng = np.random.default_rng(seed)
    parts = []
    for lab in sorted(counts.index):
        g = t[t.label == lab].reset_index(drop=True)
        parts.append(g.iloc[rng.permutation(len(g))[:k[lab]]])
    s = pd.concat(parts)[["file", "original_index", "label"]].reset_index(drop=True)
    return s


def dev_clips():
    dev, test, load = core.slices()
    if paths.WSP_READY.exists() and paths.WSP_HOLDOUT.exists():
        ids = set(np.load(paths.WSP_HOLDOUT).tolist())
        d = dev[dev.original_index.isin(ids)].sort_values("original_index")
        clips = list(d.file)
    else:
        clips = list(paths.SMOKE)
    assert not set(clips) & set(test.file) and not set(clips) & load
    return clips


def out_dir(slice_):
    return paths.X / ("results" if slice_ == "test" else "results/dev")


def scan(slice_="test", sample=50, limit=None, faith_deletion=20, faith_random=10, smooth_n=8):
    from . import faithfulness as F
    od = out_dir(slice_); od.mkdir(parents=True, exist_ok=True)
    if slice_ == "test":
        s = test_sample(sample)
        s.to_csv(od / "test_sample_ids.csv", index=False)
        clips = list(s.file)
    else:
        clips = dev_clips()[:sample]
    rows_f = od / "clip_rows.jsonl"; faith_f = od / "faith_rows.jsonl"
    done = set()
    if rows_f.exists():
        done = {json.loads(l)["file"] for l in open(rows_f)}
    fdone = {}
    if faith_f.exists():
        for l in open(faith_f):
            r = json.loads(l); fdone.setdefault(r["file"], set()).add(r["kind"])
    n_proc = 0
    for i, c in enumerate(clips):
        need_del = i < faith_deletion and "deletion" not in fdone.get(c, set())
        need_rnd = i < faith_random and "randomization" not in fdone.get(c, set())
        if c in done and not need_del and not need_rnd:
            continue
        if limit is not None and n_proc >= limit:
            break
        t0 = time.time()
        res = core.explain_clip(c, smooth_n=smooth_n, save_dir=od / "npz")
        if c not in done:
            row = clip_metrics(res)
            row["secs"] = round(time.time() - t0, 2)
            with open(rows_f, "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
        if need_del and len(res["keep"]):
            d = F.deletion(res); d["kind"] = "deletion"
            with open(faith_f, "a") as f: f.write(json.dumps(d) + "\n")
        if need_rnd and len(res["keep"]):
            d = F.randomization(res, smooth_n=smooth_n); d["kind"] = "randomization"
            with open(faith_f, "a") as f: f.write(json.dumps(d) + "\n")
        n_proc += 1
        print(f"[{i+1}/{len(clips)}] {c} {time.time()-t0:.1f}s", flush=True)
    write_tables(slice_)
    remaining = [c for c in clips if c not in {json.loads(l)["file"] for l in open(rows_f)}] if rows_f.exists() else clips
    return len(remaining)


def write_tables(slice_):
    od = out_dir(slice_)
    rows_f = od / "clip_rows.jsonl"
    if rows_f.exists():
        df = pd.DataFrame([json.loads(l) for l in open(rows_f)]).drop_duplicates("file", keep="last")
        if slice_ == "test" and (od / "test_sample_ids.csv").exists():
            order = list(pd.read_csv(od / "test_sample_ids.csv").file)
            df = df.set_index("file").reindex([c for c in order if c in set(df.file)]).reset_index()
        df.to_csv(od / "aggregate.csv", index=False)
    faith_f = od / "faith_rows.jsonl"
    if faith_f.exists():
        fr = [json.loads(l) for l in open(faith_f)]
        dl = pd.DataFrame([r for r in fr if r["kind"] == "deletion"]).drop_duplicates("file", keep="last")
        rn = pd.DataFrame([r for r in fr if r["kind"] == "randomization"]).drop_duplicates("file", keep="last")
        if len(dl) and len(rn):
            f = dl.drop(columns="kind").merge(rn.drop(columns="kind"), on="file", how="outer")
        else:
            f = (dl if len(dl) else rn).drop(columns="kind")
        f.to_csv(od / "faithfulness.csv", index=False)


def _ci_mean(x, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, ci=[np.nan, np.nan])
    ci, _ = bootstrap(lambda i: x[i].mean(), len(x), B)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), ci=ci)


def _ci_pb(df, col, B=10000):
    d = df[[col, "correct"]].dropna()
    x, y = d[col].to_numpy(float), d["correct"].to_numpy(float)
    r = pointbiserial(x, y)
    ci, nb = bootstrap(lambda i: pointbiserial(x[i], y[i]), len(x), B)
    return dict(n=int(len(x)), r=r, ci=ci, n_valid_boot=nb)


def find_h3():
    """Look for Phase 1's gold-text ablation for Cardiff with whisper-tiny in T1_final/results."""
    cands = sorted(paths.T1RES.glob("*gold*")) + sorted(paths.T1RES.glob("*ablation*"))
    return [str(c) for c in cands]


def summarize(slice_="test", h3=None):
    od = out_dir(slice_)
    df = pd.read_csv(od / "aggregate.csv")
    S = dict(slice=slice_, n_clips=int(len(df)), asr=paths.ASR_ID, asr_rev=paths.ASR_REV, sentiment=paths.SENT_ID,
             sentiment_rev=paths.SENT_REV, aadd="aadd 3.1.0 CLI, model v3 (registry default), 20 ms segments",
             flag_rule_frame="20 ms frame flagged if any class probability >= 0.5",
             flag_rule_clip="AADD published default: per-class 95th percentile >= 0.5",
             aadd_caveat="AADD is trained on synthetic LibriSpeech degradations; VoxCeleb is real YouTube audio, so "
                         "flags are estimates.", bootstrap="10,000 resamples, numpy default_rng(8001), percentile CI",
             label_counts=df.label.value_counts().to_dict())
    if "frac_flagged" in df:
        S["n_zero_flagged"] = int((df.n_flagged_frames == 0).sum())
        S["n_aadd_clip_flag"] = int(df.aadd_clip_flag.sum())
        S["mean_frac_flagged"] = float(df.frac_flagged.mean())
        S["degradation_class_counts"] = df.degradation_class.value_counts().to_dict()
    fr = _ci_mean(df.get("focus_ratio_m1", pd.Series(dtype=float)))
    S["focus_ratio_m1"] = fr
    S["focus_ratio_m4"] = _ci_mean(df.get("focus_ratio_m4", pd.Series(dtype=float)))
    S["focus_ratio_m1_by_class"] = {k: _ci_mean(g.focus_ratio_m1) for k, g in df.groupby("degradation_class_argmax")} \
        if "degradation_class_argmax" in df else {}
    h1 = "not testable"
    if fr["n"] >= 3:
        lo, hi = fr["ci"]
        h1 = "holds" if lo > 1 else ("refuted (CI below 1)" if hi < 1 else "refuted (CI includes 1)")
    S["H1"] = dict(verdict=h1, statistic="mean focus ratio M1 over clips with >=1 flagged frame", **fr)
    # H2
    if "wer_flagged" in df:
        e_in, n_in, e_out, n_out = df.err_in.sum(), df.nref_in.sum(), df.err_out.sum(), df.nref_out.sum()
        both = df[(df.nref_in > 0) & (df.nref_out > 0)]
        diff = (both.wer_flagged - both.wer_clean).to_numpy(float)
        dci = _ci_mean(diff)
        d2 = df[["err_word_imp_share", "err_word_share"]].dropna()
        d2v = (d2.err_word_imp_share - d2.err_word_share).to_numpy(float)
        ici = _ci_mean(d2v)
        a_ok = dci["n"] >= 3 and dci["ci"][0] > 0
        a_ref = dci["n"] >= 3 and dci["ci"][1] <= 0 or (n_in and n_out and e_in / n_in <= e_out / n_out)
        b_ok = ici["n"] >= 3 and ici["ci"][1] < 0
        b_ref = ici["n"] >= 3 and ici["ci"][0] >= 0 or (ici["n"] and ici["mean"] >= 0)
        if a_ok and b_ok:
            v = "holds"
        elif a_ref or b_ref:
            v = "refuted"
        else:
            v = "not supported (CIs include 0)"
        S["H2"] = dict(verdict=v,
                       pooled_err_rate_flagged=float(e_in / n_in) if n_in else None,
                       pooled_err_rate_clean=float(e_out / n_out) if n_out else None,
                       n_ref_words_flagged=int(n_in), n_ref_words_clean=int(n_out),
                       paired_wer_diff_flagged_minus_clean=dci,
                       err_imp_share_minus_word_share=ici,
                       mean_err_word_share=float(d2.err_word_share.mean()) if len(d2) else None,
                       mean_err_word_imp_share=float(d2.err_word_imp_share.mean()) if len(d2) else None,
                       sent_imp_share_flagged=_ci_mean(df.get("sent_imp_share_flagged", pd.Series(dtype=float))),
                       word_share_flagged=_ci_mean(df.get("word_share_flagged", pd.Series(dtype=float))))
    # correctness associations
    S["pointbiserial_vs_correct"] = {c: _ci_pb(df, c) for c in
                                    ["focus_ratio_m1", "focus_ratio_m4", "wer", "wer_flagged", "wer_clean",
                                     "sent_imp_share_flagged", "frac_flagged", "err_word_imp_share"] if c in df}
    S["cardiff_acc_asr_text_sample"] = float(df.correct.mean())
    S["cardiff_acc_gold_text_sample"] = float(df.correct_gold_text.mean())
    S["mean_wer_sample"] = float(df.wer.mean())
    S["H3"] = h3 or dict(verdict="pending: Phase 1 gold-text ablation not found", candidates=find_h3())
    fp = od / "faithfulness.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        if "auc_top" in f:
            d = f.dropna(subset=["auc_top"])
            S["deletion"] = dict(n=int(len(d)), auc_top=_ci_mean(d.auc_top), auc_random=_ci_mean(d.auc_random),
                                 auc_diff=_ci_mean(d.auc_diff), frac_clips_top_gt_random=float((d.auc_diff > 0).mean()),
                                 unit="mean per-token teacher-forced log-prob drop (nats), AUC over k=5..50% patches")
        if "rho_tf_abs" in f:
            d = f.dropna(subset=["rho_tf_abs"])
            S["randomization"] = dict(n=int(len(d)), rho_tf_abs=_ci_mean(d.rho_tf_abs), rho_time_abs=_ci_mean(d.rho_time_abs),
                                      rho_tf_signed=_ci_mean(d.rho_tf_signed),
                                      rho_map_vs_absinput=_ci_mean(d.rho_map_vs_absinput),
                                      verdict="PASS" if d.rho_tf_abs.mean() < 0.5 else "FAIL",
                                      rule="PASS only if mean Spearman(|M1| original, |M1| top-2-decoder-reinit) < 0.5")
    json.dump(S, open(od / "summary.json", "w"), indent=2, default=float)
    return S
