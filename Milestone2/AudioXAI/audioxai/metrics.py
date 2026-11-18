"""Per-clip aggregate metrics and bootstrap statistics."""
import numpy as np

from .core import CLASSES, LABELS3, SEG_S, flagged_mask, to_seg


def word_frames(w, n_seg):
    a = int(np.floor(w["t0"] / SEG_S)); b = int(np.ceil(w["t1"] / SEG_S))
    a = min(max(a, 0), n_seg - 1); b = min(max(b, a + 1), n_seg)
    return a, b


def clip_metrics(res):
    n_seg = res["n_seg"]
    meta = res["meta"]
    row = dict(file=meta["file"], label=meta["label"], original_index=meta["original_index"], slice=meta["slice"],
               duration=round(res["dur"], 3), n_tokens=len(res["keep"]), n_words=len(res["words"]))
    flag = flagged_mask(res["aadd"], n_seg)
    row["aadd_available"] = flag is not None
    if flag is not None:
        a = res["aadd"][:n_seg]
        p95 = np.percentile(a, 95, axis=0)
        row["frac_flagged"] = float(flag.mean())
        row["n_flagged_frames"] = int(flag.sum())
        row["aadd_p95_max"] = float(p95.max())
        row["aadd_clip_flag"] = bool(p95.max() >= 0.5)
        row["degradation_class"] = CLASSES[int(p95.argmax())] if p95.max() >= 0.5 else "clean"
        row["degradation_class_argmax"] = CLASSES[int(p95.argmax())]
    # M1 clip profile: |gxi| summed over mel, each token normalized, averaged over tokens
    if len(res["keep"]) and flag is not None:
        prof = to_seg(np.abs(res["maps"]).sum(1), n_seg)
        prof = prof / np.maximum(prof.sum(1, keepdims=True), 1e-12)
        clipprof = prof.mean(0)
        m4 = res["m4"][:, :n_seg].mean(0); m4 = m4 / max(m4.sum(), 1e-12)
        if flag.any():
            row["m1_mass_flagged"] = float(clipprof[flag].sum())
            row["focus_ratio_m1"] = row["m1_mass_flagged"] / row["frac_flagged"]
            row["focus_ratio_m4"] = float(m4[flag].sum()) / row["frac_flagged"]
            tokfr = prof[:, flag].sum(1) / row["frac_flagged"]
            row["focus_ratio_m1_token_median"] = float(np.median(tokfr))
        else:
            row["m1_mass_flagged"] = 0.0
            row["focus_ratio_m1"] = np.nan; row["focus_ratio_m4"] = np.nan; row["focus_ratio_m1_token_median"] = np.nan
    # errors inside flagged vs clean
    cnt = res["cnt"]
    row.update(S=cnt["S"], D=cnt["D"], I=cnt["I"], H=cnt["H"], N_ref=cnt["N"],
               wer=(cnt["S"] + cnt["D"] + cnt["I"]) / max(cnt["N"], 1))
    if flag is not None:
        e_in = e_out = n_in = n_out = 0
        for t, is_err, kind in res["events"]:
            if t is None:
                continue
            f = int(min(max(np.floor(t / SEG_S), 0), n_seg - 1))
            inside = bool(flag[f])
            if kind != "ins":
                if inside: n_in += 1
                else: n_out += 1
            if is_err:
                if inside: e_in += 1
                else: e_out += 1
        row.update(err_in=e_in, err_out=e_out, nref_in=n_in, nref_out=n_out,
                   wer_flagged=e_in / n_in if n_in else np.nan, wer_clean=e_out / n_out if n_out else np.nan)
    # sentiment importance
    ab = np.asarray(res["ig_abs"], float)
    if len(ab) and ab.sum() > 0:
        tags = res["tags"]
        err = np.array([t != "ok" for t in tags])
        row["err_word_share"] = float(err.mean())
        row["err_word_imp_share"] = float(ab[err].sum() / ab.sum())
        if flag is not None:
            inside = np.array([flag[slice(*word_frames(w, n_seg))].mean() for w in res["words"]])
            row["sent_imp_share_flagged"] = float((ab * inside).sum() / ab.sum())
            row["word_share_flagged"] = float(inside.mean())
    pred = LABELS3[int(np.argmax(res["probs"]))]
    predg = LABELS3[int(np.argmax(res["probs_gold"]))]
    row.update(pred=pred, pred_gold_text=predg, correct=int(pred == meta["label"]),
               correct_gold_text=int(predg == meta["label"]),
               p_label=float(res["probs"][LABELS3.index(meta["label"])]))
    return row


def bootstrap(fn, n, B=10000, seed=8001):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) == 0:
        return [np.nan, np.nan], 0
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def pointbiserial(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
