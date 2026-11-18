#!/usr/bin/env python
"""Regenerate every T1 table from cached transcripts and scores (transcription / scoring only when a cache is missing).
Usage: .venv/bin/python Milestone2/T1_final/run_t1.py [--config PATH] [--results-dir DIR] [--asr a,b,...] [--no-gpu-note]
Linux paths from config.yaml; no Windows calls; no notebook."""
import os, sys, json, time, argparse, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.common import (LABELS, short, normalize_text, normalize_no_fillers, repo_root, load_config, load_slices, load_gold, write_json, sha256_file)
from pipeline import asr as A, sentiment as S, assertions as AS, metrics as M

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--results-dir"); ap.add_argument("--asr")
    ap.add_argument("--extra-transcripts", default=None, help="name=path.csv,... extra ASR variants with cached transcripts (file,transcript)")
    args = ap.parse_args()
    cfg = load_config(args.config); root = repo_root()
    RD = os.path.join(root, args.results_dir or cfg["results_dir"]); CD = os.path.join(root, cfg["cache_dir"]); os.makedirs(RD, exist_ok=True)
    LOG = os.path.join(root, "logs", "gpu_jobs.jsonl"); audio_dir = os.path.join(root, cfg["audio_dir"])
    B = int(cfg["bootstrap_resamples"]); seed = int(cfg["seed"])
    tau_grid = [round(x, 2) for x in np.arange(cfg["tau_grid"]["start"], cfg["tau_grid"]["stop"] + 1e-9, cfg["tau_grid"]["step"])]
    t_grid = [round(x, 2) for x in np.arange(cfg["t_grid"]["start"], cfg["t_grid"]["stop"] + 1e-9, cfg["t_grid"]["step"])]
    prov = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": args.config or "Milestone2/T1_final/config.yaml", "inputs": {}}

    sl = load_slices(cfg); dev, test, lp = sl["dev"], sl["test"], sl["load_pool"]
    dt = pd.concat([dev.assign(slice="dev"), test.assign(slice="test")]).reset_index(drop=True)
    gold = load_gold(cfg); prov["inputs"]["gold_tsv_sha256"] = sha256_file(os.path.join(root, cfg["gold_tsv"]))

    # ---------- 1. gold join checks on all 3,426 clips (no audio read: durations come from the slice CSVs)
    allc = pd.concat([dev.assign(slice="dev"), test.assign(slice="test"), lp.assign(slice="load_pool")]).reset_index(drop=True)
    j = allc[["file", "slice", "label", "original_index", "duration"]].copy()
    j["in_slue"] = j["file"].isin(gold.index)
    j["slue_sentiment"] = j["file"].map(gold["sentiment"]); j["slue_start"] = j["file"].map(gold["start_second"]); j["slue_end"] = j["file"].map(gold["end_second"])
    j["slue_duration"] = j["slue_end"] - j["slue_start"]; j["dur_delta"] = j["slue_duration"] - j["duration"]
    j["gold_text"] = j["file"].map(gold["normalized_text"]).fillna("")
    j["label_agree"] = j["label"].astype(str) == j["slue_sentiment"].astype(str)
    j.to_csv(os.path.join(RD, "gold_join.csv"), index=False)
    bad_dur = (j["dur_delta"].abs() > 0.1)
    checks = {"n_clips": int(len(j)), "coverage": int(j["in_slue"].sum()), "coverage_by_slice": j.groupby("slice")["in_slue"].sum().astype(int).to_dict(),
              "label_agreement": int(j["label_agree"].sum()), "label_mismatches": int((~j["label_agree"]).sum()),
              "label_confusion": pd.crosstab(j["label"], j["slue_sentiment"]).to_dict(),
              "duration_abs_delta_gt_0.1s": int(bad_dur.sum()), "duration_mismatch_fraction": float(bad_dur.mean()),
              "duration_abs_delta_max": float(j["dur_delta"].abs().max()), "duration_abs_delta_mean": float(j["dur_delta"].abs().mean()),
              "gold_text_empty": int((j["gold_text"].str.strip() == "").sum()),
              "verdict_durations": "OK" if bad_dur.mean() <= 0.01 else "INVESTIGATE: >1% of clips differ by >0.1 s"}
    write_json(checks, os.path.join(RD, "gold_checks.json")); print("gold checks:", {k: v for k, v in checks.items() if k != "label_confusion"})

    # ---------- 2. transcripts (cache or transcribe)
    asr_names = args.asr.split(",") if args.asr else list(cfg["asr_models"].keys())
    texts, old_texts, ms = {}, {}, {}
    for name in asr_names:
        m = cfg["asr_models"][name]; cache = os.path.join(CD, f"transcripts_{short(name)}.csv")
        if m.get("teammate_cache") and not os.path.exists(cache):
            shutil.copy(os.path.join(root, m["teammate_cache"]), cache); prov["inputs"][f"transcripts_{short(name)}"] = "copied from " + m["teammate_cache"]
        tr = A.transcribe(name, m["revision"], dt["file"].tolist(), audio_dir, int(m["batch_size"]), cache, log_path=LOG)
        texts[name] = tr.set_index("file")["transcript"].reindex(dt["file"]).fillna("").values; ms[name] = tr.set_index("file")["ms"].reindex(dt["file"]).values
        prov["inputs"][f"transcripts_{short(name)}_sha256"] = sha256_file(cache)
        if m.get("old_cache"):
            old = pd.read_csv(os.path.join(root, m["old_cache"])).set_index("file")["transcript"]
            old_texts[name] = old.reindex(dt["file"]).fillna("").values
    if args.extra_transcripts:
        for item in args.extra_transcripts.split(","):
            name, path = item.split("="); tr = pd.read_csv(os.path.join(root, path)).set_index("file")["transcript"]
            texts[name] = tr.reindex(dt["file"]).fillna("").values; asr_names.append(name); prov["inputs"][f"transcripts_{short(name)}_sha256"] = sha256_file(os.path.join(root, path))
    PSEUDO = "openai/whisper-large-v3"; texts[PSEUDO] = dt[cfg["pseudo_ref_col"]].fillna("").values
    gold_text = dt["file"].map(gold["normalized_text"]).fillna("").values
    is_test = (dt["slice"] == "test").values; is_dev = ~is_test
    y_test = dt.loc[is_test, "label"].astype(str).values; y_dev = dt.loc[is_dev, "label"].astype(str).values
    dur = dt["duration"].values

    # ---------- 3. s2t loop analysis (before = teammate transcripts, after = fixed batch-1 transcripts)
    def nwords(a): return np.array([len(str(x).split()) for x in a])
    loop = {}
    for name, old in old_texts.items():
        for sl_name, mask in [("test", is_test), ("dev", is_dev)]:
            gw = nwords([normalize_text(x) for x in gold_text[mask]]); pw = nwords([normalize_text(x) for x in texts[PSEUDO][mask]])
            ow = nwords([normalize_text(x) for x in old[mask]]); nw = nwords([normalize_text(x) for x in texts[name][mask]])
            loop[f"{short(name)}:{sl_name}"] = {"n": int(mask.sum()), "before_gt_1.5x_pseudo": int((ow > 1.5 * pw).sum()), "after_gt_1.5x_pseudo": int((nw > 1.5 * pw).sum()),
                                                  "before_gt_1.5x_gold": int((ow > 1.5 * gw).sum()), "after_gt_1.5x_gold": int((nw > 1.5 * gw).sum()),
                                                  "before_mean_words": float(ow.mean()), "after_mean_words": float(nw.mean()), "gold_mean_words": float(gw.mean())}
    write_json(loop, os.path.join(RD, "s2t_loop_analysis.json")); print("s2t loops:", loop)

    # ---------- 4. WER on test (gold primary, gold filler-free secondary, pseudo-reference for comparability)
    wrows, per = [], pd.DataFrame({"file": dt.loc[is_test, "file"].values, "label": y_test})
    cands = {**{n: texts[n] for n in asr_names}, PSEUDO: texts[PSEUDO], **{short(n) + "-OLD(batch8)": old_texts[n] for n in old_texts}}
    for name, hyp in cands.items():
        h = hyp[is_test]; g = gold_text[is_test]; p = texts[PSEUDO][is_test]
        wg, skipped = M.corpus_wer(g, h); wgf, _ = M.corpus_wer(g, h, norm=normalize_no_fillers)
        wp, _ = M.corpus_wer(p, h) if name != PSEUDO else (0.0, 0)
        wrows.append({"asr": name, "wer_gold": wg, "wer_gold_nofillers": wgf, "wer_pseudo_ref": wp, "empty_refs_skipped": skipped, "n": int(is_test.sum()),
                      "mean_ms_per_clip": float(np.nanmean(ms[name][is_test])) if name in ms else np.nan})
        per[f"wer_gold:{short(name)}"] = M.per_clip_wer(g, h); per[f"wer_pseudo:{short(name)}"] = M.per_clip_wer(p, h) if name != PSEUDO else 0.0
    wer = pd.DataFrame(wrows); wer.to_csv(os.path.join(RD, "wer_test.csv"), index=False); per.to_csv(os.path.join(RD, "wer_per_clip_test.csv"), index=False)
    print(wer.round(4).to_string(index=False))

    # ---------- 5. sentiment scores for every (classifier, text source); sources: asr full/norm, gold, pseudo
    sources = {}
    for n in asr_names:
        sources[f"{short(n)}:full"] = texts[n]; sources[f"{short(n)}:norm"] = np.array([normalize_text(x) for x in texts[n]])
    sources["gold"] = gold_text; sources["pseudo"] = texts[PSEUDO]
    keys = dt["file"].values
    scores = {}
    for cname, cm in cfg["sentiment_models"].items():
        for sname, txt in sources.items():
            cache = os.path.join(CD, f"scores_{short(cname)}__{sname.replace(':', '_')}.csv")
            scores[(cname, sname)] = S.score_texts(cname, cm["revision"], list(txt), list(keys), cache, log_path=LOG).set_index("key")
    # AADD clean fraction (A4b)
    aadd_path = os.path.join(root, cfg["aadd_files_csv"]); clean = None
    if os.path.exists(aadd_path):
        ad = pd.read_csv(aadd_path); pc = [c for c in ad.columns if c in ("path", "file")][0]
        ad["file_id"] = ad[pc].astype(str).map(lambda p: os.path.splitext(os.path.basename(p))[0])
        clean = 1.0 - dt["file"].map(ad.set_index("file_id")["degraded_fraction"]).values
        prov["inputs"]["aadd_files_csv_sha256"] = sha256_file(aadd_path)
    else: print("WARNING: AADD file CSV missing; A4b skipped")

    # ---------- 6. tune on dev; 7. evaluate on test
    thresholds, res_rows, as_rows, gap_rows, preds_long = {}, [], [], [], []
    mapped = {}   # (cname, sname) -> mapped predictions for all rows (dev+test) with its dev-tuned tau
    for cname, cm in cfg["sentiment_models"].items():
        kind = cm["kind"]
        for sname in sources:
            sc = scores[(cname, sname)]
            tau, _ = S.tune_tau(kind, sc.iloc[np.where(is_dev)[0]], y_dev, tau_grid)
            mapped[(cname, sname)] = S.map_predictions(kind, sc, tau).values
            thresholds.setdefault(f"{sname}|{short(cname)}", {})["tau"] = tau
    for n in asr_names:
        for cname, cm in cfg["sentiment_models"].items():
            cfgname = f"{n}|{cname}"; full_key = (cname, f"{short(n)}:full")
            pred = mapped[full_key]; sc = scores[full_key]["raw_score"].values
            tau = thresholds[f"{short(n)}:full|{short(cname)}"]["tau"]
            # A3 threshold on dev
            correct_dev = (pred[is_dev] == y_dev).astype(int)
            t, ba, _ = AS.tune_threshold(sc[is_dev], correct_dev, t_grid)
            # A4a second classifier: tabularisai for Cardiff configurations, Cardiff otherwise (same ASR)
            second = "tabularisai/multilingual-sentiment-analysis" if short(cname) == "cardiff" else "cardiffnlp/twitter-roberta-base-sentiment-latest"
            second_pred = mapped[(second, f"{short(n)}:full")]
            t_b = ba_b = None
            if clean is not None: t_b, ba_b, _ = AS.tune_threshold(clean[is_dev], correct_dev, t_grid)
            thresholds[f"{short(n)}:full|{short(cname)}"].update({"t": t, "t_balanced_acc_dev": ba, "t_b": t_b, "t_b_balanced_acc_dev": ba_b, "second_classifier_A4a": short(second),
                                                                    "tau_boundary_hit": bool(tau is not None and tau >= tau_grid[-1])})
            # test metrics for full / norm / gold / pseudo
            for sname in [f"{short(n)}:full", f"{short(n)}:norm", "gold", "pseudo"]:
                p_ = mapped[(cname, sname)][is_test]; row = M.classification(y_test, p_)
                lo, hi = M.bootstrap_macro_f1(y_test, p_, B, seed)
                src = sname.split(":")[-1] if ":" in sname else sname
                res_rows.append({"config": cfgname, "asr": short(n), "sentiment": short(cname), "source": src, "tau": thresholds[f"{sname}|{short(cname)}"]["tau"],
                                 "macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi, **row})
            pg = mapped[(cname, "gold")][is_test]; pp = mapped[(cname, "pseudo")][is_test]; pn = mapped[(cname, f"{short(n)}:norm")][is_test]
            fg = M.classification(y_test, pred[is_test])["macro_f1"]
            for other, oname in [(pg, "gold"), (pp, "pseudo"), (pn, "norm")]:
                lo, hi = M.paired_gap_ci(y_test, pred[is_test], other, B, seed)
                gap_rows.append({"config": cfgname, "comparison": f"full_minus_{oname}", "gap": fg - M.classification(y_test, other)["macro_f1"], "ci_lo": lo, "ci_hi": hi, "ci_includes_zero": bool(lo <= 0 <= hi)})
            # assertions on test
            a = AS.assert_all(texts[n][is_test], dur[is_test], sc[is_test], t, labels=pred[is_test], second_labels=second_pred[is_test],
                              clean_fractions=None if clean is None else clean[is_test], t_b=t_b)
            correct = (pred[is_test] == y_test)
            ar = {"config": cfgname, "asr": short(n), "sentiment": short(cname), "t": t, "t_b": t_b, "macro_f1": fg, "n": int(is_test.sum())}
            sets = {"A1": a["A1"], "A2": a["A2"], "A3": a["A3"], "A4a": a["A4a"], "A123": a["A123"], "A123_A4a": a["A123"] & a["A4a"]}
            if "A4b" in a: sets["A4b"] = a["A4b"]; sets["A123_A4b"] = a["A123"] & a["A4b"]
            for k, v in sets.items():
                ar[f"pass_{k}"] = float(v.mean()); ar[f"acc_pass_{k}"] = float(correct[v].mean()) if v.any() else np.nan; ar[f"acc_fail_{k}"] = float(correct[~v].mean()) if (~v).any() else np.nan
            as_rows.append(ar)
            pl = pd.DataFrame({"file": dt.loc[is_test, "file"].values, "label": y_test, "config": cfgname, "pred_full": pred[is_test], "score": sc[is_test],
                               "pred_norm": pn, "pred_gold": pg, "pred_pseudo": pp, "second_pred": second_pred[is_test], **{k: v for k, v in sets.items()}})
            preds_long.append(pl)
    res = pd.DataFrame(res_rows); res.to_csv(os.path.join(RD, "m2_t1_results.csv"), index=False)
    asr_df = pd.DataFrame(as_rows); asr_df.to_csv(os.path.join(RD, "assertions_test.csv"), index=False)
    pd.DataFrame(gap_rows).to_csv(os.path.join(RD, "gap_ci.csv"), index=False)
    pd.concat(preds_long).to_csv(os.path.join(RD, "predictions_test.csv"), index=False)
    write_json(thresholds, os.path.join(RD, "thresholds.json"))
    # correlations pass rate vs macro-F1 across configurations
    corr = {}
    for k in ["A123", "A123_A4a"] + (["A123_A4b"] if "pass_A123_A4b" in asr_df else []):
        corr[k] = M.rank_agreement(asr_df[f"pass_{k}"], asr_df["macro_f1"])
    write_json(corr, os.path.join(RD, "correlations.json"))
    # baseline
    base_pred = np.array(["Neutral"] * int(is_test.sum())); brow = M.classification(y_test, base_pred); lo, hi = M.bootstrap_macro_f1(y_test, base_pred, B, seed)
    write_json({"baseline": "always-neutral", **brow, "macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi, "test_label_counts": pd.Series(y_test).value_counts().to_dict()}, os.path.join(RD, "baseline.json"))
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); write_json(prov, os.path.join(RD, "provenance.json"))

    # ---------- 8. tables.md
    L = []
    L.append("# T1 tables (generated by run_t1.py)\n")
    L.append(f"Gold join: coverage {checks['coverage']}/{checks['n_clips']}; label agreement {checks['label_agreement']}/{checks['n_clips']} ({checks['label_mismatches']} mismatches); durations differing by >0.1 s: {checks['duration_abs_delta_gt_0.1s']} ({checks['duration_mismatch_fraction']:.2%}), max |Δ| {checks['duration_abs_delta_max']:.3f} s → {checks['verdict_durations']}\n")
    L.append("## WER on the test slice (1,500 clips)\n"); L.append(wer.round(4).to_markdown(index=False)); L.append("")
    L.append("## s2t loop fix\n"); L.append(pd.DataFrame(loop).T.round(3).to_markdown()); L.append("")
    L.append("## Macro-F1 on test by configuration and text source (95% bootstrap CI, 10,000 resamples, seed 8001)\n")
    piv = res.pivot_table(index=["asr", "sentiment"], columns="source", values="macro_f1").round(4); L.append(piv.to_markdown()); L.append("")
    L.append("## Full table\n"); L.append(res.round(4).to_markdown(index=False)); L.append("")
    L.append("## Full-minus-gold (and pseudo, norm) paired gap CIs\n"); L.append(pd.DataFrame(gap_rows).round(4).to_markdown(index=False)); L.append("")
    L.append("## Assertions on test\n"); L.append(asr_df.round(4).to_markdown(index=False)); L.append("")
    L.append("## Pass rate vs macro-F1 across configurations\n"); L.append(json.dumps(corr, indent=1)); L.append("")
    L.append("## Baseline\n"); L.append(json.dumps({**brow, "ci": [lo, hi]}, indent=1)); L.append("")
    L.append("## Thresholds\n"); L.append("```json\n" + json.dumps(thresholds, indent=1) + "\n```")
    with open(os.path.join(RD, "tables.md"), "w") as f: f.write("\n".join(L))
    print("DONE ->", RD)

if __name__ == "__main__":
    main()
