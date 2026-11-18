# Final audit, pass 1 (2026-11-18 05:53 CST)

Auditor: fresh-context Opus. Scope: `Milestone2/OVERNIGHT_REPORT.md` as committed at HEAD 7912db2 (file mtime 05:49:21, identical to HEAD), `OVERNIGHT_STATUS.md`, `figures/manifest.md`, tracked-file rules. No GPU used; I modified no file except this one. Phase 4's n=300 extension was still running (`run_ext.sh`, 189/300 clips) during the audit, so `results/n300/` and `results/aventiq/` are not audited.

Paths given as `results/...` in a phase section are resolved against that phase's directory. In the tables below, ✓ means the value matches the file and ✗ means it does not.

## 1. Report numbers

### Run facts
| # | Claim | File checked | Value found | Match |
|---|---|---|---|---|
| 1 | Run start 04:27 | OVERNIGHT_STATUS.md | 04:27 | ✓ |
| 2 | Push 08:20 CST | OVERNIGHT_STATUS.md | 08:20 | ✓ |
| 3 | RTX 4000 Ada, 20 GB | none named | logs/monitor.md: RTX 4000 Ada, 20475 MiB | not in a named file (true) |
| 4 | main at e2a81e44… | git log, STATUS | e2a81e4 is the parent of 016d798 | ✓ |
| 5 | brief's 10 hours | STATUS / ledger | not found in any repo file | not found |

### Phase 1
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 6 | 3,426 clips, no label disagreement | T1 gold_checks.json | 3426, 0 mismatches | ✓ |
| 7 | AventIQ WER 0.1831 | T1 wer_test.csv | 0.18312 | ✓ |
| 8 | whisper-tiny WER 0.2206 | wer_test.csv | 0.22061 | ✓ |
| 9 | "the s2t models make 18–22%" | wer_test.csv | the range holds for the Whisper models; s2t-small is 0.598 (wording leaves it out) | ✓ (wording) |
| 10 | large-v3 17.9% | wer_test.csv | 0.17892 | ✓ |
| 11 | Cardiff gold 0.637 | m2_t1_results.csv | 0.63714 | ✓ |
| 12 | Machine transcripts 0.603–0.621 | m2_t1_results.csv | full: 0.6032 / 0.6150 / 0.6211 (norm goes to 0.5978–0.6222) | ✓ (full) |
| 13 | ≤ 0.034 | m2_t1_results.csv | 0.6371 − 0.6032 = 0.0339 | ✓ |
| 14 | Baseline accuracy 0.736 | baseline.json | 0.736 | ✓ |
| 15 | Baseline macro-F1 0.283 | baseline.json | 0.28264 | ✓ |
| 16 | "every Cardiff configuration … none [beats it] on accuracy" | m2_t1_results.csv | Cardiff on gold text has accuracy **0.7393 > 0.736** (every ASR row). The claim holds only for the full/norm/pseudo sources (max 0.7213). | ✗ |
| 17 | s2t "reported 0.93 WER" | s2t_loop_analysis.json | not in that file; wer_test.csv s2t-OLD(batch8) wer_pseudo_ref 0.9285 | not in the named file (true) |
| 18 | Loops 244 → 134 of 1,500 | s2t_loop_analysis.json | 244 / 134 / n=1500 | ✓ |
| 19 | WER 0.864 → 0.598 | s2t_loop_analysis.json | not in that file; wer_test.csv 0.8639 / 0.5983 | not in the named file (true) |
| 20 | Probe on 48 dev clips | s2t_bug_probe.json | 48 | ✓ |
| 21 | 11 loops at batch 8 | s2t_bug_probe.json | 11 | ✓ |
| 22 | 5 loops at batch 1 | s2t_bug_probe.json | 5 | ✓ |
| 23 | 38 loops with no attention mask | s2t_bug_probe.json | 38 | ✓ |
| 24 | Binary models' confidence 0.99 | thresholds.json | τ = 0.99; T1_REPORT gives 0.996 / 0.989 | ✓ |
| 25 | A3 threshold 0.05, passes every clip | thresholds.json, assertions_test.csv | t = 0.05, pass_A3 = 1.0 for SiEBERT and DistilBERT | ✓ |
| 26 | "A4a … among clips it passes, accuracy 0.60–0.75" | assertions_test.csv | acc_pass_A4a (A4a alone) = **0.593–0.723**. 0.60–0.75 is acc_pass_A123_A4a (A1∧A2∧A3∧A4a) | ✗ (label) |
| 27 | "… among clips it fails, 0.10–0.74" | assertions_test.csv | acc_fail_A4a = **0.070–0.812**. 0.10–0.74 is acc_fail_A123_A4a | ✗ (label) |
| 28 | Pearson −0.24 (A1–A3) | correlations.json | −0.2384 | ✓ |
| 29 | Pearson −0.08 (+A4a) | correlations.json | −0.0832 | ✓ |
| 30 | Audit 8 of 9 passed | logs/audit_phase1.md | 8 PASS, item 9 FAIL | ✓ |
| 31 | "ninth was two wrong numbers" | audit_phase1.md | 2 wrong numbers **and** one missing §4a write-up | ✓ (incomplete) |
| 32 | 25.5% of clips | gold_checks.json | 0.2554 (875) | ✓ |
| 33 | > 0.1 s | gold_checks.json | key duration_abs_delta_gt_0.1s | ✓ |
| 34 | Span-matched WER lower for every model | wer_test_spanmatched.csv | true for all 4 | ✓ |

### Phase 2
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 35 | Five ASR variants registered | registry/ | **6** ASR entries (whisper-tiny-wsp added in f84eeba) | ✗ (stale) |
| 36 | Four sentiment variants | registry/ | 4 | ✓ |
| 37 | Moonshine WER 0.161, the lowest | T2 wer_test.csv | 0.1612, the lowest | ✓ |
| 38 | Moonshine Cardiff 0.603, the lowest | m2_t1_results.csv | 0.6027 (s2t 0.6032, wsp 0.6085) | ✓ |
| 39 | Whisper WER 0.182–0.221 | wer_test.csv | 0.1817 / 0.1831 / 0.2206 | ✓ |
| 40 | Whisper Cardiff 0.615–0.621 | m2_t1_results.csv | 0.6150 / 0.6184 / 0.6211 (wsp 0.6085 left out) | ✓ |
| 41 | Grid of 5 × 4 | T2 m2_t1_results.csv | **6 × 4** (24 configs) since f84eeba | ✗ (stale) |
| 42 | 95% CIs, all pairs overlap for Cardiff | m2_t1_results.csv | yes, including wsp | ✓ |
| 43 | AventIQ weights differ ≤ 0.0023 | AventIQ meta.json | "240/245 tensors differ by ≤2.3e-3" | ✓ |
| 44 | AventIQ results differ by 0.003 | size_vs_accuracy.csv | 0.6211 − 0.6184 = 0.0027 | ✓ |
| 45 | SiEBERT 4,073 MB | T2_REPORT.md | 4,072.68 | ✓ |
| 46 | 1,423 MB loaded | T2_REPORT.md, registry_index.csv | 1,422.87 | ✓ |

### Phase 3
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 47 | 432 rows | edit_locations.csv | 432 | ✓ |
| 48 | 0.44% | REGISTERED.txt | 0.439% | ✓ |
| 49 | 165,888 | wsp_results.csv | 165888 | ✓ |
| 50 | WER 0.2206 → 0.2152 | wsp_results.csv | 0.22061 → 0.21522 | ✓ |
| 51 | Δ −0.0054 | wsp_results.csv | −0.005395 | ✓ |
| 52 | CI [−0.0079, −0.0026] | wsp_results.csv | [−0.00794, −0.00264] | ✓ |
| 53 | Cardiff 0.6150 → 0.6086 | wsp_results.csv | 0.61498 → 0.60858 | ✓ |
| 54 | Δ −0.0064 | wsp_results.csv | −0.00640 | ✓ |
| 55 | CI [−0.0119, −0.0016] | wsp_results.csv | [−0.01190, −0.00157] | ✓ |
| 56 | WER vs large-v3 0.1988 → 0.2100 | wsp_results.csv | 0.19882 → 0.21005 | ✓ |
| 57 | 17 of 50 passes | REGISTERED.txt, forward_passes.jsonl | 17 of 50, 17 lines | ✓ |
| 58 | Cap 20 | REGISTERED.txt | 20 | ✓ |
| 59 | Null-result floor of 40 passes | wsp_results.md | 40 | ✓ |
| 60 | "3 hypothesis families" | none | not found in any repo file | not found |
| 61 | 500 → 250 / 250 | splits/manifest.json | 250 / 250 | ✓ |
| 62 | Base reproduces 0.1988 exactly | eval_base_test.csv | recomputed 0.198820 | ✓ |
| 63 | 17 lines in forward_passes.jsonl | R/logs/forward_passes.jsonl | 17 (it lives at R/logs, not WSP_Whisper/logs) | ✓ |
| 64 | Registered 05:25:49 | REGISTERED.txt | 05:25:49 | ✓ |
| 65 | Commit 303b9e7 at 05:37 | git log | 05:37:14 | ✓ |
| 66 | fp10: 18 test clips loop | wsp_results.csv | long = 18 | ✓ |
| 67 | fp10 WER 0.357 | wsp_results.csv | 0.35675 | ✓ |
| 68 | cand_E2E5 9 errors lower | REGISTERED.txt | 1302 vs 1311 | ✓ |
| 69 | "Two keep/revert decisions used holdout runs" | not named | eval_runs.jsonl has 6 holdout runs (base, E5, E2E5, E2E5E8, fp10, fp17); the count of 2 is not verified | unverified |

### Phase 4
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 70 | 50 clips, extended to 100 | summary.json, results/n100 | 50; n100 exists | ✓ |
| 71 | Focus ratio 0.991 | summary.json | 0.99090 | ✓ |
| 72 | CI [0.911, 1.061] | summary.json | [0.9110, 1.0613] | ✓ |
| 73 | n = 34 | summary.json | 34 | ✓ |
| 74 | 16 clips with no flagged frame | summary.json | n_zero_flagged 16 | ✓ |
| 75 | 16 clips flagged throughout | summary.json | n_all_flagged 16 | ✓ |
| 76 | Error rate 0.242 inside flagged segments | summary.json H2 | 0.24188 | ✓ |
| 77 | 0.258 outside | summary.json H2 | 0.25763 | ✓ |
| 78 | +0.006, CI [−0.010, 0.024] | summary.json H2 | 0.00622, [−0.00992, 0.02409] | ✓ |
| 79 | H3 +0.022 | summary.json H3 | 0.02216 | ✓ |
| 80 | "0.3 gap between classifiers" | none named | T1 m2_t1_results: Cardiff 0.615 − tabularisai 0.325 / SiEBERT 0.323 = 0.29 | not in a named file (≈ true) |
| 81 | At most 0.034 | T1 m2_t1_results | 0.0339 | ✓ |
| 82 | whisper-base WER 0.208 vs tiny 0.243 | whisper-base/summary.json, summary.json | 0.2076 / 0.2431 | ✓ |
| 83 | Cardiff accuracy 0.74 for both | both summary.json files | 0.74 / 0.74 | ✓ |
| 84 | 92% identical labels | whisper-base/summary.json | **not in that file**; README line 86 says 92%, and my recompute from the two aggregate.csv files gives 0.92 | not in the named file (true) |
| 85 | HYPOTHESES.md in 016d798 at 04:49 | git log | 04:49:48 | ✓ |
| 86 | Test unlocked 05:05 | logs/ledger.md | 05:04:57 | ✓ |
| 87 | AADD 3.1.0 | summary.json | "aadd 3.1.0 CLI" | ✓ |
| 88 | AADD "model v3.1" | summary.json, T1 aadd_devtest.json | **"model v3"** in both data files (README and T1_REPORT say v3.1; this is audit_phase4 item 7, still open) | ✗ |
| 89 | 20 ms frames | summary.json | 20 ms | ✓ |
| 90 | Flag at probability ≥ 0.5 | summary.json | ≥ 0.5 | ✓ |
| 91 | Deletion AUC 1.426 vs 1.181 | faithfulness.csv (recomputed mean) | 1.4261 / 1.1815 | ✓ |
| 92 | Difference 0.245, CI [0.187, 0.303] | faithfulness.csv, summary.json | 0.2446, [0.1870, 0.3035] | ✓ |
| 93 | 19 of 20 clips | faithfulness.csv | 19/20 auc_diff > 0 | ✓ |
| 94 | Top two decoder layers re-initialized | summary.json rule | top-2-decoder-reinit | ✓ |
| 95 | Spearman 0.916 | faithfulness.csv (n=10) | 0.9159 | ✓ |
| 96 | M1 vs input magnitude 0.78 | faithfulness.csv | 0.7845 | ✓ |
| 97 | M2 focus ratio 0.997, CI includes 1 | summary.json m2 | 0.9966, [0.925, 1.069] | ✓ |
| 98 | 3 negative clips | summary.json | Negative 3 | ✓ |
| 99 | Brief's 300 clips | n300 run | sample 300 | ✓ |
| 100 | "AADD flags 54% of all frames on this real interview audio" | none named | 54% appears only in logs/agents/W2_audioxai.md and refers to **dev** frames. On the 50-clip test sample: mean per-clip flagged fraction 0.505 (summary.json mean_frac_flagged), pooled by duration 0.484 | ✗ (scope; correct test value ≈ 50%) |
| 101 | .npz about 570 MB | du | npz 130 MB + whisper-base/npz 130 MB + n100/npz 309 MB = 569 MB (n300 adds 642 MB and is growing) | ✓ (excluding n300) |

Other observations:
- The Phase 5 section is still the placeholder "_(filled in at the end of the run)_", but commit 11369f1 is titled "overnight report Phase 5 section and answer paragraph". That commit touched only OVERNIGHT_STATUS.md.
- `results/n300/` and `results/aventiq/` are mentioned as "if present". n300 is partial (210 npz) and untracked; aventiq does not exist yet.

## 2. OVERNIGHT_STATUS.md against disk
| Line | Claim | On disk | Agrees |
|---|---|---|---|
| Phase 1 DONE | T1_REPORT.md; audit 8/9, item 9 fixed | T1_REPORT.md exists; audit_phase1.md 8 PASS / item 9 FAIL; the corrected numbers 1090/410 and 0.05–0.90 are in T1_REPORT | yes |
| Phase 2 DONE | T2_REPORT, registry/, results/, MinIO unreachable | all present; minio_attempt.json reachable=false | yes |
| Phase 3 PARTIAL | 17/50 passes, fp10 and fp17 only, no fp25/fp50; WER and F1 numbers; audit 8/9 (item 9 = git order) | patch_fp10, patch_fp17 (plus cand_* patches, untracked); numbers match wsp_results.csv; audit_phase3.md 8 PASS / item 9 FAIL literal | yes |
| Phase 4 PARTIAL | 50 clips + n100 + whisper-base on 50; audit 7/8 (item 7 AADD wording, fix in progress); n300 + AventIQ running until 07:10 | all present; audit_phase4.md 7 PASS / item 7 FAIL; run_ext.sh running (189/300); summary.json still says "model v3" | yes (fix not done yet) |
| Phase 5 IN PROGRESS | 11 figures, report drafted, final audit running | 11 figures × (png, svg, spec, script) all present; report has a placeholder Phase 5 section | yes |
| Branch | m2-overnight from e2a81e4 | `git branch` = m2-overnight; history matches | yes |

## 3. figures/manifest.md
All 11 figures exist with png, svg, spec and render script. Every data-file sha256 matches the file on disk, except the first `t2_wer_per_variant` row. Its sha (74141c65…) is stale; the 05:50 re-render row's sha e43e4927… matches the current file.

Spot checks, recomputed by me from the data files:
| Figure | Spot-check values | Result |
|---|---|---|
| t1_macro_f1_grid | AventIQ/cardiff 0.6211 [0.5926, 0.6493]; s2t/distilbert 0.4423 [0.4168, 0.4684]; tiny/siebert 0.3228; baseline 0.2826 | all ✓ |
| t1_passrate_vs_f1 | AventIQ/cardiff 0.6907 / 0.3740 / 0.6211; s2t/cardiff 0.4533 / 0.2513 / 0.6032; tiny/tabularisai 0.3320 / 0.2007 / 0.3252; r −0.24 / −0.07, −0.08 / +0.06 | all ✓ |
| t1_full_norm_gold | s2t/cardiff 0.6032 / 0.5978 / 0.6371; tiny/distilbert 0.4575 / 0.4650 / 0.4802; AventIQ/tabularisai 0.3304 / 0.3252 / 0.3389 | all ✓ |
| t2_wer_per_variant (both rows) | moonshine 0.1612 / 0.1477 / 0.1482; s2t 0.5983 / 0.6518; OLD 0.8639; large 0.1789; wsp 0.2152 / 0.1997 / 0.2100; tiny 0.2206 / 0.1988 | all ✓ |
| t2_accuracy_vs_size | AventIQ 147.12 / 0.6211 / 0.1831; base 294.78 / 0.6184 / 0.1817; moonshine 248.07 / 0.6027 [0.5741, 0.6309] / 0.1612 | all ✓ |
| wsp_results_table | base 0.2206 [0.1937, 0.2553], 0.6150, long 3 / short 6; fp10 0.3568, +0.1361 [0.0644, 0.2144], 168,960, long 18; fp17 −0.0054 [−0.0079, −0.0026], 0.6086 [0.5795, 0.6367], 165,888 | all ✓ |
| wsp_edit_locations | fp17 165,888 / 432 / 0.00833; fp10 168,960 / 440; E5 and E2net in edits_fp17.json | all ✓ |
| xai_focus_ratio_by_class | ZpPPaMazpf0_00003 1.5426 (time_stretch, correct = 1); 8fKRoUIpAJY_00012 0.2602 (degradation_class "clean"); noise mean 0.9926 (n = 15); 16 NaN | all ✓ (note: summary.json's by-class table files the 0.2602 clip under "hum") |
| xai_wer_flagged_vs_clean | f_k09R8r_cA_00004 1.0 / 0.0833 / 12; ZpPPaMazpf0_00008 0.3158 / 1.0 / 19; hW0Q6eHqN-U_00004 0.0 / 0.2703 / 36; 12 points | all ✓ |
| xai_deletion_curves | k=5: 1.0039 / 0.4207; k=25: 3.4033 / 2.6516; k=50: 4.0643 / 3.9883; AUC 1.426 / 1.181 / 0.245 | all ✓ |
| xai_example_clip | probs Neutral 0.889; S6 D2 I0 N20; 29.6% of 311 frames; peak 0.993 time_stretch at 6.06 s; IG "and" 0.148, "you're" 0.118 (sub), "I" 0.006 | all ✓ |

## 4. Tracked-file rules (`git ls-files`, 3,785 files, branch m2-overnight)
- **Files over 50 MB in the working copy:** four PipelineEvaluation/data CSVs (development_slice and its checkpoint copy, load_pool_slice, test_slice; 225–652 MB). All four are Git LFS paths (`.gitattributes`: `Milestone2/PipelineEvaluation/data/** filter=lfs`), and the committed blobs are 131–134-byte pointers. They come from main (a4fad80, an ancestor of e2a81e4). The largest real committed blob is 9.5 MB (report_whisper-base.html). **PASS**, with a caveat: git-lfs is not installed on this machine (`git lfs` is not a git command), so any `git add` of those paths would not run the LFS clean filter.
- **Weights:** the only tracked weight-like files are `Milestone2/WSP_Whisper/edits/patch_fp10.safetensors` and `patch_fp17.safetensors`, both allowed. No .bin, .pt, .ckpt, .h5, .onnx, .gguf or .npz files are tracked. The `patch_cand_*.safetensors` files and all `*.npz` files (including the growing results/n300/npz) are covered by .gitignore. **PASS**
- Untracked right now: results/n300/, run_ext.log, scripts/run_ext.sh. Modified: audioxai/paths.py and scan.py.

## 5. Numbers not found in any named file
1. RTX 4000 Ada, 20 GB: in logs/monitor.md and T1_REPORT.md, not named.
2. "The brief's 10 hours": in no repo file.
3. "3 hypothesis families" (null-result floor): in no repo file; wsp_results.md mentions only the 40 passes.
4. "Reported 0.93 WER" and "0.864 → 0.598": cited to s2t_loop_analysis.json but they are in T1 wer_test.csv (0.9285 pseudo, 0.8639 / 0.5983 gold).
5. "92% identical predicted labels": cited to whisper-base/summary.json but only in AudioXAI/README.md (my recompute: 0.92).
6. "0.3 gap between classifiers": no file named; ≈ 0.29 from T1 m2_t1_results.csv.
7. "54% of all frames": no file named; logs/agents/W2_audioxai.md says 54% of **dev** frames.
8. "Two keep/revert decisions used holdout runs": no file named; not verified.

## 6. Summary table
| Numbers checked | Matched | Mismatched | Not in named file / unverified |
|---|---|---|---|
| 101 | 87 | 7 | 7 (5 of them true elsewhere; "10 hours" and "3 families" found nowhere, and "two keep/revert" unverified) |

Mismatches and correct values:
1. #16 "none [beats the baseline] on accuracy": Cardiff on gold text has accuracy 0.7393 > 0.736. Restrict the claim to machine-transcript configurations (max 0.7213).
2. #26 A4a pass accuracy 0.60–0.75: for A4a alone it is 0.59–0.72; 0.60–0.75 is A1∧A2∧A3∧A4a.
3. #27 A4a fail accuracy 0.10–0.74: for A4a alone it is 0.07–0.81; 0.10–0.74 is A1∧A2∧A3∧A4a.
4. #35 "Five speech-to-text variants are registered": the registry now has 6 (whisper-tiny-wsp).
5. #41 "Grid of 5 × 4": T2 m2_t1_results.csv is now 6 × 4 (24 configs).
6. #88 AADD "model v3.1": summary.json and aadd_devtest.json record "model v3". Either fix the data file per audit_phase4 item 7, or write v3.
7. #100 "flags 54% of all frames": 54% is the dev-frame figure; on the 50 test clips the mean per-clip flagged fraction is 0.505 (pooled 0.484).

Non-numeric issues:
- The Phase 5 section is still a placeholder, even though commit 11369f1's message says it was filled.
- The first t2_wer_per_variant manifest row carries a stale sha (74141c65…).
- The Phase 1 audit summary leaves out the missing §4a write-up.

# Final audit, pass 2 (2026-11-18 06:30 CST)

Auditor: fresh-context Opus, second pass. Scope: `Milestone2/OVERNIGHT_REPORT.md` at HEAD e87f0b8 (the working copy is identical to HEAD and `git status` is clean), `OVERNIGHT_STATUS.md`, `figures/manifest.md`, and the tracked-file rules. I used no GPU and modified no file except this one. The Phase 4 extension had finished (ledger 06:27:02), so `results/n300/` and `results/aventiq/` are audited this time. Paths written as `results/...` resolve against the phase directory. ✓ = matches the file, ✗ = does not, NIF = not in the named file.

## 1. Report numbers

### Run facts
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 1 | Start 04:27 | OVERNIGHT_STATUS.md | 04:27 | ✓ |
| 2 | Push 08:20 CST | OVERNIGHT_STATUS.md | 08:20 | ✓ |
| 3 | RTX 4000 Ada, 20 GB | STATUS / ledger named; neither has it | logs/monitor.md: RTX 4000 Ada, 20475 MiB | NIF (true) |
| 4 | main e2a81e44… | ledger.md preflight, STATUS | e2a81e44736f… | ✓ |
| 5 | brief's 10 hours | brief.md (launch folder) | ../brief.md l.150 "10 hours from launch" | ✓ |

### Phase 1
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 6 | 3,426 clips, no label disagreement | gold_checks.json | 3426, 0 mismatches | ✓ |
| 7 | Whisper-family 18–22% | wer_test.csv | 0.1831 / 0.2206 | ✓ |
| 8 | AventIQ 0.1831 | wer_test.csv | 0.18312 | ✓ |
| 9 | whisper-tiny 0.2206 | wer_test.csv | 0.22061 | ✓ |
| 10 | s2t 0.598 | wer_test.csv | 0.59832 | ✓ |
| 11 | large-v3 17.9% | wer_test.csv | 0.17892 | ✓ |
| 12 | Cardiff gold 0.637 | m2_t1_results.csv | 0.63714 | ✓ |
| 13 | 0.603–0.621 from machine transcripts | m2_t1_results.csv | full source 0.6032–0.6211 (norm 0.5978–0.6222, pseudo 0.6066) | ✓ (full) |
| 14 | At most 0.034 | m2_t1_results.csv | 0.6371 − 0.6032 = 0.0339 | ✓ |
| 15 | Baseline accuracy 0.736 | baseline.json | 0.736 | ✓ |
| 16 | Baseline macro-F1 0.283 | baseline.json | 0.28264 | ✓ |
| 17 | "none on accuracy (**best 0.718**)" | m2_t1_results.csv | Cardiff accuracy on machine transcripts: full 0.7167 / 0.718 / 0.714; norm 0.7153 / 0.7127 / **0.7213** (whisper-tiny norm); pseudo 0.7013. The best is **0.721**; 0.718 is the best on `full` only. "None beats 0.736" still holds. | ✗ |
| 18 | Gold text 0.739 | m2_t1_results.csv | 0.73933 | ✓ |
| 19 | s2t reported 0.93 WER, row s2t-small-OLD(batch8) | wer_test.csv | wer_pseudo_ref 0.92853 | ✓ |
| 20 | Loops 244 → 134 of 1,500 | s2t_loop_analysis.json | 244 / 134 / n 1500 | ✓ |
| 21 | 0.864 → 0.598 | wer_test.csv | 0.86389 → 0.59832 | ✓ |
| 22 | 48 dev clips | s2t_bug_probe.json | 48 | ✓ |
| 23 | 11 loops at batch 8 | s2t_bug_probe.json | 11 | ✓ |
| 24 | 5 at batch 1 | s2t_bug_probe.json | 5 | ✓ |
| 25 | 38 with no mask | s2t_bug_probe.json | 38 | ✓ |
| 26 | Binary confidence 0.99 | thresholds.json | tau 0.99 | ✓ |
| 27 | A3 threshold 0.05 passes every clip | thresholds.json, assertions_test.csv | t 0.05; pass_A3 1.0 (6 rows) | ✓ |
| 28 | Joint pass accuracy 0.60–0.75 | assertions_test.csv acc_pass_A123_A4a | 0.5989–0.7480 | ✓ |
| 29 | Joint fail 0.10–0.74 | acc_fail_A123_A4a | 0.0984–0.7401 | ✓ |
| 30 | A4a alone pass 0.59–0.72 | acc_pass_A4a | 0.5933–0.7225 | ✓ |
| 31 | A4a alone fail 0.07–0.81 | acc_fail_A4a | 0.0705–0.8120 | ✓ |
| 32 | Pearson −0.24 | correlations.json | −0.2384 | ✓ |
| 33 | Pearson −0.08 | correlations.json | −0.0832 | ✓ |
| 34 | Audit 8 of 9 | logs/audit_phase1.md | 8 PASS, item 9 FAIL | ✓ |
| 35 | 25.5% of clips | gold_checks.json | 0.2554 (875) | ✓ |
| 36 | > 0.1 s | gold_checks.json | key duration_abs_delta_gt_0.1s | ✓ |
| 37 | Span-matched WER lower for every model | wer_test_spanmatched.csv | 4 of 4 | ✓ |

### Phase 2
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 38 | Six speech-to-text variants (5 models + wsp) | registry/asr | 6 directories | ✓ |
| 39 | Four sentiment variants | registry/sentiment | 4 | ✓ |
| 40 | Moonshine WER 0.161, the lowest | T2 wer_test.csv | 0.16125, the lowest | ✓ |
| 41 | Moonshine Cardiff 0.603, the lowest | m2_t1_results.csv | 0.6027 (s2t 0.6032) | ✓ |
| 42 | Whisper variants WER 0.182–0.221 | wer_test.csv | base 0.1817, AventIQ 0.1831, wsp 0.2152, tiny 0.2206 | ✓ |
| 43 | Whisper variants give Cardiff **0.615–0.621** | m2_t1_results.csv | tiny 0.6150, base 0.6184, AventIQ 0.6211, **wsp 0.6086**. The same paragraph now counts wsp among the six variants, so the range is **0.609–0.621**, or the sentence should say it excludes wsp. | ✗ (scope) |
| 44 | Grid 6 × 4 = 24 | T2 m2_t1_results.csv | 6 ASR × 4 sentiment (96 rows over 4 sources) | ✓ |
| 45 | Every pair of ASR CIs overlaps for Cardiff | m2_t1_results.csv | max lo 0.5926 < min hi 0.6308 | ✓ |
| 46 | AventIQ weights ≤ 0.0023 | AventIQ meta.json | "240/245 tensors differ by ≤2.3e-3" | ✓ |
| 47 | Results differ by 0.003 | AventIQ meta.json | That file holds only AventIQ's 0.6211. The difference needs whisper-base (0.6184, its meta / size_vs_accuracy.csv): 0.0027 | NIF (true) |
| 48 | SiEBERT 4,073 MB | T2_REPORT.md | 4,072.68 | ✓ |
| 49 | 1,423 MB loaded | T2_REPORT.md | 1,422.87 | ✓ |

### Phase 3
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 50 | 432 rows | edit_locations.csv | n_rows_changed 432 | ✓ |
| 51 | 0.44% | REGISTERED.txt | 0.439% | ✓ |
| 52 | 165,888 | wsp_results.csv | 165888 | ✓ |
| 53 | WER 0.2206 → 0.2152 | wsp_results.csv | 0.22061 → 0.21522 | ✓ |
| 54 | Δ −0.0054 | wsp_results.csv | −0.005395 | ✓ |
| 55 | CI [−0.0079, −0.0026] | wsp_results.csv | [−0.00794, −0.00264] | ✓ |
| 56 | Cardiff 0.6150 → 0.6086 | wsp_results.csv | 0.61498 → 0.60858 | ✓ |
| 57 | Δ −0.0064 | wsp_results.csv | −0.00640 | ✓ |
| 58 | CI [−0.0119, −0.0016] | wsp_results.csv | [−0.01190, −0.00157] | ✓ |
| 59 | vs large-v3 0.1988 → 0.2100 | wsp_results.csv | 0.19882 → 0.21005 | ✓ |
| 60 | 17 of 50 passes | REGISTERED.txt | 17 of 50 | ✓ |
| 61 | Cap 20 | REGISTERED.txt | 20 | ✓ |
| 62 | Floor 40 passes and 3 families, brief.md §6 | ../brief.md | l.280 (inside §6, which starts at l.241): "at least 40 forward passes and testing 3 distinct hypothesis families" | ✓ |
| 63 | 500 → 250 / 250 | splits/manifest.json | 250 / 250 | ✓ |
| 64 | Base reproduces 0.1988 exactly | eval_base_test.csv | recomputed 0.198820 | ✓ |
| 65 | 17 lines | logs/forward_passes.jsonl (at R/logs) | 17 | ✓ |
| 66 | Registered 05:25:49 | REGISTERED.txt | 05:25:49 | ✓ |
| 67 | 303b9e7 at 05:37 | git log | 05:37:14 | ✓ |
| 68 | fp10: 18 test clips loop | wsp_results.csv | long 18 | ✓ |
| 69 | fp10 WER 0.357 | wsp_results.csv | 0.35675 | ✓ |
| 70 | Two keep/revert decisions (E1, E8) used holdout | logs/agents/W1_wsp.md, status 05:20 | l.33 "holdout used for 2 keep/revert decisions (E1, E8)" | ✓ |
| 71 | cand_E2E5 9 errors lower | REGISTERED.txt | 1302 vs 1311 | ✓ |

### Phase 4, headline (50 clips)
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 72 | 50 clips, extended to 100 | summary.json, n100/summary.json | 50, 100 | ✓ |
| 73 | Focus ratio 0.991 | summary.json H1 | 0.99090 | ✓ |
| 74 | CI [0.911, 1.061] | summary.json | [0.9110, 1.0613] | ✓ |
| 75 | n = 34 | summary.json | 34 | ✓ |
| 76 | 16 with no flagged frame | summary.json | n_zero_flagged 16 | ✓ |
| 77 | 16 flagged throughout | summary.json | n_all_flagged 16 | ✓ |
| 78 | 0.242 inside flagged | summary.json H2 | 0.24188 | ✓ |
| 79 | 0.258 outside | summary.json H2 | 0.25763 | ✓ |
| 80 | +0.006, CI [−0.010, 0.024] | summary.json H2 | 0.00622 [−0.00992, 0.02409] | ✓ |
| 81 | H3 +0.022 | summary.json H3 | 0.02216 | ✓ |
| 82 | "gap of **about 0.29** between Cardiff and **the binary classifiers** on the same transcripts" | T1 m2_t1_results.csv | whisper-tiny/full: Cardiff 0.6150 vs SiEBERT 0.3228 (gap **0.29**) but vs DistilBERT 0.4575 (gap **0.16**). Across T1 ASR the gap to SiEBERT is 0.28–0.29 and to DistilBERT 0.16. The 0.29 also matches tabularisai (0.325), which is not binary. | ✗ |
| 83 | At most 0.034 | T1 m2_t1_results.csv | 0.0339 | ✓ |
| 84 | whisper-base WER 0.208 vs 0.243 | whisper-base/summary.json, summary.json | 0.2076 / 0.2431 | ✓ |
| 85 | Cardiff accuracy 0.74 for both | both summary.json | 0.74 / 0.74 | ✓ |
| 86 | 92% identical labels | AudioXAI/README.md (l.86), from the two aggregate.csv files | README 92%; recomputed 46/50 = 0.92 | ✓ |
| 87 | HYPOTHESES.md in 016d798 at 04:49 | git log | 04:49:48 | ✓ |
| 88 | Test unlocked 05:05 | ledger.md | 05:04:57 | ✓ |
| 89 | AADD 3.1.0 | summary.json | "aadd 3.1.0 CLI" | ✓ |
| 90 | Registry model "v3", described as v3.1 | summary.json | "model v3 (v3.1)"; README l.41 "model v3.1" | ✓ |
| 91 | 20 ms frames, `--resolution 0.02` | summary.json | "--resolution 0.02 (20 ms)" | ✓ |
| 92 | Flag at p ≥ 0.5 | summary.json flag_rule_frame | ≥ 0.5 | ✓ |
| 93 | Deletion AUC 1.426 vs 1.181 | faithfulness.csv (recomputed) | 1.4261 / 1.1815 | ✓ |
| 94 | Diff 0.245, CI [0.187, 0.303] | faithfulness.csv, summary.json deletion | 0.2446 [0.1870, 0.3035] | ✓ |
| 95 | 19 of 20 clips | faithfulness.csv | 19 with auc_diff > 0 | ✓ |
| 96 | Top two decoder layers | summary.json randomization rule | top-2 decoder re-init | ✓ |
| 97 | Spearman 0.916 | faithfulness.csv | 0.9159 (n 10) | ✓ |
| 98 | M1 vs input magnitude 0.78 | faithfulness.csv | 0.7845 | ✓ |
| 99 | M2 0.997, CI includes 1 | summary.json m2 | 0.9966 [0.9252, 1.0686] | ✓ |

### Phase 4, extension (new in this pass)
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 100 | Full 300-clip sample | n300/summary.json | n_clips 300 | ✓ |
| 101 | H1 0.994, CI [0.966, 1.021] | n300/summary.json H1 | 0.99365 [0.96614, 1.02081] | ✓ |
| 102 | n = 214 | n300/summary.json | 214 (= 300 − 86 zero-flag) | ✓ |
| 103 | Only music (1.060) and hum (0.837) have intervals that exclude 1 | n300 focus_ratio_m1_by_class | music 1.060 [1.007, 1.123]; hum 0.837 [0.670, 0.988]; codec, highpass, noise, reverb and time_stretch include 1. Note: clipping has n = 1 and a degenerate interval [1.030, 1.030], which literally excludes 1. | ✓ (note) |
| 104 | Paired diff −0.126 [−0.275, −0.017] | n300 H2 paired_wer_diff | −0.12570 [−0.27522, −0.01652] | ✓ |
| 105 | 85 clips with both regions | n300 H2 | n 85 | ✓ |
| 106 | Pooled 0.343 vs 0.223 | n300 H2 | 0.34336 / 0.22325 | ✓ |
| 107 | +0.020 [0.010, 0.032] | n300 H2 err_imp_share_minus_word_share | 0.02044 [0.00988, 0.03153] | ✓ |
| 108 | AventIQ WER 0.210 | aventiq/summary.json | 0.21042 | ✓ |
| 109 | AventIQ Cardiff 0.78, equal to gold-text | aventiq/summary.json | 0.78 / 0.78 | ✓ |
| 110 | 96% labels identical to tiny's | README l.114 | 96%; recomputed 48/50 = 0.96 | ✓ |
| 111 | AventIQ H1 0.991 [0.913, 1.060], H2 refuted | aventiq/summary.json | 0.99122 [0.91284, 1.05991]; H2 "refuted" | ✓ |
| 112 | M2 vs M1 Spearman 0.167 [0.147, 0.186] | n100/summary.json m2 | 0.16710 [0.14724, 0.18616] | ✓ |
| 113 | M2 focus 0.989 [0.917, 1.069] | n100/summary.json m2 | 0.98905 [0.91696, 1.06931] | ✓ |
| 114 | Deletion beats random on every run; 300-clip diff 0.201 [0.146, 0.251] | all 5 summary.json | n300 0.2006 [0.1461, 0.2514]; every run's CI lies above 0 (n100 0.246, base 0.183, AventIQ 0.167) | ✓ |
| 115 | Randomization fails on every run, Spearman 0.909–0.931 | all 5 summary.json | 0.9159 / 0.9089 / 0.9309 / 0.9176 / 0.9305, all FAIL | ✓ |

### Phase 4, limitations
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 116 | 50 clips, 3 negative | summary.json label_counts | Negative 3 | ✓ |
| 117 | 300 tiny, 50 base and AventIQ, M2 on 100, segment replacement on 10 | n300, whisper-base, aventiq, n100 summaries; results/e2e_segments.csv | 300 / 50 / 50 / m2 n 100 / 10 rows | ✓ |
| 118 | Mean flagged fraction 0.505 | summary.json mean_frac_flagged | 0.50476 | ✓ |
| 119 | ".npz, **about 570 MB**" | du | Stale after the extension: npz 136 MB + whisper-base 136 + n100 327 + **n300 1,149** + **aventiq 138** = **1,885 MB (≈1.9 GB)**, or 1,916 MB with dev/npz (31 MB). None of it is tracked. | ✗ |

### Phase 5
| # | Claim | File | Value found | Match |
|---|---|---|---|---|
| 120 | 11 figures | figures/manifest.md, figures/ | 11 names, each with png + svg + spec + script | ✓ |
| 121 | Phase 1 audit 8 of 9 | audit_phase1.md | 8/9 | ✓ |
| 122 | Phase 3 audit 8 of 9 | audit_phase3.md | 8/9 | ✓ |
| 123 | Registration written 13 s before that run | audit_phase3.md item 9 | registered_at 05:25:49, gpu_jobs start 05:26:02 | ✓ |
| 124 | Commit 11 minutes after it | audit_phase3.md | "11 min after it started" | ✓ |
| 125 | Phase 4 audit 7 of 8, fixed afterwards | audit_phase4.md; README l.41, summary.json aadd | 7/8; v3.1 wording now present | ✓ |
| 126 | Pass 1: 7 mismatches among 101 | audit_final.md pass 1 | 101 / 7 | ✓ |
| 127 | Monitor loop every 15 minutes | ledger.md 04:34:20, monitor.md | every 15 min | ✓ |
| 128 | 17 WSP passes | forward_passes.jsonl | 17 lines | ✓ |
| 129 | Phase 1 finished 05:02 | ledger.md | 05:02:46 "Phase 1 DONE at 05:02" | ✓ |
| 130 | Phase 2 finished **05:03** | ledger.md | 05:04:57 "Phase 2 DONE at **05:04**" (commit 98e2cdf at 05:04:57). No Phase-2 event at 05:03. | ✗ |
| 131 | Two workers at 05:41 | ledger.md | W1 DONE 05:41; W2 DONE 05:40 (its commit 71d3163 is at 05:41:45) | ✓ (W2 is 05:40) |
| 132 | About two hours ahead of the compressed plan | logs/oracle_plan.md l.4 | Phase 2 planned by 07:05 (done 05:04); P3/P4 until 07:40 (done 05:41) | ✓ |
| 133 | Gold text adds only 0.02–0.03 (Phase 1) | m2_t1_results.csv | gold − full for Cardiff: AventIQ 0.016, tiny 0.022, s2t 0.034 | ✓ (rounded) |
| 134 | Switching classifiers moves it by about 0.3 (Phase 1) | Phase 1 section; no file | The Phase 1 section gives no 0.3. From m2_t1_results.csv the largest gap is 0.29 (Cardiff vs SiEBERT or tabularisai); the gap to DistilBERT is 0.16 | NIF (true as the maximum) |

Other observations:
- The wording of the pass-1 fixes is correct: #17's companion clause (gold 0.739), the A4a ranges (#28–31), the six variants (#38) and 6 × 4 (#44) now match. #17 adds a new parenthetical "best 0.718", and it is wrong.
- The Phase 5 header says the final commit hash is in `OVERNIGHT_STATUS.md`. STATUS does not have it yet, which is expected before the push.
- AventIQ's H3 verdict in aventiq/summary.json is "holds" (+0.016 [−0.001, 0.033]). The report does not mention it. That is consistent with ledger 05:13 ("not refuted for AventIQ") and is not a mismatch.

## 2. OVERNIGHT_STATUS.md against disk
| Line | On disk | Agrees |
|---|---|---|
| VERDICT RUNNING, 04:27, 08:20 | run in progress, final audit running | yes |
| Phase 1 DONE, audit 8/9, item 9 fixed | T1_REPORT.md; audit_phase1.md 8 PASS / item 9 FAIL | yes |
| Phase 2 DONE, MinIO unreachable | T2_REPORT.md, registry/ (6 ASR, 4 sentiment), results/; minio_attempt.json reachable=false | yes |
| Phase 3 PARTIAL: 17/50 passes, cap 20, fp10 and fp17 only, floor 40 not reached; 0.2206→0.2152, 0.6150→0.6086, CIs exclude 0; audit 8/9 | wsp_results.csv and REGISTERED.txt match; only patch_fp10/fp17 among fp* | yes |
| Phase 4 PARTIAL: 50 headline; n300, 50 base and AventIQ, M2 on 100, segment replacement on 10; audit 7/8, item 7 fixed | n300/ (300), whisper-base/ (50), aventiq/ (50), n100 m2 (100), e2e_segments.csv (10); README l.41 and summary.json carry v3.1 and --resolution 0.02 | yes |
| Phase 5 IN PROGRESS: 11 figures, report drafted, final audit running | 11 figures present; report Phase 5 section written; this audit | yes |
| Branch m2-overnight from e2a81e4 | git log e2a81e4..HEAD = 20 commits | yes |

## 3. figures/manifest.md
All 11 figures exist (png, svg, spec JSON and render script each). Every data-file sha256 in the manifest equals both the file on disk and the hash in its spec. The one exception is the first `t2_wer_per_variant` row (74141c65…), which is the pre-wsp hash by design, as the orchestrator's 05:55 note says. The re-render row's e43e4927… matches. The data files are unchanged since pass 1. I recomputed these spot checks:
| Figure | Values | Result |
|---|---|---|
| t1_macro_f1_grid | 0.6211 [0.5926, 0.6493]; 0.4423; 0.3228; baseline 0.2826 | ✓ |
| t1_passrate_vs_f1 | sha d7fcdf4c… matches; Pearson −0.24 / −0.08 and Spearman −0.07 / +0.06 equal correlations.json (−0.0668, +0.0559) | ✓ |
| t1_full_norm_gold | s2t/cardiff 0.6032 / 0.5978 / 0.6371 | ✓ |
| t2_wer_per_variant (both rows) | moonshine 0.1612 / 0.1477 / 0.1482; wsp 0.2152 / 0.1997 / 0.2100; tiny 0.2206 / 0.1988; large 0.1789; OLD 0.8639 | ✓ |
| t2_accuracy_vs_size | 147.12 / 0.6211 / 0.1831; 294.78 / 0.6184; 248.07 / 0.6027 | ✓ |
| wsp_results_table | base 0.2206, long 3 / short 6; fp10 long 18, 168,960; fp17 −0.0054 [−0.0079, −0.0026] | ✓ |
| wsp_edit_locations | 165,888 / 432 / 0.00833; fp10 168,960 / 440 | ✓ |
| xai_focus_ratio_by_class | ZpPPaMazpf0_00003 time_stretch, correct 1, 1.54255; 8fKRoUIpAJY_00012 clean 0.2602; noise mean 0.9926 (n 15); 34 valid / 16 NaN | ✓ |
| xai_wer_flagged_vs_clean | f_k09R8r 1.0 / 0.0833 / 12; ZpPPaMazpf0_00008 0.3158 / 1.0 / 19; hW0Q6eHqN-U 0.0 / 0.2703 / 36; 12 clips with both rates | ✓ |
| xai_deletion_curves | k5 1.0039 / 0.4207; k25 3.4033 / 2.6516; k50 4.0643 / 3.9883; AUC 1.4261 / 1.1815 / 0.2446 | ✓ |
| xai_example_clip | npz sha 00221b2b… matches; frac_flagged 0.2958 | ✓ (sha and aggregate row; npz internals not re-read) |

## 4. Tracked-file rules (`git ls-files`, 3,804 files, HEAD e87f0b8)
- Over 50 MB: in the working copy, the four PipelineEvaluation/data CSVs (225–652 MB) are Git LFS paths, as in pass 1. In `git ls-tree -r -l HEAD` no blob exceeds 50 MB. The largest committed blobs are the three report HTML files (9.16, 9.49 and 9.50 MB; `results/aventiq/report_aventiq.html` is new). **PASS**
- Weights: the only tracked weight files are `Milestone2/WSP_Whisper/edits/patch_fp10.safetensors` and `patch_fp17.safetensors`. No .bin, .pt, .pth, .ckpt, .h5, .onnx, .gguf, .msgpack or .npz is tracked. The new n300/ and aventiq/ entries are CSV, JSONL, JSON and HTML only. **PASS**
- `git status`: clean. The run_ext logs and script that were untracked in pass 1 are now either committed or ignored.

## 5. Summary
| Numbers checked | Matched | Mismatched | Not in named file |
|---|---|---|---|
| 134 | 126 | 5 | 3 (all true elsewhere: #3 GPU, #47 0.003, #134 "about 0.3") |

Mismatches and correct values:
1. #17 Phase 1, "none on accuracy (best 0.718)": the best Cardiff accuracy on machine transcripts is **0.721** (whisper-tiny, normalized text). 0.718 is the best on full transcripts (s2t). The claim that none beats 0.736 holds.
2. #43 Phase 2, "the Whisper variants … give 0.615–0.621": whisper-tiny-wsp, now counted among the six variants, gives **0.6086**. Either write **0.609–0.621**, or say "the unedited Whisper variants".
3. #82 Phase 4, "a gap of about 0.29 between Cardiff and the binary classifiers": that holds for SiEBERT (0.615 − 0.323 = 0.29) but not for DistilBERT (0.615 − 0.457 = **0.16**). Write "between Cardiff and SiEBERT", or "0.16–0.29 to the binary classifiers".
4. #119 Phase 4 limitations, ".npz, about 570 MB": after the extension the uncommitted .npz arrays total **≈1.9 GB** (1,885 MB across the 50-clip, whisper-base, n100, n300 and aventiq runs; 1,916 MB including dev).
5. #130 Phase 5, "Phases 1 and 2 finished at 05:02 and 05:03": Phase 2 finished at **05:04** (ledger 05:04:57 "Phase 2 DONE at 05:04"; commit 98e2cdf 05:04:57).

Minor, not counted as mismatches: W2 finished at 05:40, not 05:41 (#131); clipping's degenerate n = 1 interval (#103); the npz internals of the example clip were not re-read.
