# Phase 4 audit (fresh-context auditor): AudioXAI
Run 2026-11-18 05:30–05:47 CST against commit 71d3163. AudioXAI is unchanged from 71d3163 to HEAD f84eeba; the working tree has uncommitted edits to `paths.py`/`scan.py` (AventIQ rev, AUDIOXAI_NOTE) and an untracked `results/n300`, and none of these touch the audited files. I wrote my own scripts (scratchpad `a4/i2.py i3.py i4.py i5.py i6.py`) and ran them with `.venv/bin/python`. Project code is imported only for item 3, as the brief requires (`core.m1_smoothgrad`, `core.teacher_forced_logp`, `core.features`, `core.transcribe`, `core.load_audio`). X = `Milestone2/AudioXAI`. No load-pool audio was read. The GPU was used only for item 3, under `flock /tmp/m2_gpu.lock`, for about 4 s of compute (well under 5 min). Nothing was written to gpu_jobs.jsonl, so that no other file was modified.

Known deviation, recorded in ledger.md at 05:04:57: TEST_UNLOCKED was created after Phases 1 and 2 were committed (98e2cdf), not after Phases 1 to 3. I note it here and do not count it as a failure.

## Item 1 — PASS (with a note)
Cmd: `git log --diff-filter=A` on HYPOTHESES.md, `stat` on TEST_UNLOCKED, and grep of gpu_jobs.jsonl for w2_test jobs.
- HYPOTHESES.md was added in 016d798 at **04:49:48**. TEST_UNLOCKED was born at **05:04:57.49**. The first W2 test job is `w2_test_scan_a`, which started at **05:06:03** (line 93). Before that line, every w2 job is dev or smoke (w2_aadd_smoke, w2_smoke_explain, w2_dev_scan_gpu, w2_dev_m2, w2_dev_e2e). The order is commit < unlock < first test job.
- Note: the **Addendum** (informative-clip secondary statistic, graded Spearman, zero/all-flag counts) is not in 016d798. It first reached git in 71d3163 at 05:41:45, after the test results. Evidence that it predates the test run: the file mtime is 04:56:45, and W2_audioxai.md line 19 records "HYPOTHESES.md sha256 6045a4492cbb8a4b frozen before first test job". The current file and the 71d3163 blob both hash to `6045a4492cbb8a4b…`. The addendum leaves the primary H1 rule unchanged. Its pre-test status rests on mtime plus a self-reported hash, not on a commit.
Verdict: the pre-registration holds for the primary hypotheses. The addendum is pre-test only by local evidence.

## Item 2 — PASS
Cmd: my own allocation. test.csv sorted by original_index, labels alphabetical, quota = n·share, floor with a ≥ 3 per class minimum, largest remainder, one `default_rng(8001)` with `permutation` within each label.
- n=50: quotas are Neg 2.47 / Neu 36.80 / Pos 10.73, giving an allocation of 3/37/10. This is identical, in order, to `results/test_sample_ids.csv` (50 rows). `whisper-base/test_sample_ids.csv` is identical too.
- n=100: allocation 5/74/21, identical to `n100/test_sample_ids.csv`. Within each label the 50-sample is an ordered prefix of the 100-sample, as the README claims.
- Note: the ≥ 3 floor moves one Positive slot to Negative. Pure largest-remainder would give 2/37/11. The floor is pre-registered in HYPOTHESES.md and the README.
Verdict: the sample reproduces exactly.

## Item 3 — PASS (tolerance is limited by float16 storage)
Cmd: `default_rng(8001).choice(50,5,replace=False)` gives sample rows 12, 22, 26, 35, 45. For each clip I chose one random token (same rng), loaded the audio through X's guard, recomputed the features, re-decoded, and called X's `m1_smoothgrad(n=8, seed default 8001)` for that token. I also ran a plain grad×input without noise. For the token's argmax M2 patch, I occluded it with the clip log-mel minimum and recomputed the teacher-forced log-prob drop.

| clip | token | seq reproduced | lp0 saved = new | M1 rel L2 | fp16 rounding floor | plain g×i Pearson / sign agree (top 10%) / scale | M2 saved → new (rel) |
|---|---|---|---|---|---|---|---|
| id10271_zWyD72sHVwA_00007 | 12/13 | yes | yes | 2.17e-4 | 2.17e-4 | 0.958 / 0.984 / 0.87 | 0.98438 → 0.98416 (2.1e-4) |
| id10295_ywZWrIt52KQ_00003 | 14/19 | yes | yes | 2.66e-4 | 2.66e-4 | 0.959 / 0.994 / 1.30 | 0.20911 → 0.20908 (1.2e-4) |
| id10298_hjvQiiG71rM_00021 | 0/12 | yes | yes | 2.15e-4 | 2.15e-4 | 0.982 / 0.997 / 0.98 | 2.20313 → 2.20342 (1.3e-4) |
| id10291_iPZGPcrYIZs_00004 | 1/14 | yes | yes | 2.26e-4 | 2.26e-4 | 0.995 / 1.000 / 0.97 | 0.17725 → 0.17719 (3.0e-4) |
| id10283_vaK4t1-WD4M_00020 | 18/37 | yes | yes | 2.50e-4 | 2.50e-4 | 0.911 / 0.965 / 0.98 | 0.24634 → 0.24631 (1.3e-4) |

- The M1 relative error equals ‖a − fp16(a)‖/‖a‖ to 3 significant figures. The saved map is therefore exactly the float16 rounding of a fresh recompute. The 1e-4 relative tolerance in the brief is below float16 resolution (eps 9.8e-4), so nothing stored in fp16 can meet it literally. Against the fp16 floor the match is exact.
- M2 values are stored as fp16 (for example 0.984375). The 1.2–3.0e-4 differences are within fp16 half-ulp.
- Sanity check without noise: sign and scale match SmoothGrad (Pearson 0.91–0.995, and the saved/plain |·| ratio is 0.87–1.30). The sign convention is gradient of log p × input.
Verdict: M1 and M2 reproduce to storage precision.

## Item 4 — PASS
Cmd: recomputed the summary statistics from `results/faithfulness.csv` (B = 10,000, `default_rng(8001)`), recomputed each AUC from the saved drop curves with `np.trapezoid`, and parsed the `const D=` JSON embedded in report.html.
- Deletion: n = 20, AUC from curves matches the stored values to 4e-16. Top 1.4261 [1.2621, 1.6077], random 1.1815 [1.0256, 1.3456], diff 0.2446 [0.1870, 0.3035], top > random in 19/20. These match summary.json exactly.
- Randomization: n = 10, mean ρ(|M1|) 0.9159 [0.9026, 0.9276]. Per-clip pass_check is 0/10, so the verdict is **FAIL**. Signed ρ 0.092, |M1| vs |input| 0.785. These match summary.json. The failure is **reported** in summary.json, the README ("**FAIL**") and the report's Faithfulness card, which renders `S.randomization.verdict`.
- report.html: the embedded summary is identical to summary.json (0 differing leaves). The embedded faith rows equal faithfulness.csv rounded to 4 dp (max diff 0.0). whisper-base has the same structure: deletion diff 0.1834 [0.1259, 0.2412], randomization 0.9305, FAIL, all matching. M2 (n = 50): median ρ 0.162 [0.1356, 0.1886], matching.
Verdict: both checks ran, and their results, including the FAIL, are reported consistently.

## Item 5 — PASS
Cmd: from `aggregate.csv`, took clips with n_flagged_frames ≥ 1, computed the mean of focus_ratio_m1, and ran my own vectorised bootstrap (`default_rng(8001).integers(0,n,(10000,n))`, percentile).
- n=50: zero-flag clips 16 (stored 16). n = 34, mean 0.990904, CI [0.911035, 1.061254]. These match summary.json to every printed digit. Under the pre-registered rule (CI includes 1) the verdict is "refuted". Informative subset n = 18: 0.9828 [0.8323, 1.1132], matching. All-flagged 16 (stored 16). M3 0.9724 [0.9229, 1.0115] and M4 0.9671 [0.8732, 1.0537] match.
- n100: zero-flag 29, n = 71, 0.9951 [0.9390, 1.0490], matching. whisper-base: 16, n = 34, 0.9908 [0.9118, 1.0586], matching.
- Independent check of the per-clip focus ratio from the npz arrays (|M1| summed over mel, 10→20 ms, token-normalised, mean over tokens; any-class p ≥ 0.5): max |diff| vs aggregate.csv is 5.9e-4 over the 34 clips, which is fp16 storage. The NaN pattern agrees on all 50.
Verdict: exact match.

## Item 6 — PASS on numbers (stale text noted)
Cmd: parsed the embedded `table` and drew 8 cells with `default_rng(8001)` across 8 columns. Compared with aggregate.csv rounded to 3 dp.
- 8/8 match in both report.html and report_whisper-base.html. Examples: id10303_Zs8VK91yVMI_00009 focus_ratio_m1 1.0311 → 1.031, and id10293_k1jiAINXHRI_00007 frac_flagged 0.99769 → 0.998. Table rows 50 = aggregate rows 50, and 50 clips are embedded. All rendered summary numbers come from the identical embedded summary (item 4). For example, H1 renders 0.991 [0.911, 1.061] and deletion diff 0.245 [0.187, 0.303].
- README numbers spot-checked against summary.json all match: H2 0.242/0.258 (554/590), 13.0% vs 12.4%, +0.006 [−0.010, 0.024], sentiment share 0.50 [0.38, 0.63], graded ρ 0.033 [−0.027, 0.094], point-biserial −0.08 [−0.25, 0.13], e2e 0.054 / 0.126, H3 0.615 → 0.637, +0.022 [0.003, 0.042].
- Stale text (not numbers):
  - (a) The report.html footer says "SPES … (M2, not run tonight)", but the same page shows M2 n = 50.
  - (b) README Limitations says "Not run tonight: M2 …, M3 …, end-to-end segment replacement, and other ASR variants. Only whisper-tiny + Cardiff are wired." All four were run and are reported in the README's own Results section.
  - (c) report_whisper-base.html keeps the hard-coded heading "does whisper-tiny attend…" and the answer text "Whisper-tiny's attribution…", and its H3 card reads "pending: Phase 1 gold-text ablation not found". The README handles base's H3 correctly.

## Item 7 — FAIL (documentation only)
Cmd: grep of README.md and summary.json. The authoritative values come from T1_final/results/AADD_READY (`aadd 3.1.0 model v3 (v3.1) threshold 0.5 resolution 0.02`) and aadd_devtest.log (`model=v3`; the seg csv has 0.02 s segments).
- Recorded: aadd 3.1.0 (both files). Threshold 0.5 (summary.json `flag_rule_frame`, "any class probability >= 0.5", and the clip rule p95 ≥ 0.5). 20 ms segments (summary.json).
- Missing from both README.md and summary.json: "v3.1" (both say only "model v3") and `--resolution 0.02` / `--segments`. These appear only in HYPOTHESES.md and AADD_READY. The README also gives no flag threshold.
Smallest fix: in `scan.py` summarize, set `aadd="aadd 3.1.0 CLI, model v3 (v3.1), threshold 0.5, --segments --resolution 0.02 (20 ms)"` and re-run `audioxai summarize` (CPU). In the README "Models and versions", change "AADD 3.1.0 (model v3)" to "AADD 3.1.0 (model v3.1; `aadd --segments --resolution 0.02`; frame flagged if any class p ≥ 0.5)".

## Item 8 — PASS (with notes)
Cmd: `git log` on both thresholds.json files, and a JSON leaf diff of T2 98e2cdf vs f84eeba vs the working tree. Code review: grep of every `label` use in `audioxai/*.py`.
- `T1_final/results/thresholds.json`: the only commit is 016d798 (04:49:48). mtime 04:46:43, working tree clean, so it is unchanged after TEST_UNLOCKED.
- `T2_variants/results/thresholds.json`: it was committed at 98e2cdf (05:04:57) and then **touched at f84eeba (05:45:01)**, after unlock. Leaf diff: 168 → 200 leaves, **0 changed, 0 removed**, 32 added. All the additions are new `whisper-tiny-wsp:*` entries from the P5 grid. The working tree equals f84eeba. No existing reported configuration changed.
- Label use in X's code:
  - correctness after prediction (`metrics.py` 84–86: correct, correct_gold_text, p_label);
  - ΔP(gold label) for the e2e window replacement (`occlusion.py` 108), an evaluation metric after prediction;
  - stratified sampling (`scan.py` 18–34), pre-registered and required by item 2;
  - display and counts (report, label_counts);
  - **clip ordering by label for the M2/e2e subsets** (`occlusion.py` 53, 117: non-neutral first). This does not match the brief's wording. It only chooses which test clips get the descriptive M2 and e2e analysis (M2 later ran on all 50, e2e on 10), and it feeds no threshold, model or configuration.
- AudioXAI reads no thresholds or configs, and no label reaches attribution: IG targets Cardiff's *predicted* class.
Verdict: no test finding changed a reported configuration.

## Summary
| # | Item | Verdict |
|---|---|---|
| 1 | HYPOTHESES committed before first test run | PASS (addendum committed only post-test; pre-test by mtime + recorded sha) |
| 2 | Test sample reproduces from seed | PASS |
| 3 | M1 / M2 recompute vs npz | PASS (exact at fp16 storage; 1e-4 is below fp16 resolution) |
| 4 | Deletion + randomization ran and reported (incl. FAIL) | PASS |
| 5 | Focus ratio, CI, zero-flag count | PASS |
| 6 | report.html numbers match CSVs | PASS (stale non-numeric text, see 6a–c) |
| 7 | AADD version and flag rule recorded | FAIL (v3.1 and --resolution 0.02 missing from README/summary.json) |
| 8 | Configs unchanged after unlock; label use | PASS (T2 thresholds only appended; label-ordered M2/e2e subset noted) |

Other notes:
- `scripts/run_ext.sh` (untracked, in progress) writes M2 into the tracked `results/n100`, so n100 files may change after this audit.
- n100's M3 CI [0.930, 0.997] lies below 1. This is descriptive and does not affect H1.
