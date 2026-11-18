# Phase 1 audit (fresh-context auditor)
Run 2026-11-18 04:47–05:02 CST. I wrote my own scripts (scratchpad `i12.py i3.py i4.py i5.py i6.py i7.py i8.py i9.py`) and ran them with `.venv/bin/python`. The only project code imported is `pipeline/assertions.py`, for item 6. T1 = `Milestone2/T1_final`. No load-pool audio was read. The GPU was used only for item 8, under flock.

## Item 1 — PASS
Cmd: loaded `{development,test,load_pool}.csv` and `*_indices.npy`, then compared sets and counted labels.
- Counts: 500 / 1500 / 1426. Each index file has the same set as the CSV `original_index` column, and every entry is unique.
- Overlap between slices by index is 0/0/0. Overlap by `file` is also 0/0/0.
- Test labels: Negative 74, Neutral 1104, Positive 322.
Verdict: the slices are correct and disjoint.

## Item 2 — PASS
Cmd: joined all 3,426 slice rows to `slue_voxceleb_test.tsv` on id == file.
- The TSV has 3,553 rows, all with unique ids.
- Coverage is 3426/3426 (dev 500, test 1500, load_pool 1426). Label agreement is 3426/3426.
- |(end−start) − duration| > 0.1 s on **875** clips. Max 1.5701 s, mean 0.07149 s. All 875 are one-sided (dur_delta < −0.1 for 875 clips, > +0.1 for 0). Median shortfall among them is 0.2001 s.
- `gold_join.csv`: 3426 rows, 3426 in_slue, 3426 agree, 875 with |dur_delta| > 0.1. These match `gold_checks.json` exactly.
- What the report says: T1_REPORT §1 and §3 give 875 (25.5%), max 1.57 s and one-sided. The decision is to keep gold as primary and add a span-matched sensitivity check (`wer_test_spanmatched.csv`).
- I recomputed the sensitivity WERs: test span-matched n = 1090, s2t 0.4936 / 0.8797 and whisper-tiny 0.2018 / 0.2711. These match the CSV. The report text gets the n wrong (see item 9).
Verdict: the join holds, and the duration mismatch and the decision are both stated.

## Item 3 — PASS
Cmd: own normalization and jiwer corpus WER on the test slice, gold reference, empty references skipped (0 skipped). No transcript was missing or NaN.

| ASR | gold mine/stored | filler-free | pseudo-ref |
|---|---|---|---|
| AventIQ | 0.1831/0.1831 | 0.1662/0.1662 | 0.1478/0.1478 |
| s2t-small (fixed) | 0.5983/0.5983 | 0.6047/0.6047 | 0.6518/0.6518 |
| whisper-tiny | 0.2206/0.2206 | 0.2052/0.2052 | 0.1988/0.1988 |
| whisper-large-v3 (pseudo-ref column) | 0.1789/0.1789 | 0.1628/0.1628 | 0/0 |
| s2t OLD batch8 | 0.8639/0.8639 | 0.8765/0.8765 | 0.9285/0.9285 |

Verdict: all 15 values match to 4 decimals.

## Item 4 — PASS
Cmd: sklearn on `predictions_test.csv` (12 configs × 1500 rows; 0 predictions outside the 3 classes), compared with `m2_t1_results.csv` rows where source = full.
- I checked macro-F1, accuracy, and P/R/F1/n for each class. Mismatches at 4 decimals: 0 of 12 configurations.
- Bootstrap for whisper-tiny × Cardiff: `default_rng(8001).integers(0,1500,(10000,1500))`, then sklearn macro-F1 for each resample. This gives [0.5864778727, 0.6433315163]. The stored CI is [0.5864778727, 0.6433315163], so the difference is 0.0 / 0.0.
Verdict: exact match.

## Item 5 — PASS
Cmd: `grep -n "tune_tau\|tune_threshold\|is_dev\|is_test" run_t1.py pipeline/*.py`
- run_t1.py:65 sets `is_test = (dt["slice"]=="test")` and `is_dev = ~is_test`. Here `dt` is dev + test only, so `~is_test` is exactly the dev slice.
- :121 `S.tune_tau(kind, sc.iloc[np.where(is_dev)[0]], y_dev, tau_grid)`
- :130–131 `correct_dev = (pred[is_dev]==y_dev)` and `AS.tune_threshold(sc[is_dev], correct_dev, t_grid)`
- :136 `AS.tune_threshold(clean[is_dev], correct_dev, t_grid)`
- `sentiment.py::tune_tau` and `assertions.py::tune_threshold` read only their arguments. Score frames come back in `dt` key order (`score_texts` returns `allc.loc[keys]`), so the positional dev mask is aligned with the labels.
- Test rows appear only in evaluation: lines 141–162 and 175.
- Spot check from the score caches: dev-tuned (τ, t) reproduces the stored values for whisper-tiny×cardiff (None, 0.70), AventIQ×siebert (0.99, 0.05) and s2t×distilbert (0.99, 0.05).
Verdict: τ, t and t_b are tuned on dev only.

## Item 6 — PASS
Cmd: called the `pipeline/assertions.py` functions directly; 26/26 checks pass.
- A1: "" → F, None → F, NaN → F, np.float64 NaN → F, "   " → F, 1 word → F, 2 words → T.
- A2: empty/None/NaN transcript → F. Duration 0, −1, NaN or None → F. Rate 0.5 → T and 5.0 → T (bounds inclusive); 0.49 → F; 6 → F.
- A3: NaN → F, None → F, s == t → T, s < t → F.
- `tune_threshold`: the grid is 0.05…0.95 (19 values). On random data it picks the first argmax of sklearn balanced accuracy (0.60). With constant scores, where every value ties, it returns 0.05. On a perfect plateau it returns the first value, 0.15. A NaN score counts as a fail.
Verdict: matches the guide (A1 ≥ 2 words, 0.5 ≤ rate ≤ 5.0, score ≥ t, balanced-accuracy rule with the first maximum winning).

## Item 7 — PASS (with note)
Cmd: own word counts (normalized) on the test and dev slices; old = `PE/transcripts_s2t-small-librispeech-asr.csv`, new = `T1/data/transcripts_s2t-small.csv`.
- Test, > 1.5× pseudo-ref: before 244, after 134. Against gold: 229 → 115. Mean words 40.867 → 33.634; gold 26.816.
- Dev: 92 → 49 (pseudo-ref) and 82 → 40 (gold).
- All of these equal `s2t_loop_analysis.json`.
- `s2t_bug_probe.json` (finished 05:00, 48 dev clips; > 1.5× pseudo-ref):

| condition | clips | mean words |
|---|---|---|
| pipeline batch 8 | 11 | 43.6 |
| pipeline batch 1 | 5 | 31.6 |
| generate batch 8 with mask | 11 | 43.6 (output identical to the pipeline) |
| generate batch 8 without mask | 38 | 60.7 |

  The pipeline's preprocess keys include `attention_mask`.
- Is the cause supported? The stated cause is that batch-of-8 padding causes the loops and batch 1 fixes them. That is supported: batch 1 halves the loops, 11 → 5. The probe also shows the pipeline **already passes an attention mask**, so a missing mask is *not* the cause. The batch-8 degradation persists even with a correct mask, which points to padding or batch-generation effects. Dropping the mask makes things far worse (38).
- Note: T1_REPORT §4 still says "write-up in section 4a below, appended when the probe finished", and no §4a exists as of 05:02.
Verdict: the fix is real and the counts are reproduced. The batching cause is supported and the missing-mask hypothesis is refuted, but the report has not yet written this up.

## Item 8 — PASS
Cmd: `cd R && flock /tmp/m2_gpu.lock P Milestone2/T1_final/run_t1.py --results-dir Milestone2/T1_final/results_audit`
- Exit 0. The log has 34 lines, with no traceback or warning.
- Wall time was 11:29, almost all of it waiting for the lock. The script itself ran in 10 s (provenance 05:00:34 → 05:00:44).
- No transcription or scoring ran (all caches were hit), and `gpu_jobs.jsonl` has no entry from this run.
- Diff of `results/` against `results_audit/`:
  - `m2_t1_results.csv` (48×21), `wer_test.csv` (5×7) and `assertions_test.csv` (12×31): max |Δ| = 0.0, NaN patterns and text columns identical.
  - `thresholds.json`: same 104 flattened keys, 0 differences.
Verdict: reproduces bit-exactly.

## Item 9 — FAIL
Cmd: `i9.py` read each named file and compared it with the report text. I sampled 13 numbers:
1. 875 / 25.5% / max 1.57 s → `gold_checks.json`: 875, 0.2554, 1.5701. OK
2. WER 0.1831 / 0.2206 / 0.5983 / 0.1789, filler-free 0.1662 / 0.2052 / 0.6047 / 0.1628 → `wer_test.csv`. OK
3. Cardiff gold 0.6371; full 0.6211 / 0.6150 / 0.6032; pseudo 0.6066 → `m2_t1_results.csv`. OK
4. Gaps −0.016 [−0.033, +0.001], −0.022 [−0.042, −0.003], −0.034 [−0.056, −0.012]; "7 of 12 significant" → `gap_ci.csv`: −0.0160 [−0.0332, 0.0008], −0.0222 [−0.0417, −0.0032], −0.0339 [−0.0561, −0.0123]; 7/12. OK
5. Correlations −0.238 / −0.067, −0.083 / +0.056, −0.293 / −0.240 → `correlations.json`. OK
6. Baseline 0.7360, 0.2826 [0.2776, 0.2874] → `baseline.json`. OK
7. A4a passes 46–63% → `assertions_test.csv` gives 0.4627–0.626. OK
8. AventIQ×Cardiff row: 0.985, 0.699 (0.65), 0.481, 0.441 (0.70), 0.691, 0.765 / 0.608, … → `assertions_test.csv`. OK
9. **"on the 1,118 span-matched test clips versus the 382 mismatched ones" → WRONG.** `wer_test_spanmatched.csv` has n = 1090 / 410, and I recomputed 1090 test clips with |Δ| ≤ 0.1. The WER values quoted in that sentence do match the file.
10. **"t_b (A4b): 0.15–0.90" → WRONG.** `thresholds.json` has t_b = 0.05 for s2t×DistilBERT; the range is 0.05–0.90.
11. Words/s 3.43 vs 3.47, 4 clips > 1 s → `gold_join.csv` gives 3.429 / 3.467 (mean of ratios) and 4. OK. The "99th percentile 0.77 s" is the p99 of the 875 mismatched clips (0.773); the p99 over all clips is 0.548. These are cited only to `logs/ledger.md`, not a results file.
12. τ = 0.99 for every binary configuration and text source → `thresholds.json`. OK
13. **§4 promises a §4a probe write-up that does not exist.** The cause numbers (11 / 5 / 11 / 38) are in no report text.

Verdict: 2 numbers contradict their named file, and 1 promised section is missing.

## Summary
| Item | Result |
|---|---|
| 1 Slices | PASS |
| 2 Gold join | PASS |
| 3 WER | PASS |
| 4 F1 / CI | PASS |
| 5 Dev-only tuning | PASS |
| 6 Assertions | PASS |
| 7 s2t fix | PASS (report write-up pending) |
| 8 Rerun reproduces | PASS |
| 9 Report numbers | FAIL |

## Failed items and the smallest fix
- **Item 9**, three edits to T1_REPORT.md:
  - §3: change "1,118 span-matched … 382 mismatched" to "1,090 … 410".
  - §6: change "t_b … 0.15–0.90" to "0.05–0.90".
  - Add §4a from `results/s2t_bug_probe.json`. Batch 8 gives 11/48 clips over 1.5×, batch 1 gives 5/48. Batch 8 through `generate` with a mask gives 11, the same as the pipeline, and without a mask gives 38. The pipeline already passes `attention_mask`, so the cause is batch-8 padding or batched generation, not a missing mask.
  - Optional: say the 0.77 s figure is the p99 among the 875 mismatched clips.
