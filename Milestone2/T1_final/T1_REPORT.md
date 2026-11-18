# T1 final report — evaluation harness, gold transcripts, assertions (Milestone 2, overnight run 2026-11-18)

Every number below is read from a file under `Milestone2/T1_final/results/` (abbreviated `results/`), all produced by `run_t1.py` from `config.yaml` on this Linux workstation (RTX 4000 Ada), except the Milestone 1 numbers, which are quoted from `pipeline-results.md`. The gap analysis written before the work is `T1_gap_analysis.md`. Reproduce with:

```
.venv/bin/python Milestone2/T1_final/run_t1.py          # regenerates every table from the caches in data/
.venv/bin/python Milestone2/T1_final/s2t_bug_probe.py   # 48 dev clips, GPU
.venv/bin/python Milestone2/T1_final/wer_sensitivity.py
```

## 1. Result first

- **Gold transcripts exist and join perfectly.** SLUE-VoxCeleb `normalized_text` covers 3,426 / 3,426 clips and its sentiment label agrees with the DynamicSuperb label on 3,426 / 3,426 (`results/gold_checks.json`). The SLUE span is shorter than the WAV by more than 0.1 s on 875 clips (25.5%, max 1.57 s), always one-sided (`results/gold_join.csv`, column `dur_delta` ≤ 0). We kept gold as the primary reference and report a sensitivity check (section 3).
- **WER against human gold on the 1,500 test clips** (`results/wer_test.csv`): AventIQ 0.1831, whisper-tiny 0.2206, s2t-small (fixed) 0.5983, and the whisper-large-v3 pseudo-reference itself 0.1789. Filler-free: 0.1662, 0.2052, 0.6047, 0.1628. The pseudo-reference numbers the teammate reported (0.1478, 0.1988) are reproduced to 4 decimals in the same file (column `wer_pseudo_ref`).
- **The s2t "0.93 WER" was half a measurement bug.** Re-running with batch size 1 drops test transcripts longer than 1.5× the pseudo-reference from 244 to 134 and WER vs gold from 0.8639 to 0.5983 (`results/s2t_loop_analysis.json`, `wer_test.csv` row `s2t-small-OLD(batch8)`). The remaining loops are the model's own behaviour on this audio (section 4).
- **Gold text helps the classifier a little, not a lot.** Cardiff reaches macro-F1 0.6371 on gold text versus 0.6211 (AventIQ), 0.6150 (whisper-tiny) and 0.6032 (s2t) on ASR text (`results/m2_t1_results.csv`). The paired full-minus-gold gap is −0.016 [−0.033, +0.001] for AventIQ (CI includes 0), −0.022 [−0.042, −0.003] for whisper-tiny and −0.034 [−0.056, −0.012] for s2t (`results/gap_ci.csv`). The ceiling with perfect transcripts is therefore 0.637: the text classifier, not the transcript, is the bottleneck (H3 of Phase 4 holds in the sense that gold adds ≤ 0.034 macro-F1).
- **Assertions are in code** (`pipeline/assertions.py`: A1, A2, A3, A4a, A4b, with edge cases) and **A3 is still dead for the binary models** (t = 0.05, pass rate 100%, `results/thresholds.json`, `results/assertions_test.csv`). **A4a (classifier agreement) is the assertion we add**: it passes 46–63% of clips and raises accuracy among passing clips to 0.60–0.75 for every configuration, including the binary models where confidence carries nothing (section 5).
- **Pass rate still does not track macro-F1 across configurations.** Across the 12 configurations the correlation between the joint pass rate and macro-F1 is Pearson −0.238 / Spearman −0.067 for A1∧A2∧A3, −0.083 / +0.056 with A4a added, and −0.293 / −0.240 with A4b (`results/correlations.json`). A4a removes the negative correlation but does not create a positive one; the reason is structural (section 5).
- **Baseline**: always-neutral accuracy 0.7360, macro-F1 0.2826 [0.2776, 0.2874] (`results/baseline.json`). Every Cardiff configuration beats it on macro-F1 (0.60–0.62); no configuration beats it on accuracy (best 0.718).
- τ hit the top of the grid (0.99) for SiEBERT and DistilBERT in every configuration (`results/thresholds.json`, `tau_boundary_hit: true`), as in the teammate's run. These are boundary hits: the grid ends at 0.99 and both models put > 0.99 on almost every clip, so "Neutral when score < τ" barely fires.

## 2. Definitions: Milestone 1 versus Milestone 2 (brief §4.4)

| | Milestone 1 (`pipeline-results.md`) | Milestone 2 T1 (this report) |
|---|---|---|
| Clips | 3,553 (whole test split) | 3,426 after removing 104 Disagreement + 23 `<mixed>`; evaluated on the 1,500-clip test slice, tuned on the 500-clip dev slice |
| Classes | 5 (incl. Disagreement, `<mixed>`) | 3 (Negative, Neutral, Positive) |
| ASR models | AventIQ, s2t-small | AventIQ, s2t-small (batch-1 fix), whisper-tiny (+ whisper-base, moonshine-base, whisper-tiny-wsp in Phase 2) |
| Sentiment models | SiEBERT, Cardiff, tabularisai | + DistilBERT SST-2; binary models mapped to Neutral below τ (dev-tuned) |
| Metric | accuracy; macro-F1 over 5 classes | macro-F1 over 3 classes (primary), accuracy, per-class P/R/F1, 95% bootstrap CIs |
| Reference transcripts | none (no WER) | SLUE human gold (primary) and whisper-large-v3 pseudo-reference |
| Cardiff result | 67.0% / 0.35 (Whisper), 68.1% / 0.35 (s2t) | 0.7167 / 0.6211 (AventIQ), 0.7180 / 0.6032 (s2t), 0.7140 / 0.6150 (whisper-tiny) |
| SiEBERT result | 23.1% / 0.13 (Whisper), 21.1% / 0.12 (s2t) | 0.3580 / 0.3341 (AventIQ), 0.3453 / 0.3254 (s2t), 0.3447 / 0.3228 (whisper-tiny) |
| Majority baseline | 71% accuracy (2,521 / 3,553 neutral) | 0.7360 accuracy, 0.2826 macro-F1 (1,104 / 1,500 neutral) |

Why the numbers moved: macro-F1 rose from 0.35 to 0.60–0.62 for Cardiff mainly because two impossible classes (Disagreement, `<mixed>`) left the average, not because the pipeline changed; the per-class F1 for Neutral (0.78–0.80) and Positive (0.69–0.72) is in line with Milestone 1's per-class accuracy for Cardiff. SiEBERT moved from 0.12–0.13 to 0.32–0.33 because the τ mapping now lets it output Neutral at all, and from 3-class scoring. Accuracy barely changed (67–68% → 71–72%) because the data is 74% neutral either way.

## 3. Gold transcripts and WER (professor comment 2; brief §4.1)

Join and checks (`results/gold_checks.json`, `results/gold_join.csv`): coverage 3,426 / 3,426 (dev 500, test 1,500, load pool 1,426); label agreement 3,426 / 3,426; gold text empty on 0 clips. Durations: 875 clips (25.5%) have |SLUE span − WAV duration| > 0.1 s. Investigation (6 minutes, logged in `logs/ledger.md`): the difference is never positive, i.e. SLUE's (start, end) always lies inside the WAV; median shortfall 0.20 s among the mismatched clips, 99th percentile 0.77 s among them (0.55 s over all clips), 4 clips above 1 s; gold words per second are 3.43 against WAV length and 3.47 against the SLUE span. So the transcript describes the same utterance with trimmed edge silence, not a different span. The brief's 1% threshold is exceeded and reported as such.

Sensitivity (`results/wer_test_spanmatched.csv`): on the 1,090 span-matched test clips versus the 410 mismatched ones, WER vs gold is 0.1752 vs 0.2044 (AventIQ), 0.2018 vs 0.2711 (whisper-tiny), 0.4936 vs 0.8797 (s2t), 0.1680 vs 0.2083 (whisper-large-v3). Mismatched clips are harder for every model, including the pseudo-reference, so part of the headline WER is edge speech that SLUE did not transcribe; the ranking of models is the same on both subsets.

WER on the test slice (`results/wer_test.csv`; per-clip values in `results/wer_per_clip_test.csv`):

| ASR | WER vs gold | gold, fillers removed | vs whisper-large-v3 pseudo-ref |
|---|---|---|---|
| AventIQ-AI/whisper-audio-to-text | 0.1831 | 0.1662 | 0.1478 |
| openai/whisper-tiny | 0.2206 | 0.2052 | 0.1988 |
| facebook/s2t-small-librispeech-asr (batch 1, fixed) | 0.5983 | 0.6047 | 0.6518 |
| s2t-small, teammate's batch-8 transcripts | 0.8639 | 0.8765 | 0.9285 |
| openai/whisper-large-v3 (the pseudo-reference itself) | 0.1789 | 0.1628 | — |

The pseudo-reference is 0.18 WER away from the humans, so every "WER vs pseudo-reference" number understates the true error by roughly that much for the Whisper family (they share Whisper's conventions) and overstates it for s2t. Removing fillers helps the Whisper models (SLUE keeps `uh`/`um`, Whisper drops them) and slightly hurts s2t, which emits other fillers (`er`).

Gold-input ablation, three numbers per pair plus the pseudo-reference (`results/m2_t1_results.csv`, macro-F1 on test; τ re-tuned on dev per text source):

| ASR × sentiment | full pipeline | normalized ASR text | gold text | pseudo-ref text |
|---|---|---|---|---|
| AventIQ × Cardiff | 0.6211 | 0.6173 | 0.6371 | 0.6066 |
| s2t × Cardiff | 0.6032 | 0.5978 | 0.6371 | 0.6066 |
| whisper-tiny × Cardiff | 0.6150 | 0.6222 | 0.6371 | 0.6066 |
| AventIQ × DistilBERT | 0.4580 | 0.4653 | 0.4802 | 0.4651 |
| AventIQ × SiEBERT | 0.3341 | 0.3434 | 0.3473 | 0.3436 |
| AventIQ × tabularisai | 0.3304 | 0.3252 | 0.3389 | 0.3344 |

Normalizing the ASR text (lowercase, no punctuation, like SLUE) changes Cardiff by at most 0.007, so format explains none of the full-vs-gold gap; content does. Note the teammate's finding is reproduced: the whisper-large-v3 "gold" text scores *below* every real ASR for Cardiff (0.6066), which is why the old ablation showed a negative "upstream loss". With human gold the sign is the expected one. Paired gap CIs for all 12 configurations are in `results/gap_ci.csv`; the gap is significant (CI excludes 0) for 7 of 12.

## 4. The s2t loop bug (known defect 1; brief §4.2)

Before/after on the test slice (`results/s2t_loop_analysis.json`): transcripts with more than 1.5× the pseudo-reference word count 244 → 134 (vs gold: 229 → 115); mean words 40.9 → 33.6 against 26.8 for gold. WER vs gold 0.8639 → 0.5983. The fixed transcripts (`data/transcripts_s2t-small.csv`) are used everywhere; the old file stays at `Milestone2/PipelineEvaluation/transcripts_s2t-small-librispeech-asr.csv`.

Cause probe on 48 dev clips (`results/s2t_bug_probe.json`, `s2t_bug_probe.py`; dev clips chosen with `default_rng(8001)`), counting transcripts longer than 1.5× the pseudo-reference:

| condition | long transcripts / 48 | mean words (reference 24.3) |
|---|---|---|
| notebook pipeline, batch 8 | 11 | 43.6 |
| notebook pipeline, batch 1 (the fix) | 5 | 31.6 |
| `model.generate`, batch 8, explicit attention mask | 11 | 43.6 |
| `model.generate`, batch 8, no attention mask | 38 | 60.7 |

The pipeline's preprocess output does contain `attention_mask`, and its batch-8 output is identical to `generate` with an explicit mask, so the suspected "missing mask" is not the cause. The cause is padding itself: with a mask the loops halve relative to no mask, but padded log-mel frames still reach s2t's convolutional subsampler (whose receptive field crosses the mask boundary) and the decoder's cross-attention, and they more than double the long-transcript count compared with batch 1 (11 vs 5). Batch size 1 removes padding entirely, which is why it is the fix. The 5 / 48 loops that remain at batch 1 (134 / 1,500 on test) are the model's own behaviour on conversational VoxCeleb audio, outside LibriSpeech's read-speech domain.

## 5. Assertions (professor comment 1; brief §4.3)

Code: `pipeline/assertions.py` (pure functions, imported by Phase 2 and intended for the T3 services). A1 ≥ 2 words; A2 0.5 ≤ words/duration ≤ 5.0; A3 top-1 score ≥ t; A4a the configuration's mapped label equals a second classifier's mapped label on the same transcript (tabularisai for Cardiff configurations, Cardiff for all others); A4b AADD clean fraction (1 − `degraded_fraction` from `results/aadd_devtest_files.csv`, aadd 3.1.0 model v3.1) ≥ t_b. t and t_b are chosen on the dev slice by the guide's rule (balanced accuracy at predicting correctness over {0.05, …, 0.95}); τ by dev macro-F1 over {0.50, …, 0.99}. All values: `results/thresholds.json`.

Justification from the Milestone 1 calibration table (`pipeline-results.md`, lines 97–108): SiEBERT's confidence is 0.996 when right and 0.989 when wrong, so no threshold can separate them, and the guide's rule duly returns t = 0.05 (pass everything). Cardiff's 0.76 vs 0.69 is a weak signal, and the rule finds t = 0.65–0.75 with a pass rate of 49–70%. A label-free check that does not depend on the classifier's own confidence is needed for the binary models, hence A4a (a second opinion) and A4b (audio quality).

Pass rates and accuracy among passing/failing clips on test (`results/assertions_test.csv`; A1 passes 100% everywhere):

| config | A2 | A3 (t) | A4a | A4b (t_b) | A1∧A2∧A3 | acc pass / fail | +A4a | acc pass / fail | +A4b | acc pass / fail |
|---|---|---|---|---|---|---|---|---|---|---|
| AventIQ × Cardiff | 0.985 | 0.699 (0.65) | 0.481 | 0.441 (0.70) | 0.691 | 0.765 / 0.608 | 0.374 | 0.677 / 0.740 | 0.301 | 0.794 / 0.684 |
| whisper-tiny × Cardiff | 0.983 | 0.593 (0.70) | 0.489 | 0.362 (0.90) | 0.584 | 0.789 / 0.609 | 0.339 | 0.720 / 0.711 | 0.214 | 0.807 / 0.689 |
| s2t × Cardiff | 0.910 | 0.492 (0.75) | 0.474 | 0.582 (0.15) | 0.453 | 0.812 / 0.640 | 0.251 | 0.748 / 0.708 | 0.263 | 0.820 / 0.682 |
| AventIQ × SiEBERT | 0.985 | 1.000 (0.05) | 0.506 | 0.502 (0.45) | 0.985 | 0.359 / 0.273 | 0.499 | 0.613 / 0.104 | 0.493 | 0.364 / 0.353 |
| AventIQ × DistilBERT | 0.985 | 1.000 (0.05) | 0.618 | 0.458 (0.65) | 0.985 | 0.519 / 0.545 | 0.611 | 0.701 / 0.235 | 0.450 | 0.501 / 0.535 |
| AventIQ × tabularisai | 0.985 | 0.687 (0.50) | 0.481 | 0.441 (0.70) | 0.679 | 0.369 / 0.281 | 0.369 | 0.624 / 0.175 | 0.301 | 0.364 / 0.331 |

Reading: within a configuration, A3 separates correct from incorrect clips for Cardiff (0.77–0.81 vs 0.61–0.64) and does nothing for the binary models (A3 dead, as the calibration table predicted). A4a is the only assertion that works for every classifier: for SiEBERT it lifts accuracy among passing clips from 0.36 to 0.61 and drops failing clips to 0.10; for DistilBERT 0.52 → 0.70 / 0.24. For Cardiff, A4a's passing set is slightly *less* accurate than A3's (0.68–0.75), because agreement with the weaker tabularisai throws away many correct Cardiff answers; A4b (audio quality) adds a little on top of A3 for Cardiff (0.79–0.82 among passing) but discards 70–80% of clips.

Does A4 fix the negative correlation across configurations? No. With A1∧A2∧A3 the pass rate anti-correlates with macro-F1 (Pearson −0.238, Spearman −0.067, `results/correlations.json`) because the binary models pass 98.5% of clips at macro-F1 0.32–0.46 while Cardiff passes 45–69% at 0.60–0.62. Adding A4a brings the correlation to zero (−0.083 / +0.056); adding A4b makes it more negative (−0.293 / −0.240). The rankings never agree (`same_ranking: false`). This is structural: every threshold is tuned per configuration, so a pass rate is a within-configuration quantity and carries no information about which configuration is better. The honest statement for T1 is that assertions are useful for flagging individual clips (accuracy among passing clips is 0.10–0.30 higher than among failing clips for most configurations) and useless as a proxy for choosing a configuration.

## 6. Mappings, thresholds, exclusions, CIs (professor comments 3 and 4; brief §4.4)

- 3,426 clips verified in code (`pipeline/common.py::load_slices` asserts the counts, the index files and disjointness; test labels 74 / 1,104 / 322 in `results/baseline.json`).
- τ (binary models): 0.99 for every configuration and every text source, a boundary hit (`results/thresholds.json`). t (A3): Cardiff 0.65 / 0.75 / 0.70, tabularisai 0.50 / 0.50 / 0.70, binary 0.05 (dead). t_b (A4b): 0.05–0.90 (unstable across ASR, a sign that the AADD clean fraction is a weak predictor of correctness; the balanced accuracies are in `results/thresholds.json`).
- 95% bootstrap CIs (10,000 resamples, `default_rng(8001)`) for every macro-F1 (`results/m2_t1_results.csv`, columns `macro_f1_ci_lo/hi`, width about ±0.03 for Cardiff) and paired CIs for full-minus-gold, full-minus-pseudo and full-minus-normalized (`results/gap_ci.csv`).
- Per-class precision, recall and F1 for all 12 configurations and 4 text sources: `results/m2_t1_results.csv`. Cardiff's weakest class is Negative (F1 0.33–0.37 on 74 clips).

## 7. Reproducible entry point (brief §4.5)

`run_t1.py` + `config.yaml`: Linux paths rebuilt from the `file` column, no Windows calls, transcription and scoring only when a cache under `data/` is missing. It ran end to end once on this machine (`results/run_t1.log`, `results/provenance.json` with SHA-256 of every input cache); the auditor reran it into `results_audit/` and diffed the tables (`logs/audit_phase1.md`: 8 of 9 items passed on the first pass; the failed item was two wrong numbers in this report, corrected).

## 8. Limitations and open items

- Offline GPU timing columns (`mean_ms_per_clip`) are batch averages on this workstation and are **not** SLA measurements; T5/T6 must run on the course VM's CPUs.
- The s2t model still loops on 134 / 1,500 test clips after the fix; a decoding-side guard (repetition penalty, length cap tied to duration) would be a T2 variant knob, not a bug fix, and was not applied so that the T1 numbers stay those of the unmodified model.
- AADD was trained on synthetic LibriSpeech degradations; on VoxCeleb its flags are estimates, and A4b's threshold instability reflects that.
- Not done tonight: T3–T7 (services, Kubernetes, load generator, profiling), which need the course VM.
