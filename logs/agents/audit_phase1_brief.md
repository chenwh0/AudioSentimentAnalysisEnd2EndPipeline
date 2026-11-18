# Phase 1 audit brief (fresh-context Opus auditor)

You audit Phase 1 (T1 final) of an overnight run. You have NOT seen the orchestrator's reasoning; judge only from files. Write pass/fail per item with the evidence (commands and numbers you recomputed) to R/logs/audit_phase1.md. Do not modify any other file. Do not read load-pool audio. You may use the GPU only through `flock /tmp/m2_gpu.lock` and only for item 8.

R = /home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline ; P = /home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python
T1 = R/Milestone2/T1_final (config.yaml, pipeline/, run_t1.py, s2t_bug_probe.py, data/, results/, T1_gap_analysis.md, T1_REPORT.md)
Slices: R/Milestone2/PipelineEvaluation/{development,test,load_pool}.csv and {development,test,load_pool}_indices.npy. Gold: T1/data/slue_voxceleb_test.tsv (id == file; normalized_text). Pseudo-reference: column openai/whisper-large-v3_transcript.
Normalization: lowercase, replace em/en dashes with spaces, re.sub(r"[^\w\s]", "", text), collapse whitespace (T1/pipeline/common.py::normalize_text). Filler-free variant drops uh/um/uhm/mm/hmm.

Checklist (recompute independently with your own short scripts; do not just read the orchestrator's numbers):
1. Slices rebuilt from the saved index files have counts 500 / 1,500 / 1,426 and no overlap; the test label counts are 74 / 1,104 / 322.
2. The gold join holds: coverage, label agreement and duration agreement are reported in T1/results/gold_checks.json and gold_join.csv; recompute coverage and label agreement from the TSV and CSVs; recompute the count of |slue span − duration| > 0.1 s and confirm the report states it and what was decided.
3. WER recomputed from the saved transcripts (T1/data/transcripts_*.csv, test slice only, gold reference, normalization above, empty refs skipped, jiwer corpus WER) matches T1/results/wer_test.csv to 4 decimals for every ASR, primary and filler-free; also the pseudo-reference column.
4. Macro-F1, accuracy and per-class metrics recomputed from T1/results/predictions_test.csv (pred_full vs label, per config) match m2_t1_results.csv (source=full) to 4 decimals; recompute one bootstrap CI (10,000 resamples, numpy default_rng(8001), resample clip indices, macro-F1 over Negative/Neutral/Positive) for one Cardiff configuration and compare with the stored CI (expect equality if you use rng.integers(0, n, size=(10000, n)); otherwise report the difference).
5. τ and t were tuned on dev only: grep T1/pipeline and run_t1.py for any use of test rows inside tune_tau / tune_threshold; confirm the dev mask is used; report the lines.
6. assertions.py matches the guide: A1 ≥ 2 words, A2 0.5 ≤ words/duration ≤ 5.0, A3 score ≥ t; test the edge cases by calling the functions: empty transcript, None, NaN; duration 0 and negative; NaN score; and the threshold rule (balanced accuracy over the 0.05…0.95 grid, first maximum wins).
7. The s2t fix is real: from T1/data/transcripts_s2t-small.csv (fixed) and R/Milestone2/PipelineEvaluation/transcripts_s2t-small-librispeech-asr.csv (old), recompute the test-slice counts of transcripts with more than 1.5× the pseudo-reference word count (normalized) before and after, and compare with T1/results/s2t_loop_analysis.json. Read s2t_bug_probe.json and say whether the stated cause is supported by its numbers.
8. run_t1.py runs clean and reproduces the tables: run `cd R && flock /tmp/m2_gpu.lock P Milestone2/T1_final/run_t1.py --results-dir Milestone2/T1_final/results_audit` (caches exist, so it should take under 3 minutes and use the GPU only if a cache is missing); diff m2_t1_results.csv, wer_test.csv, assertions_test.csv and thresholds.json between results/ and results_audit/ (allow floating differences below 1e-9).
9. Every numeric claim in T1/T1_REPORT.md points to a file; sample 10 numbers from the report and find each in the named file.

Format of R/logs/audit_phase1.md: a header with the time, then one block per item: `## Item N — PASS|FAIL`, the command(s) you ran, the numbers, and one line of verdict. End with a summary table and a list of failed items with the smallest fix. Be terse and exact.
