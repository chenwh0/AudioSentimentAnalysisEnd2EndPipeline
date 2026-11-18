# Phase 4 audit brief (fresh-context Opus auditor) — AudioXAI, scoped run

R = /home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline ; P = /home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python ; X = R/Milestone2/AudioXAI
Write pass/fail with evidence to R/logs/audit_phase4.md; modify nothing else; GPU only via `flock /tmp/m2_gpu.lock`, only for item 3, under 5 minutes total; never read load-pool audio.

1. HYPOTHESES.md was committed before the first test run: compare `git -C R log --format=%ci --diff-filter=A -- Milestone2/AudioXAI/HYPOTHESES.md` with the first R/logs/gpu_jobs.jsonl line whose job is a W2/audioxai test-slice job, and with the creation time of X/TEST_UNLOCKED (stat).
2. The test sample reproduces from the seed: test.csv sorted by original_index, labels alphabetical, default_rng(8001), permutation within label, proportional allocation to the stated n; compare with X/results/test_sample_ids.csv.
3. For 5 random clips (your own default_rng(8001) choice over the sample), recompute one token's M1 map (gradient × input on the log-mel with SmoothGrad as X's code defines it; call X's own function with its own seed so the comparison is fair, and separately a plain gradient×input without noise to sanity-check sign and scale) and, if M2 exists, one M2 patch score; compare with the saved .npz arrays (tolerance 1e-4 relative, float16 storage).
4. The deletion curves and the model-randomization check ran and their results are reported, including failures: read X/results/faithfulness.csv and X/results/summary.json; confirm the report.html shows the same numbers.
5. Focus ratio and CIs recomputed from X/results/aggregate.csv: mean focus ratio over clips with ≥ 1 flagged frame, 10,000-resample bootstrap CI with default_rng(8001); number of clips with zero flagged frames; compare with summary.json.
6. The numbers in report.html match the CSVs (sample 8 numbers).
7. The AADD version and flag rule are recorded (aadd 3.1.0, model v3.1, threshold 0.5, 20 ms segments via --resolution 0.02) in README.md or summary.json.
8. No test finding changed a reported configuration: confirm that Milestone2/T1_final/results/thresholds.json and Milestone2/T2_variants/results/thresholds.json are unchanged after TEST_UNLOCKED's timestamp (git log of those files), and that X's code never reads the test labels for anything except computing correctness after prediction.
Format as in the other audits.
