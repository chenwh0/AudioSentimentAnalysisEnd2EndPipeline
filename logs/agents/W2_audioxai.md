# W2 AudioXAI log
## 04:37 CST
- done: Milestone2/AudioXAI/HYPOTHESES.md written (before any test audio read)
- running: none
- blockers: AADD_READY, WSP splits/READY, TEST_UNLOCKED absent
- next: inspect data, aadd on 2 smoke dev clips, build package
## 04:49 CST
- done: package Milestone2/AudioXAI/{pyproject.toml,audioxai/{core,metrics,faithfulness,scan,report,cli,paths}.py,README.md,scripts/gpujob.sh}; pip install -e into .venv OK; AADD scan of 2 smoke dev clips -> results/dev/aadd_smoke*.{json,csv}; explain on 2 smoke clips worked (results/dev/npz)
- running: dev smoke scan with faithfulness (queued on GPU lock behind T1/T2/W1 jobs)
- blockers: GPU lock contention; AADD_READY, WSP READY, TEST_UNLOCKED absent
- next: verify dev summary + report, then wait for TEST_UNLOCKED
## 05:06 CST
- done: dev pipeline validated on 8 WSP-holdout dev clips (results/dev/{aggregate.csv,faithfulness.csv,summary.json,report_dev.html,m2.csv,e2e_segments.csv}); M1, M3 (Grad-CAM, time-only), M4, M2 (SPES-style occlusion), item 7 (segment replacement) all implemented; H3 wired to T1_final/results/gap_ci.csv
- dev finding: AADD flags 54% of dev 20 ms frames; 27% clips zero-flag, 21% all-flag -> pre-registered addendum in HYPOTHESES.md (informative-subset FR, graded rho) before TEST_UNLOCKED
- dev finding: model-randomization rho ~0.89 on dev (grad x input dominated by input) -> expect FAIL
- GPU: ~3-8 s/clip incl. faithfulness when lock free; CPU fallback ~45 s/clip
- running: none; blockers: TEST_UNLOCKED absent
- next: wait for TEST_UNLOCKED, then scan test 50 (chunks of 25), M2 n=20, e2e n=10, report
- 05:05 TEST_UNLOCKED seen; HYPOTHESES.md sha256 6045a4492cbb8a4b frozen before first test job
## 05:14 CST
- done: test 50-clip scan (results/test_sample_ids.csv, aggregate.csv, faithfulness.csv 20 deletion/10 randomization), preliminary summary.json
- prelim: H1 mean FR(M1) 0.991 [0.911,1.061] n=34 -> refuted (CI includes 1); 16 zero-flag, 16 all-flag clips; deletion AUC top 1.43 vs random 1.18, diff 0.245 [0.187,0.303]; randomization rho 0.916 -> FAIL; H2 refuted (err flagged 0.242 vs clean 0.258); H3 strictly refuted (gold-ASR +0.022 [0.003,0.042])
- running: M2 occlusion n=20, then e2e n=10, report (scripts/run_test.sh)
- next: per coordinator, extend to 100 clips in results/n100 (50 is a per-label prefix subset by construction), then whisper-base if time
## 05:26 CST
- done (priority 1): results/{test_sample_ids.csv,aggregate.csv,faithfulness.csv,summary.json,m2.csv,e2e_segments.csv}, report.html (8.9 MB, 50 clips, offline, no audio)
  H1 refuted: FR(M1) 0.991 [0.911,1.061] n=34 (16 zero-flag NaN, 16 all-flag); informative 0.983 [0.832,1.113] n=18
  H2 refuted: err rate flagged 0.242 vs clean 0.258; err-word importance share - word share +0.006 [-0.010,0.024]
  H3 strictly refuted by Phase 1: gold - ASR macro-F1 +0.022 [0.003,0.042]; gold ceiling 0.637
  deletion AUC top 1.426 vs random 1.181, diff 0.245 [0.187,0.303], n=20 (19/20 top>random); randomization rho 0.916 n=10 FAIL
  M2 on all 50: M1-M2 time-profile Spearman 0.162 [0.136,0.189]
- done (priority 2): results/n100 (100 clips, 50 reused by construction): H1 0.995 [0.939,1.049] n=71 refuted; H2 refuted
- running (priority 3): whisper-base on same 50 (scripts/run_test_base.sh -> results/whisper-base)
- next: README results section, final audit of numbers
