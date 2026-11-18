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
