# Oracle plan (agent-oracle: evaluation, multi-agent, harness, long-horizon) — 2026-11-18 04:30 CST

Launch: GO at 04:27 CST. Deadline: push to origin/m2-overnight by 08:20 CST, verified by 08:25. Hard stop on new work 07:40.
Compressed plan (confirmed by the user): Phase 1 complete by 06:30; Phase 2 by 07:05; Phases 3 and 4 PARTIAL in parallel until 07:40; Phase 5 07:40–08:20. Leftover time → Phase 4. Never cut Phase 1 or 5.

## Evaluation (success predicates, per phase)
- P1 DONE iff: T1_gap_analysis.md exists; gold join reported (coverage, label agreement, duration agreement); WER vs gold (primary + filler-free) and vs pseudo-reference for every ASR from saved transcripts; s2t fixed with before/after long-transcript counts; assertions.py with A1–A3 + A4a/A4b and pass-rate / correlation tables; τ,t re-derived on dev; baseline; M1-vs-M2 table; bootstrap CIs (paired gap); run_t1.py reproduces tables; T1_REPORT.md with file pointers; Opus audit written to logs/audit_phase1.md.
- P2 DONE iff: register_variants.py writes meta.json per variant (SHA revisions, sizes from listed files, test metrics); grid on test for all available variants; MinIO attempt logged.
- P3 PARTIAL target: splits + manifest; locked harness + HARNESS_LOCK; base test run; ≤20 forward passes through fp_counter; fp10 checkpoint; registered choice before any edited test run; results table with paired bootstrap CI.
- P4 PARTIAL target: HYPOTHESES.md committed before test; package + CLI; M1 (SmoothGrad n=8) + M4 + AADD timeline + word errors + sentiment IG on 50 stratified test clips, whisper-tiny; focus ratio with CIs; deletion curve ≥20 clips; randomization ≥10 clips; aggregate.csv; report.html.
- P5 DONE iff: figures (PNG) with specs + manifest; OVERNIGHT_REPORT.md; final audit; pushed hash verified by ls-remote.
- Deterministic checks: every reported number recomputed by an auditor from saved files. Baseline (always-neutral 0.7360 / 0.2826) kept runnable.

## Multi-agent (topology)
- Supervisor (this session) + 2 long-running Opus workers in disjoint directories of the same working tree (not separate worktrees: single branch, orchestrator is the only committer; decision recorded). W1: Milestone2/WSP_Whisper/. W2: Milestone2/AudioXAI/. Shared append-only logs: logs/gpu_jobs.jsonl, logs/forward_passes.jsonl, logs/eval_runs.jsonl.
- Context isolation: workers get briefs in logs/agents/*_brief.md, not my reasoning. Auditors are fresh Opus agents with checklists + paths only.
- Monitors: background bash loop → logs/monitor.md every 15 min, plus a Sonnet monitor agent at each phase boundary (brief §2).
- Agreement between agents is not evidence; only recomputed numbers.

## Harness (locked surfaces, durable state)
- Locked: wsp/eval_harness.py (SHA in HARNESS_LOCK.txt), fp_counter (raises at 51), gold join CSV, slice indices. GPU access only via `flock /tmp/m2_gpu.lock`. Test-clip gating for W2: file flag Milestone2/AudioXAI/TEST_UNLOCKED created by the orchestrator only after P1–P3 configs are committed.
- Durable: logs/ledger.md (append-only), OVERNIGHT_STATUS.md, per-agent logs. Rollback: git (every phase boundary commits).
- Approval boundaries: no push to main, no force, no PR; pushes only to refs/heads/m2-overnight; no system installs; secrets only from env (HF_TOKEN) / never logged. .gitignore for npz, model weights, large artifacts (>50 MB).

## Long-horizon (stop rules, decided per BRIEF_AMENDMENTS C1–C10)
- Download/load failure: 1 retry after 60 s, then skip + PARTIAL. OOM: halve batch once. Stall >20 min: kill, retry once with half batch. Audit: one fix cycle. Push failure: drop >50 MB files, retry once, else record local hash. Duration-mismatch investigation capped at 15 min. Final audit failing after 08:10: list failures, push anyway.
- Non-counting outcomes: numbers not computed from files saved tonight; test-slice tuning; a WSP "edit" with no measurable holdout effect; AudioXAI on fewer clips than aggregate.csv states; figures without spec + spot-check.
- Return condition: artifact-based (OVERNIGHT_STATUS.md + pushed hash verified).
