## 2026-11-18T04:34:20-06:00
975 MiB, 20475 MiB, 21 %
disk avail 867G
lines: ledger 12 gpu_jobs 1 fp  eval 
python procs: 1
newest files: 7 modified in last 15 min


## Sonnet monitor, Phase 1 start
Time: 2026-11-18T04:34:56-06:00
1. GPU: memory.used 976 MiB / 20475 MiB, utilization 16 % (RTX 4000 Ada). Idle, no job holding the GPU.
2. Disk /: 867G available (51 % used of 1.8T). OK.
3. Logs:
   - logs/ledger.md: 12 lines, mtime 2026-11-18 04:34:20
   - logs/gpu_jobs.jsonl: 1 line, mtime 2026-11-18 04:15:17
   - logs/forward_passes.jsonl: does not exist yet
   - logs/eval_runs.jsonl: does not exist yet
4. gpu_jobs.jsonl pairing: the only line is an "end" event for job preflight_gpu_warmup (pid 382308, 0.148 s, 04:15:17) with no matching "start" line. No unmatched start lines, so no job is left open. Minor anomaly: the preflight wrote an end without a start.
5. Python processes from /e2eML_course/.venv/bin/python: none running. pgrep matched only bash -c wrappers whose command text contains the pattern (the orchestrator's setup command and this check). W1/W2 workers have not started GPU jobs yet.
6. Changes in last 20 min under logs/ and Milestone2/: yes. Modified: logs/ledger.md, logs/oracle_plan.md, logs/gpu_jobs.jsonl, logs/monitor.md, logs/agents/W1_wsp_brief.md, logs/agents/W2_audioxai_brief.md, Milestone2/T1_final/data/slue_voxceleb_test.tsv, Milestone2/T1_final/data/SLUE_LICENSE. Newest change 04:34. No stall.
Verdict: HEALTHY at phase start. Watch items: no worker python process yet; forward_passes/eval_runs logs absent until workers run.
## 2026-11-18T04:49:20-06:00
3666 MiB, 20475 MiB, 99 %
disk avail 867G
lines: ledger 13 gpu_jobs 52 fp 3 eval 2
python procs: 11
newest files: 132 modified in last 15 min

## 2026-11-18T05:04:20-06:00
2722 MiB, 20475 MiB, 97 %
disk avail 867G
lines: ledger 17 gpu_jobs 83 fp 7 eval 2
python procs: 6
newest files: 119 modified in last 15 min


## Sonnet monitor, Phase 2 done / Phases 3-4 running
- 2026-11-18 05:06 CST. Verdict: HEALTHY.
- GPU: 977/20475 MiB used, 1% util; one compute app pid 236492 (28 MiB). Disk: 867G free (51% used).
- Logs (lines, mtime): ledger.md 19 (05:04:57); gpu_jobs.jsonl 92 (05:05:57); forward_passes.jsonl 9 (05:05:52); eval_runs.jsonl 2 (04:38:59); W1_wsp.md 16 (04:54:01); W2_audioxai.md 19 (05:05:57).
- gpu_jobs: 0 malformed lines; 1 unmatched start: w2_test_scan_a pid 424512, age ~0.1 min (running now, `audioxai scan --slice test --sample 50 --limit 25`, pid 424514).
- forward_passes: 9 lines (<=20), every line has hypothesis_id.
- GPU lock /tmp/m2_gpu.lock held by flock 424511 / bash 424512 / audioxai 424514 (the active W2 job); no other waiters.
- Activity in last 20 min: yes. W2_audioxai.md at 05:05, W1_wsp.md at 04:54 (about 12 min ago), plus edits under Milestone2/WSP_Whisper and Milestone2/AudioXAI. No stall. W1 is closest to the 20-minute limit (quiet since 04:54, but WSP files were modified in the window).
## 2026-11-18T05:19:20-06:00
8773 MiB, 20475 MiB, 100 %
disk avail 866G
lines: ledger 23 gpu_jobs 121 fp 14 eval 6
python procs: 5
newest files: 216 modified in last 15 min

## 2026-11-18T05:34:20-06:00
2723 MiB, 20475 MiB, 97 %
disk avail 866G
lines: ledger 23 gpu_jobs 135 fp 17 eval 8
python procs: 4
newest files: 123 modified in last 15 min


## Sonnet monitor, Phase 5 start
2026-11-18 05:44 CST. Verdict: HEALTHY.
- GPU: 6664/20475 MiB, util 100%. Compute apps: pid 463020 (5686 MiB, audioxai scan), pid 236492 (28 MiB).
- Disk /: 866G free (51% used).
- Logs (lines, mtime): ledger.md 26, 05:43:25; gpu_jobs.jsonl 151, 05:43:57; forward_passes.jsonl 17, 05:24:32 (20 min old, expected: no forward-pass job running other than scan); eval_runs.jsonl 10, 05:40:39; agents/W2_audioxai.md 53, 05:43:14; agents/figures.md 16, 05:07:39 (37 min old, but figures/ files changed in last 20 min).
- Unmatched starts in gpu_jobs.jsonl: 1, w2_n300_1 pid 463018, started 05:43:57 (age under 1 min), matches live process `audioxai scan --slice test --sample 300 --limit 55` (pid 463020) holding /tmp/m2_gpu.lock (flock 461765, bash 463018, audioxai 463020).
- run_t1_P5grid ended exit=0 at 05:43:51.
- Git: modified T2_variants registry/results, AudioXAI paths.py/scan.py, logs/gpu_jobs.jsonl; untracked n300 results, run_ext.sh, OVERNIGHT_REPORT.md. HEAD febbbe2 (P3 done partial scope).
- Changed in last 20 min: logs/agents/W2_audioxai.md, W1_wsp.md, figures/wsp_results_table.*, figures/specs/*.json, AudioXAI/results n300/*, summary.json, run_ext.log.
## 2026-11-18T05:49:20-06:00
8498 MiB, 20475 MiB, 100 %
disk avail 865G
lines: ledger 30 gpu_jobs 153 fp 17 eval 10
python procs: 2
newest files: 268 modified in last 15 min

