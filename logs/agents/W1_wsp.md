# W1 WSP-Whisper log

## Status 04:39 CST
- Passes used: 0/20 (cap tonight), budget 50.
- Done: splits (Milestone2/WSP_Whisper/splits/, READY written 04:35; 250 practice / 250 holdout, 12/184/54 per label each).
- Harness: wsp/eval_harness.py locked, sha256 in wsp/HARNESS_LOCK.txt. Before locking, one base holdout eval run (logged in eval_runs.jsonl): holdout WER gold 0.1843, pseudo 0.1514.
- Base test run (ONLY one, never rerun): n=1500 WER gold 0.2206, WER vs whisper-large-v3 pseudo-ref 0.19882 == teammate's 0.1988 (gap 0.0000 < 0.01, no explanation needed). A1 1.000, A2 0.983, long 3, short 6. File results/eval_base_test.csv.
- Next: fp_counter.py, diagnostic passes on practice, layer audit.
