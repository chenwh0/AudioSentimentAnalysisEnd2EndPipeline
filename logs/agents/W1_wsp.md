# W1 WSP-Whisper log

## Status 04:39 CST
- Passes used: 0/20 (cap tonight), budget 50.
- Done: splits (Milestone2/WSP_Whisper/splits/, READY written 04:35; 250 practice / 250 holdout, 12/184/54 per label each).
- Harness: wsp/eval_harness.py locked, sha256 in wsp/HARNESS_LOCK.txt. Before locking, one base holdout eval run (logged in eval_runs.jsonl): holdout WER gold 0.1843, pseudo 0.1514.
- Base test run (ONLY one, never rerun): n=1500 WER gold 0.2206, WER vs whisper-large-v3 pseudo-ref 0.19882 == teammate's 0.1988 (gap 0.0000 < 0.01, no explanation needed). A1 1.000, A2 0.983, long 3, short 6. File results/eval_base_test.csv.
- Next: fp_counter.py, diagnostic passes on practice, layer audit.

## Status 04:54 CST
- Passes used: 3/20 (diagnostic, practice clips 1-192 of rng(8001) order).
- Done: results/layer_audit.csv (every parameter: shape, norm, mean, std, absmax, SVD, activation stats of owning module, outlier channels; tied proj_out<->embed_tokens = True, consequence noted: a vocab-row edit changes both the token's input embedding and its output logit). Diagnostic wsp/diag/diag_summary.json.
- Diagnostic (practice, 192 clips, 5299 gold words, WER 0.193: S 400, D 491, I 132). Repetition (trigram x3) 4 clips, long 0, short 1, clips with digits 12. Largest single error source: gold disfluencies dropped: 'uh' deleted 67, 'um' 46 (gold has uh 83, um 52; model emits 2+1). Second: colloquial forms rewritten: cause->because 10, gonna->to 8 (+'going' ins 9), kinda->of 7 (+'kind' ins 5), gotta->to 3.
- Hypotheses: H1 fillers suppressed in the tied output rows of ' uh'/' um' -> scale those rows up. H2 colloquial tokens (' gonna',' kinda',' gotta',' cause') under-scored -> scale rows. H3 (backup) digit tokens -> spelled numbers in gold.
- Running: pass 4, teacher-forced probe (wsp/probe.py) measuring logit ratio max_other/target at gold positions; waiting on GPU lock (other workers' jobs).
- Next: choose one row-scale per group from the probe margins (written rationale), measure each with one practice pass.
