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

## Status 05:09 CST
- Passes used: 10/20. Edit ledger: Milestone2/WSP_Whisper/edits/edit_log.jsonl (hypothesis, tensor, rows, op, before/after per edit); state edits/state.json.
- Pass 4 teacher-forced probe (wsp/diag/probe_margins.json): median max_other/target ratio at gold positions uh 1.08, um 1.05, cause 1.06, kinda 1.05; logits reconstruct from final-LN hidden @ tied rows (max abs err 0.016).
- M = first 64 practice clips of rng(8001) order; base M WER 0.1832 (practice, not a result).
- E1 H1_fillers: tied rows uh/um (6 case variants) x1.10 -> M 0.1832->0.1793 KEEP (2,304 weights).
- E2 H2_colloquial: tied rows cause/kinda/gonna/gotta (8) x1.10 -> M 0.1793->0.1782 KEEP (3,072 weights).
- E3 H1_fillers: filler rows to x1.20 total -> M 5.57 (uh-uh-uh repetition collapse: tied row also scales the fed-back input embedding) REVERT. H1 failure 1 of 2; filler rows stay at x1.10.
- E4 H3_digits: 426 digit-token tied rows x0.80, isolated on the 12 diag clips with digits: 0.1870->0.1543 KEEP; E5 replaces E4 with x0.65: 0.1370 KEEP (163,584 weights).
- Pass 10: cumulative E1+E2+E5 on M: 0.1733 (base 0.1832). Saving fp10 (edits/patch_fp10.safetensors + edits_fp10.json).
- Next: holdout eval of fp10 (harness), then passes 11-20 on new mechanisms, fp20, register by holdout WER at <=07:20.

## Status 05:20 CST
- Passes used: 14 (+E9 running = 15). fp10 saved: edits/patch_fp10.safetensors, edits/edits_fp10.json (E1+E2+E5, 168,960 weights = 0.447%).
- Holdout eval runs (summary only, harness): base 0.1843; fp10 0.2302 (long=1: filler repetition loop); cand_E2E5 0.1758; cand_E5 0.1802; cand_E2E5E8 0.1760.
- Decisions: E1 reverted, H1 (filler row scaling) BLOCKED after two consecutive failures (E3 collapse on M, E1 loop on holdout). E6 (cause x1.20) REVERT on M. E7 (digits x0.50) REVERT ('$$$' collapse on one clip). E8 new mechanism (cancel half of final-LN-bias offset on filler rows): +4 errors fixed on M but no holdout gain -> REVERT.
- Caveat recorded: holdout used for 2 keep/revert decisions (E1, E8) as the brief allows; slight optimistic bias on holdout numbers; test untouched.
- Running: E9 (undo kinda part of E2). Next: practice generalization pass on unseen practice clips, save final checkpoint, register by holdout by 07:20 latest (target ~06:00).

## Status 05:35 CST
- Passes used: 17 (final). E9 (kinda rows back to base) KEEP on M (0.1793->0.1766); E2+E9 consolidated as E2net (cause/gonna/gotta rows x1.10). Passes 16-17: generalization check on the 58 never-used practice clips: base 0.2883 -> state 0.2836, no new loops (the one loop clip loops in base too).
- Editing stopped at 17 passes (05:25): remaining error mass (function-word deletions at clip edges, disfluent repetitions, acoustic substitutions) has no row-level mechanism I could state as a hypothesis; H1 blocked, H2/H3 refinements already failed once each.
- Saved fp17: edits/patch_fp17.safetensors + edits/edits_fp17.json (E2net + E5, 165,888 weights, 0.439%). Holdout fp17 0.1770.
- REGISTERED (05:25:49): fp17, results/REGISTERED.txt, before any edited test run.
- Test (once each): fp17 WER gold 0.2152 (base 0.2206), pseudo 0.2100, long 3, short 9. fp10 test running; then fp17 dev run, final_results.py, post.py.
