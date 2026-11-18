# WSP-Whisper results (openai/whisper-tiny @ 169d4a4341b33bc18d8881c4b69c2e104e1cc0af)

All numbers are on the 1,500-clip test slice and come from one locked-harness run per checkpoint (`wsp/eval_harness.py`, sha256 in `wsp/HARNESS_LOCK.txt`). CIs come from a paired bootstrap: 10,000 resamples of clip indices with `default_rng(8001)`, the same indices for every model, and corpus WER recomputed from summed errors over summed gold words. Cardiff is `cardiffnlp/twitter-roberta-base-sentiment-latest` run on each checkpoint's transcripts with its 3 classes used as they are. Macro-F1 CIs use the same 10,000 resamples.

| checkpoint | changed weights | test WER (gold) [95% CI] | ΔWER vs base [95% CI] | WER vs large-v3 | long | short | A1 | A2 | Cardiff macro-F1 [95% CI] | ΔF1 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.2206 [0.1937, 0.2553] | – | 0.1988 | 3 | 6 | 1.000 | 0.983 | 0.6150 [0.5860, 0.6435] | – |
| fp10 | 168,960 (0.447%) | 0.3568 [0.2793, 0.4436] | +0.1361 [+0.0644, +0.2144] | 0.3692 | 18 | 5 | 1.000 | 0.974 | 0.6096 [0.5802, 0.6376] | −0.0054 [−0.0137, +0.0020] |
| **fp17 (registered)** | 165,888 (0.439%) | 0.2152 [0.1873, 0.2509] | **−0.0054 [−0.0079, −0.0026]** | 0.2100 | 3 | 9 | 1.000 | 0.983 | 0.6086 [0.5795, 0.6367] | **−0.0064 [−0.0119, −0.0016]** |

Changed-weight percentages are out of 37,760,640 unique parameters. Every edit sits in `model.decoder.embed_tokens.weight`, which shares storage with `proj_out.weight`, so each edited row changes both the token's input embedding and its output logit.

## Reading
- **fp17 (registered on holdout before any edited test run):** gold-reference WER drops by 0.54 points and the paired CI excludes zero. That is a small, real WER gain from 432 rescaled rows of the tied embedding:
  - x0.65 on 426 digit-containing tokens, so numbers get spelled out the way the gold transcripts write them;
  - x1.10 on the `cause`, `gonna` and `gotta` rows, so the colloquial forms the gold keeps get more weight.
- **The gain is mostly a style match to the SLUE gold normalization, not better acoustics.** WER against the whisper-large-v3 pseudo-reference rises from 0.1988 to 0.2100, because that reference writes digits and "because".
- **Cardiff macro-F1 falls by 0.0064, and that CI also excludes zero.** The WER gain does not help the downstream sentiment task. It slightly hurts it, plausibly because Cardiff handles spelled-out numbers and colloquial forms worse. Report both effects together.
- **fp10 is a failure, kept in the table for honesty.** Its filler-row boost (x1.10 on uh/um) sends 18 test clips into uh-uh-uh repetition loops. The holdout eval had already flagged this (0.2302, 1 long clip), and the edit was reverted before fp17.
- **Budget:** 17 forward passes of the 50 allowed. Tonight's cap was 20, so no fp20, fp25 or fp50 exists. Because the brief requires at least 40 passes before calling a null result, this is a partial result, not a completed WSP study.

## Audit notes (added by the orchestrator after logs/audit_phase3.md, 05:47 CST)
- Registration preceded the first edited test run (REGISTERED.txt written 05:25:49; the fp17 test job started 05:26:02), but the git commit of REGISTERED.txt (303b9e7) came at 05:37:14, after that run, because only the orchestrator commits. The order rests on the file timestamp, the harness check that refuses an edited test run without REGISTERED.txt, the GPU job log and the ledger (line 24), not on git. The auditor marked item 9 FAIL on the literal criterion.
- After registration and both test runs, the registered checkpoint was run once on the 500-clip dev split through the harness (eval_runs.jsonl line 10, 05:40:39) to export `transcripts_whisper-tiny-wsp.csv` for Phase 2/5. That run put the 250 practice clips through the model outside the forward-pass counter; it could not influence any edit or the selection, and it is not one of the 17 counted passes.

## Files
- splits: `splits/` (manifest with SHA-256)
- harness: `wsp/eval_harness.py`, `wsp/HARNESS_LOCK.txt`
- forward-pass counter: `wsp/fp_counter.py` (log in `logs/forward_passes.jsonl`)
- patches and rebuild: `edits/patch_fp10.safetensors`, `edits/patch_fp17.safetensors`, `edits/edits_fp*.json`, `wsp/apply_edits.py`
- edit ledger: `edits/edit_log.jsonl`
- diagnostics: `results/layer_audit.csv`, `wsp/diag/`
- selection: `results/REGISTERED.txt`
- per-clip results: `results/eval_*_test.csv`, `results/cardiff_*_test.csv`
- edit locations: `results/edit_locations.csv`
- transcripts of the registered checkpoint (test + dev): `results/transcripts_whisper-tiny-wsp.csv`

Practice and holdout numbers appear only in the ledger and the log, never as results.
