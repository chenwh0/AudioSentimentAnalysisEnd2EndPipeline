# Phase 3 audit (WSP-Whisper, minimal scope): fresh-context auditor

Audit time: 2026-11-18 05:42–05:55 CST. Auditor scripts live in the session scratchpad (i1.py, i2.py, i5.py, i8.py). This audit modified no file except this one. It made no GPU use: item 8 was recomputed on CPU from the saved CSVs, so no forward pass ran and no practice clip went through the model. No load-pool audio was read.
R = AudioSentimentAnalysisEnd2EndPipeline, W = R/Milestone2/WSP_Whisper, PE = R/Milestone2/PipelineEvaluation. Checkpoints that exist: fp10 and fp17. fp20, fp25 and fp50 do not exist because the run stopped at 17 passes.

## Item 1 — PASS
Commands: I loaded PE/development.csv and sorted it by original_index. For each label in alphabetical order I ran `rng.permutation` on that label's row positions, with one `default_rng(8001)` shared across labels, and sent the first floor(n/2) to practice.
- Label counts: Negative 24→12, Neutral 368→184, Positive 108→54. That gives 250 practice and 250 holdout clips, which matches the manifest.
- Recomputed practice and holdout arrays equal the saved .npy files **element for element, in the same order**.
- sha256: wsp_practice_ids.npy d6ec781a…3ed3f1 and wsp_holdout_ids.npy 1b743fbb…44f8c0. Both equal manifest.json.
- Disjointness, with test_indices (1500) and load_pool_indices (1426) both in original_index space: P∩H, P∩T, H∩T, P∩LP, H∩LP and T∩LP are all 0. I confirmed that set(test.csv.original_index) equals test_indices.
Verdict: the splits reproduce exactly from the seed and are disjoint from test and the load pool.

## Item 2 — PASS
- logs/forward_passes.jsonl has 17 lines, all with worker W1 and pass_id 1–17. That is within ≤20 and ≤50. n_clips is 64, 12 or 58, all ≤64.
- The log stores only `clip_ids_sha256`, which is sha256 of the sorted IDs joined by commas (fp_counter._log). I rebuilt candidate ID sets from wsp/diag (practice_order chunks, probe_clips.csv, M_E*.csv, gen_*.csv) and hashed them. **All 17 hashes matched a set that is a subset of the practice IDs:**
  - passes 1, 5–7, 10–12, 14 and 15 match practice_order[0:64], the set M.
  - passes 2 and 3 match practice_order[64:128] and [128:192].
  - pass 4 matches probe_clips (64).
  - passes 8, 9 and 13 match the 12 E5 measure_ids.
  - passes 16 and 17 match practice_order[192:250] (58).
- Checkpoints claimed and present: edits/patch_fp10.safetensors + edits_fp10.json (passes_used 10; pass 10 was the cumulative E1+E2+E5 re-baseline) and patch_fp17.safetensors + edits_fp17.json (passes_used 17). There are no other fp* checkpoints, and none is claimed.
- Observation (not a failure of this item): logs/eval_runs.jsonl line 10 is a harness run of fp17 on split `dev` at 05:40:39 (n=500). That covers the 250 practice clips, outside the counter. It is logged, came after registration and after both test runs, and was used only to write transcripts for Phase 2/5 (gpu_jobs.jsonl note, W1_wsp.md l.44). It cannot have influenced selection.
Verdict: the pass log is within budget, practice-only, and consistent with the saved checkpoints.

## Item 3 — PASS (one listed exception, not a Whisper call)
Command: `grep -rnE 'generate\(|model\(|pipeline\(|forward\(' W/` (excluding csv). Hits:
- wsp/fp_counter.py:67 `model(...)` (teacher_forced_pass) and :83 `pipeline("automatic-speech-recognition", …)` (run_pass). Both are inside the counter and come after `_log`.
- wsp/eval_harness.py:82 `pipeline("automatic-speech-recognition", …)` is the harness.
- **Exception:** wsp/final_results.py:18 `pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)`. This is the Cardiff text classifier on CPU, applied to saved test transcripts. It does not call Whisper and runs no audio.
- Other scripts that use a model (diag.py, probe.py, edit_step.py, gen_check.py) call only FP.run_pass or FP.teacher_forced_pass. post.py and apply_edits.py only build or load models. None of them contains a `.generate` call.
Verdict: every Whisper call site is in the counter or the harness.

## Item 4 — PASS
Command: `grep -rnE 'backward\(|torch\.optim|Trainer|requires_grad_\(True\)|LoRA|peft' W/` and `grep -rli 'lora|peft' W/`. **Zero hits.**
Verdict: no gradient training, optimizer, Trainer or adapter code.

## Item 5 — PASS
Commands: `AE.build('fp17')` and `AE.build('fp10')` on CPU. I compared each state_dict tensor by tensor against `AE.load_base()`, and also against the raw snapshot model.safetensors (blob 7ebd0e69…, rev 169d4a43…) loaded with safetensors, independently of transformers.
- fp17 changes only `model.decoder.embed_tokens.weight`: 165,888 entries. `proj_out.weight` shows the same 165,888 because it is tied (same data_ptr; it is absent from the raw file). The raw-file diff contains only embed_tokens.
- Unique parameters: 37,760,640. 165,888 / 37,760,640 = **0.4393%**, which matches REGISTERED.txt, edits_fp17.json and wsp_results (165,888, 0.439%).
- Changed rows: 432 = 426 digit rows + 6 cause/gonna/gotta rows. Each changed row equals base row × the factor in edits_fp17.json (0 mismatches, rtol 1e-6).
- fp10: 168,960 entries (0.4475%) across 440 rows, also exact against edits_fp10.json. This matches the 168,960 (0.447%) reported.
Verdict: the checkpoints differ from the base only in their patch entries, and the counts match the reports.

## Item 6 — PASS
- The harness and apply_edits load the tokenizer, processor and generation_config directly from the HF snapshot, through `from_pretrained("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")` with HF_HUB_OFFLINE=1 (gpu_run.sh). No copy of any config, tokenizer or processor file exists under W/ (find for *.json and *.txt outside edits/diag/splits returns only REGISTERED.txt and HARNESS_LOCK.txt).
- Snapshot integrity: for each non-LFS file I checked that git-blob sha1(`blob <len>\0`+content) equals its ../../blobs name (the Hub ETag), and for model.safetensors that sha256 equals the blob name. **All 12 match:** README.md, added_tokens.json, config.json, generation_config.json, merges.txt, model.safetensors, normalizer.json, preprocessor_config.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json.
- apply_edits writes only parameter values. It does not touch generation_config or the processor.
Verdict: the harness uses unmodified base files.

## Item 7 — PASS
- `sha256sum W/wsp/eval_harness.py` gives fdfad7215076d5afcea57b1b83a0feb6c645d11991a9a9d89afb68ecea0d7a5d, which equals HARNESS_LOCK.txt. The committed versions in 016d798, 303b9e7 and febbbe2 hash the same. The file is read-only (r--r--r--), mtime 04:35:58, and the lock was written at 04:36:42.
- eval_runs.jsonl has 10 lines. Line 1 (base holdout, 04:36:35) predates the lock by 7 s. It is the smoke/eval run that the lock text names ("after one base holdout smoke/eval run"), so it is explained. Lines 2–10 (04:38:59 … 05:40:39) all postdate the lock.
Verdict: the harness is locked and unchanged, and all runs postdate the lock or are explained.

## Item 8 — PASS (CPU only; no GPU needed)
Commands (i8.py): I re-joined gold from T1_final/data/slue_voxceleb_test.tsv (`normalized_text` on file id) independently of the CSV gold column, and checked the CSV gold column against it (identical after normalization). I normalized both sides with T1 `pipeline.common.normalize_text` and computed per-clip S+D+I with jiwer. The recomputed err_gold and nref_gold equal the CSV columns exactly. I ran 1500 clips with no empty references. Corpus WER is summed errors over summed gold words, and I cross-checked it with jiwer.wer. The bootstrap is `default_rng(8001).integers(0,1500,(10000,1500))` with the same indices for every model, using 2.5/97.5 percentiles.

| ckpt | errs/words | WER (recomputed = reported) | WER CI | ΔWER vs base | ΔCI |
|---|---|---|---|---|---|
| base | 8874/40224 | 0.220615 | [0.193670, 0.255304] | – | – |
| fp10 | 14350/40224 | 0.356752 | [0.279328, 0.443643] | +0.136138 | [+0.064365, +0.214369] |
| **fp17 (registered)** | 8657/40224 | 0.215220 | [0.187346, 0.250866] | −0.005395 | [−0.007944, −0.002642] |

The maximum absolute difference from wsp_results.csv is 9e-17 (float noise).
Verdict: WER and the paired-bootstrap CIs reproduce exactly. The fp17 ΔCI excludes 0.

## Item 9 — FAIL (literal git criterion); substantive ordering is supported and disclosed
Timestamps:
- REGISTERED.txt internal `registered_at`: **2026-11-18T05:25:49-06:00** (holdout run 05:25:33 is eval_runs line 7). File mtime is 05:26:02.70, 13 s after registered_at.
- First eval_runs line with split=test and checkpoint≠base: line 8, fp17 test, record ts **05:28:36**. This is written at the end of the run. The gpu_jobs.jsonl start of W1_eval_fp17_test is **05:26:02**.
- `git log --format=%ci -- …/REGISTERED.txt`: **2026-11-18 05:37:14 -0600** (commit 303b9e7, the only commit touching the file). The committed content is byte-identical to the current file.
- logs/ledger.md line 24 (05:37:15) records the lag and its cause: the orchestrator is the only committer. W1_wsp.md l.40 and l.55 say registration came at 05:25:49, before any edited test run.

Reading: the commit came 8.6 min after the first edited test run ended and 11 min after it started, so the brief's literal check ("committed before the first edited test run") is not met. The ordering of registration before test rests on non-git evidence:
- the internal timestamp;
- the harness guard, which refuses an edited test run unless REGISTERED.txt exists;
- the gpu_jobs start time;
- the ledger and W1 log.
That evidence is consistent but mutable, and the file mtime (05:26:02.70) shows the file was last written in the same second the test job started. I find no sign that the content changed, since the committed file equals the current one and selection follows the holdout ranking. Even so, git does not independently prove that the registration preceded the test.

## Summary

| Item | Check | Result |
|---|---|---|
| 1 | Splits reproduce from seed 8001, SHA-256s, disjointness | PASS |
| 2 | ≤50 passes (17), n_clips ≤64, practice-only (17/17 hashes matched), checkpoints fp10/fp17 present | PASS |
| 3 | Whisper calls only in counter or harness (exception: Cardiff text pipeline in final_results.py) | PASS |
| 4 | No backward, optim, Trainer, requires_grad_(True), LoRA or peft | PASS |
| 5 | Rebuilt fp17 (and fp10) differ only in patch rows; 165,888 = 0.439% | PASS |
| 6 | Snapshot tokenizer, processor and generation_config unmodified (12/12 blob hashes); loaded directly from snapshot | PASS |
| 7 | Harness sha matches lock; runs postdate lock (line 1 explained) | PASS |
| 8 | WER and paired-bootstrap CIs reproduce exactly | PASS |
| 9 | REGISTERED.txt committed before first edited test run | FAIL (literal; disclosed in ledger) |

## Failed items and smallest fix
- **Item 9:** The past ordering cannot be fixed now. Smallest fix: add one sentence to W/results/wsp_results.md stating that registration (05:25:49, file timestamp, harness guard, gpu_jobs start 05:26:02) preceded the first edited test run, but the git commit of REGISTERED.txt (303b9e7, 05:37:14) came after it, and point to ledger line 24. For future phases, the orchestrator should commit REGISTERED.txt, or have the harness log its sha256 in eval_runs.jsonl, before any edited test run is launched.
- Non-failing observation (item 2): the post-registration fp17 `dev` harness run (eval_runs line 10) pushed the 250 practice clips through the model outside the counter. It is harmless to selection, but it should be named in wsp_results.md next to the "17 forward passes" budget statement.
