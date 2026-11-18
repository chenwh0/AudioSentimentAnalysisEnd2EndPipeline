Milestone 2 overnight pipeline

The run went from 04:27 to 08:20 CST on 2026-11-18, on the local RTX 4000 Ada. Branch `m2-overnight` starts from `main` at `e2a81e4` and its pushed tip is `77f7fdb`. Each box carries the status recorded in `OVERNIGHT_STATUS.md` and the ledger, not the status the brief asked for.

● DONE ◔ PARTIAL ○ NOT RUN H1 H2 H3hypothesis the step tests Solid arrows carry files. The dashed arrow is the one data path reserved for the course VM.

Four inputs feed Phase 1. Phase 1 feeds Phases 2, 3 and 4. Phase 3 sends its edited model back into the Phase 2 grid and on to Phase 4, and Phases 2 and 4 feed Phase 5, which pushed the branch. The course-VM tasks start from the pushed branch and are the only place the load pool gets read. T3 to T7 carry the titles the brief and the overnight report use. The Milestone 2 PDF has the full wording.

## Phase 1 · Finish T1

● DONEfinished 05:02 CST · audit 8 of 9, item 9 fixed

**In:** WAVs, saved splits, SLUE transcripts, teammate transcript caches. **Out:** `Milestone2/T1_final/results/` (WER tables, assertions, gap CIs, predictions), `pipeline/` modules, `run_t1.py`, `T1_REPORT.md`.

| Step | Inputs → outputs | Tests | Status |
| --- | --- | --- | --- |
| 1.1 Gold transcripts | SLUE tsv + slices → join 3,426/3,426, 0 label disagreements, span check (25.5% of clips trimmed >0.1 s), `wer_test.csv` vs gold and vs pseudo-reference | H3 supplies gold text | ● DONE |
| 1.2 s2t loop bug | 48-clip dev probe, 4 conditions → the cause is batch padding. Batch 1 cuts loops from 244 to 134 and WER from 0.864 to 0.598 | none | ● DONE |
| 1.3 Assertions in code | dev scores → `assertions.py` with A1, A2, A3 (t tuned on dev), A4a agreement, A4b AADD, plus `assertions_test.csv`, `thresholds.json`, `correlations.json` | A3 is dead for the binary models. Pass rate does not rank configs | ● DONE |
| 1.4 Mappings, baseline, ablation | full / normalized ASR / gold / pseudo text → `m2_t1_results.csv`, `gap_ci.csv` (gold raises Cardiff macro-F1 by only 0.02–0.03, to 0.637), majority baseline | H3 | ● DONE |
| 1.5 Entry point + audit | `run_t1.py --config` regenerates every table from the caches. The auditor reran it and matched every number | none | ● DONE |

## Phase 2 · T2 variants and registry

● DONEfinished 05:04 CST, wsp row added after Phase 3

**In:** Phase 1 caches and modules, Hub model files, Phase 3 wsp transcripts. **Out:** `Milestone2/T2_variants/registry/`, `results/` (24-config grid, `size_vs_accuracy.csv`), `T2_REPORT.md`.

| Step | Inputs → outputs | Tests | Status |
| --- | --- | --- | --- |
| 2.1 Transcribe variants | whisper-tiny, whisper-base, AventIQ, s2t (batch 1), moonshine-base + whisper-tiny-wsp → transcript caches. Moonshine has the best WER (0.161) and the lowest Cardiff F1 (0.603) | none | ● DONE |
| 2.2 Grid 6 × 4 | transcripts × Cardiff / tabularisai / SiEBERT / DistilBERT → `m2_t1_results.csv` with bootstrap CIs, `gap_ci.csv` for all 24 | H3 every ASR pair overlaps for Cardiff | ● DONE |
| 2.3 Register variants | Hub snapshots → `meta.json` per variant: SHA revision, weight sha256, size of loaded files vs all files, offline accuracy | AventIQ is a whisper-base fine-tune (weights differ ≤ 0.0023) | ● DONE |
| 2.4 MinIO upload | registry → object store | none | ○ NOT RUN server unreachable, recorded |

## Phase 3 · WSP-Whisper

◔ PARTIALpass cap lowered to 20 for the 08:20 deadline · audit 8 of 9

**In:** whisper-tiny at pinned SHA, dev 500 split into practice / holdout, gold transcripts. **Out:** `Milestone2/WSP_Whisper/` splits, locked harness, `edits/patch_fp10|fp17.safetensors`, `results/wsp_results.csv`, `transcripts_whisper-tiny-wsp.csv`.

| Step | Inputs → outputs | Tests | Status |
| --- | --- | --- | --- |
| 3.1 Practice / holdout split | dev 500, `default_rng(8001)`, label-stratified → 250 / 250 with SHA-256 manifest, disjoint from test and load pool | none | ● DONE |
| 3.2 Locked harness + base run | notebook settings → `eval_harness.py`, `HARNESS_LOCK.txt`. The base test WER reproduces the teammate's 0.1988 exactly | none | ● DONE |
| 3.3 Layer audit | practice activations + weight stats → `layer_audit.csv`, tied embedding detected | none | ● DONE |
| 3.4 Edits under fp_counter | hypotheses + practice passes → `edit_log.jsonl`, 17 of 50 passes used. Neither the 3 hypothesis families nor the 40-pass null floor was reached | none | ◔ PARTIAL |
| 3.5 Checkpoints | fp10 and fp17 saved as sparse patches + edits json. fp25 and fp50 never reached | none | ● fp10, fp17 ○ fp25, fp50 |
| 3.6 Register + test eval | lowest holdout WER → fp17 registered (432 embedding rows, 0.44% of weights). Test WER −0.0054 \[−0.0079, −0.0026\], Cardiff F1 −0.0064 \[−0.0119, −0.0016\] | the gain is a style match to human transcripts. The registration commit lagged the first test run by 11 min, disclosed in the ledger | ● DONE |

## Phase 4 · AudioXAI

◔ PARTIALheadline 50 test clips, extended with spare time · audit 7 of 8, fixed

**In:** test clips sampled with `default_rng(8001)` after P1–P3 were committed, whisper-tiny / whisper-base / AventIQ, Cardiff, AADD 3.1.0, gold transcripts, Phase 1 `gap_ci.csv`. **Out:** `audioxai` package + CLI, per-clip `.npz` (gitignored, \~1.9 GB local), `results/aggregate.csv`, `summary.json`, `faithfulness.csv`, `report.html`, `README.md`.

| Step | Inputs → outputs | Tests | Status |
| --- | --- | --- | --- |
| 4.0 Pre-register hypotheses | `HYPOTHESES.md` with refutation criteria, committed 04:49, test unlocked 05:05 | H1 H2 H3 stated | ● DONE |
| 4.1 AADD timeline | all 3,426 WAVs → 12 class probabilities on a 20 ms grid, flag p ≥ 0.5 (about half of all frames flagged on this real audio) | feeds H1, H2, A4b | ● DONE |
| 4.2 Per-clip attribution | log-mel + tokens with timestamps → M1 gradient×input + SmoothGrad, M3 Grad-CAM (time-only), M4 cross-attention (time-only): 300 clips whisper-tiny, 50 whisper-base, 50 AventIQ. The brief asked for 300 on all three | H1 | ◔ PARTIAL |
| 4.3 M2 occlusion | 100 ms × 1/8-mel patches on 100 of the 300 clips → `m2.csv`, M1–M2 agreement | H1 model-agnostic check | ● DONE |
| 4.4 Word errors + sentiment importance | jiwer alignment vs gold placed on the timeline, and Integrated Gradients on Cardiff embeddings projected to audio time | H2 WER in flagged vs clean, sentiment share in clean | ● DONE |
| 4.5 Segment replacement (item 7) | 500 ms windows replaced by noise floor, re-transcribe, re-classify → `e2e_segments.csv` on 10 clips. The brief asked for 100 | end-to-end importance | ◔ PARTIAL |
| 4.6 Aggregate + verdicts | focus ratio 0.991 \[0.911, 1.061\] on n = 34, and 0.994 \[0.966, 1.021\] on n = 214 at 300 clips. Gold-text gain +0.022 \[0.003, 0.042\] | H1 refuted · H2 refuted · H3 refuted by its criterion | ● DONE |
| 4.7 Faithfulness | deletion curves AUC 1.426 vs 1.181 random (pass). Model-randomization ρ ≈ 0.92 (FAIL, reported as a finding) | validity of M1 | ● DONE |
| 4.8 Report + README | self-contained `report.html` (waveform, mel + switchable overlays, AADD track, colored transcript), credits to AADD, Captum, SPES, Whisper | none | ● DONE |

## Phase 5 · Figures, report, audit, push

● DONEmandatory step, tip verified against the remote at 06:34 CST

**In:** every results file from Phases 1–4. **Out:** `figures/` (11 PNG + SVG, specs, manifest), `Milestone2/OVERNIGHT_REPORT.md`, `logs/audit_final.md`, `OVERNIGHT_STATUS.md`, branch `origin/m2-overnight`.

| Step | Inputs → outputs | Tests | Status |
| --- | --- | --- | --- |
| 5.1 Figures | JSON spec per figure naming data file + SHA-256 → matplotlib renders by Opus agents, orchestrator spot-checks in `figures/manifest.md` | none | ● DONE |
| 5.2 Overnight report | one section per phase, result first, then evidence, limits, open items for T3–T7 | none | ● DONE |
| 5.3 Final audit | fresh Opus auditor checks every number: pass 1 7 of 101 mismatched, pass 2 5 of 134, all text-side, all fixed | none | ● DONE |
| 5.4 Push | pre-push guard (no main, no force, no weights > 50 MB, no secrets) → `77f7fdb60ef2119edd7e4d293e11e48ffe291cef`. No PR opened | none | ● DONE |

## Course VM · T3 to T7

○ NOT RUNout of scope by the brief: system installs and CPU-side latency belong on the VM

**In:** the pushed branch. **Out:** nothing yet. Task titles below follow the brief and the overnight report's open-items list. Check the Milestone 2 PDF for the exact grading wording.

| Task | What it needs from tonight's run | Tests | Status |
| --- | --- | --- | --- |
| T3 Services | import `pipeline/assertions.py` and load each variant by registry path and pinned revision | none | ○ NOT RUN |
| T4 Deployment | containers / Kubernetes. Each `meta.json` lists the files to copy for its variant | none | ○ NOT RUN |
| T5 Load generator | the 1,426-clip load pool, untouched tonight. Serve s2t at batch 1 or with length-capped decoding | none | ○ NOT RUN |
| T6 SLA profiling | latency on the VM's CPUs only. Tonight's timing columns are GPU batch averages and make no SLA claim | none | ○ NOT RUN |
| T7 Analysis | the gold-input ablation is already computed for every configuration in `T2_variants/results/gap_ci.csv` | H3 input ready | ○ NOT RUN data ready |
