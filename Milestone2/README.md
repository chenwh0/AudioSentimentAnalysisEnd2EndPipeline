# Milestone 2 on branch `m2-overnight`

This folder holds the overnight run of 2026-11-18 (04:27 to 08:20 CST, one local RTX 4000 Ada). It finishes T1, registers the T2 variants, runs a reduced WSP-Whisper study and a reduced AudioXAI study, and leaves T3 to T7 for the course VM. This page is the short way through it: the theory behind each measurement, then every figure with its reading. If you want one file only, read this one.

Where the rest lives:

- `OVERNIGHT_REPORT.md`: one section per phase, result first, then evidence and limits.
- `../OVERNIGHT_STATUS.md`: the verdict line, the commit hashes, what was not done.
- `../logs/ledger.md`: every decision with its clock time. `../logs/audit_*.md`: what the fresh-context auditors checked.
- `T1_final/`, `T2_variants/`, `WSP_Whisper/`, `AudioXAI/`: code, caches and results per phase. `AudioXAI/README.md` has the tool's usage and its full result list.
- `../figures/`: the 11 figures below, each with a JSON spec naming its data file and that file's SHA-256.

## Theory

### The pipeline and the question

Each clip goes through two models. A speech-to-text model (ASR) turns the audio into a transcript, and a text classifier turns the transcript into negative, neutral or positive. The project question is why swapping the ASR model barely changes the sentiment result. Three answers were possible: the ASR models make the same errors, the classifier ignores the words the ASR gets wrong, or the classifier is the weak link and better text would not help it. The gold-input ablation in Phase 1 and the AudioXAI study in Phase 4 are built to tell these apart.

### Splits and leakage

The 3,426 clips were split once by Chen's notebook and never again: 500 development clips, 1,500 test clips, and a 1,426-clip load pool. Every threshold, label mapping and model choice was tuned on the 500 development clips. Every number reported here comes from the 1,500 test clips. The load pool was never read, because T5's load generator on the course VM needs it untouched.

### Word error rate

WER counts substitutions, deletions and insertions between a hypothesis and a reference, divided by the number of reference words:

$$
\mathrm{WER} = \frac{S + D + I}{N}
$$

It is computed over the whole corpus, so long clips weigh more than short ones. Both texts go through the notebook's `normalize_text` first (lowercase, punctuation removed). Two references are used. The gold reference is the human transcript from the SLUE benchmark, joined to all 3,426 clips with no label disagreement. Against that stands the pseudo reference, the whisper-large-v3 machine transcript the earlier notebook used in place of truth. Being itself 0.179 WER away from the humans, it measures agreement with another model, not accuracy. A filler-free WER (um, uh and friends removed from both sides) is reported as a secondary number.

### Macro-F1 and the baseline

Accuracy rewards a model that always says "neutral", because 73.6% of the test clips are neutral. Macro-F1 averages the F1 score of the three classes with equal weight, so the 2 rarer classes count as much as the common one. The always-neutral baseline scores 0.7360 accuracy but only 0.2826 macro-F1. Every configuration is judged against that 0.2826.

### Confidence intervals

Every CI is a percentile bootstrap: 10,000 resamples of clip indices drawn with `numpy.random.default_rng(8001)`. When two models are compared, the same resampled indices are used for both and the gap is recomputed on each resample. A paired CI that excludes zero means the two models differ on the same clips, which is a stronger claim than two overlapping single-model CIs.

### Assertions

An assertion is a check that runs on a single request without a label, so a service can run it in production. A1 asks for at least 2 words, and A2 for a speaking rate between 0.5 and 5.0 words per second. A3 wants a classifier score at or above a threshold t, where t is chosen on the development clips by balanced accuracy at predicting whether the label is correct. Two new ones follow: A4a, that a second classifier agree with the first, and A4b, that the acoustic degradation detector (AADD) does not flag the clip. Thresholds are tuned per configuration, which has a consequence shown in the figures: pass rate does not rank configurations.

### Gold-input ablation

For each ASR and sentiment pair, the classifier is run on four texts: the raw ASR transcript (full), the same transcript lowercased without punctuation (norm), the human transcript (gold) and the machine pseudo reference (pseudo). The gold minus full gap, with its paired CI, measures how much a perfect transcript would help. It is the direct test of hypothesis H3.

### Variant registry

Each ASR and sentiment variant is recorded with its Hugging Face revision SHA, the SHA-256 of its weight file, the size of the files a service would load (not the sum of every weight format on the Hub) and its test-slice accuracy. The registry is what T3 and T4 load from.

### Weight-space programming (WSP)

WSP edits a model without training. There is no gradient, no optimizer and no search over values. A person loads the weights, reasons about what a tensor computes, writes a hypothesis, changes specific entries and measures the change with a forward pass on practice clips. Each forward pass is a batch of at most 64 practice clips and the budget is 50. Forward passes use only the practice half of the development clips. The holdout half is touched only by a locked evaluation harness whose SHA-256 is pinned before the first edit. The checkpoint with the lowest holdout WER is registered before any edited model sees the test slice. Edits are stored as sparse patches, so the full model is rebuilt from the base weights plus the patch. Whisper-tiny ties its token embedding to its output projection, so an edited embedding row changes both how a token is read and how likely it is to be emitted.

### AudioXAI

The tool asks where in the audio the ASR model looks, and whether that is where the audio is degraded.

- Input: the exact 80-bin log-mel spectrogram the Whisper feature extractor produces (10 ms hop).
- Degradation timeline: AADD 3.1.0 gives 12 class probabilities every 20 ms. A frame is flagged when any class probability reaches 0.5. AADD was trained on synthetic degradations of read speech, so its flags on real interview audio are estimates.
- M1, the primary map: gradient × input on the log-mel for each generated token's teacher-forced log-probability, smoothed with SmoothGrad (8 noisy copies, σ = 0.1 × feature std, where the brief asked for 16 and the code and HYPOTHESES.md use 8). It gives a time × frequency map per token.
- M2, model-agnostic: occlude 100 ms × 1/8-mel patches and measure the drop in each token's log-probability, in the spirit of SPES.
- M3, Grad-CAM on the last encoder layer. Time-only, because Whisper's Conv1d front end leaves no frequency axis.
- M4, decoder cross-attention averaged over the alignment heads. Also time-only.
- Word errors against gold from jiwer's alignment, placed on the timeline through token timestamps.
- Sentiment importance: Integrated Gradients on the classifier's embeddings, projected back onto audio time through the word timestamps.

The focus ratio is the attribution mass inside flagged frames divided by the fraction of frames that are flagged. A ratio of 1 means the model attends to degraded audio exactly in proportion to how much of the clip is degraded. Above 1 means it concentrates there.

Two faithfulness checks guard the maps. Deletion curves remove the top k% of patches by attribution and compare the drop in log-probability with removing random patches of the same size. The model-randomization check (Adebayo et al., 2018) re-initializes the top decoder layers and re-runs M1. If the maps barely change, the method is reading the input, not the model.

### The three hypotheses

They were written to `AudioXAI/HYPOTHESES.md` and committed at 04:49, before the test slice was unlocked at 05:05.

- H1 (the project's own guess): ASR attribution concentrates in degraded regions more than their share of the clip. Refuted if the mean focus ratio's CI includes 1.
- H2: ASR errors concentrate in degraded segments and land on words that carry little sentiment. Refuted if the error rate in flagged audio is not higher than in clean audio, or if erroneous words carry their share of sentiment importance.
- H3: the bottleneck is the text classifier, so gold text would not raise macro-F1 beyond its CI. Refuted if the gold minus ASR gap's CI lies above zero.

## Results

Headline numbers from the test slice. Every figure below links to its data file through `../figures/specs/`.

| What | Value |
|---|---|
| Best configuration, Cardiff on AventIQ, macro-F1 | 0.6211 [0.5926, 0.6493] |
| Same classifier on human gold text | 0.6371 |
| Always-neutral baseline macro-F1 | 0.2826 |
| WER vs humans: moonshine / whisper-base / AventIQ / whisper-tiny / s2t | 0.1612 / 0.1817 / 0.1831 / 0.2206 / 0.5983 |
| WSP fp17, change in test WER | −0.0054 [−0.0079, −0.0026] |
| WSP fp17, change in Cardiff macro-F1 | −0.0064 [−0.0119, −0.0016] |
| H1 focus ratio, 300 clips, whisper-tiny | 0.994 [0.966, 1.021] |
| H3 gold minus ASR macro-F1, Cardiff on whisper-tiny | +0.022 [0.003, 0.042] |

### Phase 1: the sentiment model decides, the speech model barely matters

![Macro-F1 grid](../figures/t1_macro_f1_grid.png)

Macro-F1 of each ASR × sentiment pair with 95% CIs. Read it by columns: swapping the speech model moves Cardiff by about 0.02, swapping the sentiment model moves it by about 0.3. The worst pair, whisper-tiny with SiEBERT, scores 0.3228, barely above the 0.2826 baseline, because a binary model cannot produce the neutral class that 73.6% of clips carry.

![Gold-input ablation](../figures/t1_full_norm_gold.png)

The same classifier on the ASR transcript (full), the normalized transcript (norm) and the human transcript (gold). Normalization changes almost nothing. Human text adds at most 0.034 macro-F1 for Cardiff (s2t: 0.6032 to 0.6371), and the gap for whisper-tiny is +0.022 with a CI above zero. That small gain is the H3 result: the classifier on perfect text tops out at 0.637, so better transcripts cannot move the pipeline much.

![Pass rate against macro-F1](../figures/t1_passrate_vs_f1.png)

Assertion pass rate against macro-F1 over 12 configurations. Left uses A1 to A3 (Pearson −0.24), right adds classifier agreement A4a (Pearson −0.08). Each configuration's threshold is tuned on its own development scores, so a high pass rate says the thresholds are lenient, not that the configuration is good. Pass rate is a per-request guard, not a model selector. A3 is useless for the two binary classifiers, whose scores sit at 0.99 on almost every clip.

Not in a figure: the s2t model's reported 0.93 WER was partly a measurement artefact. Transcribing one clip at a time instead of eight cut looping outputs from 244 to 134 of 1,500 and WER against humans from 0.864 to 0.598. The cause is batch padding.

### Phase 2: six speech models, four classifiers, one registry

![WER per variant](../figures/t2_wer_per_variant.png)

WER against human transcripts (solid), without fillers (light), and against the whisper-large-v3 pseudo reference (hatched). Moonshine-base is the most accurate against humans at 0.1612. The three Whisper variants sit between 0.1817 and 0.2206, and the edited whisper-tiny-wsp row is at 0.2152. Note the dashed line at 0.179: that is how far the pseudo reference itself is from the humans, which is why the earlier WER numbers against it ranked models differently.

![Accuracy against size](../figures/t2_accuracy_vs_size.png)

Cardiff macro-F1 against model size for the five ASR variants. Moonshine-base has the lowest WER and the lowest macro-F1 (0.6027). AventIQ at 147 MB is a fine-tune of whisper-base at 295 MB: their weights differ by at most 0.0023 and their macro-F1 by 0.003. Size buys nothing here because the classifier, not the transcript, sets the ceiling.

### Phase 3: a real but useless WER gain from 432 embedding rows

![WSP results table](../figures/wsp_results_table.png)

Base whisper-tiny, the failed fp10 checkpoint and the registered fp17, all on the 1,500 test clips through the locked harness. fp17 rescales 426 digit-containing token rows by 0.65 and the `cause`, `gonna`, `gotta` rows by 1.10, so numbers get spelled out the way the human transcripts write them. WER falls by 0.0054 with a paired CI that excludes zero. Cardiff macro-F1 falls by 0.0064, also with a CI that excludes zero. WER against the pseudo reference rises from 0.1988 to 0.2100, which shows the gain is a style match to the human transcripts, not better hearing. fp10 is kept in the table as the failure it was: raising the uh/um rows sent 18 clips into repetition loops.

![Where the edits sit](../figures/wsp_edit_locations.png)

All 165,888 changed weights (0.44% of the model) are rows of the decoder's token-embedding matrix, which shares storage with the output projection. Encoder, attention and feed-forward weights are untouched. Only 17 of the 50 forward passes were used, so fp25 and fp50 do not exist and the study is partial by the brief's own rule (40 passes before any null result).

### Phase 4: the speech model does not stare at the bad audio

![One clip through the tool](../figures/xai_example_clip.png)

One test clip (gold neutral, predicted neutral, WER 0.40, degradation class time-stretch, 29.6% of frames flagged). Top to bottom: waveform, log-mel with the M1 attribution overlay, the AADD timeline, the transcript colored by error type, and which words carried the sentiment decision. The attribution follows the speech, flagged or not.

![Focus ratio by degradation class](../figures/xai_focus_ratio_by_class.png)

Focus ratio per clip, by AADD class, on the 50-clip headline sample. The mean is 0.991 [0.911, 1.061] on the 34 clips that have at least one flagged frame, and the CI includes 1. On the 300-clip extension it is 0.994 [0.966, 1.021] on 214 clips. M2, M3 and M4 agree, and whisper-base and AventIQ give 0.991 on the same 50 clips. H1 is refuted everywhere it was tested. The only classes whose CIs exclude 1 at n = 300 are music (1.060, n = 21) and hum (0.837, n = 12), and those are exploratory.

![WER in flagged against clean audio](../figures/xai_wer_flagged_vs_clean.png)

Per-clip error rate inside flagged audio against the rate in clean audio. On the 50-clip sample the pooled rates are 0.242 flagged against 0.258 clean. At 300 clips the pooled rate is higher in flagged audio (0.343 against 0.223), but within clips that have both kinds of audio the paired difference is negative (−0.126 [−0.275, −0.017], n = 85), and erroneous words carry slightly more than their share of sentiment importance (+0.020 [0.010, 0.032]). Both halves of H2 fail: errors do not hide in degraded audio, and they do not avoid the sentiment-bearing words.

![Deletion curves](../figures/xai_deletion_curves.png)

Removing the patches M1 ranks highest lowers whisper-tiny's confidence faster than removing random patches of the same size: area under the curve 1.426 against 1.181, difference 0.245 [0.187, 0.303], top above random on 19 of 20 clips. The maps carry real signal about the input. They fail the other check: after re-initializing the top two decoder layers, the absolute maps correlate at ρ = 0.916 with the originals, far above the 0.5 rule, and |M1| correlates 0.78 with |input|. Gradient × input here is dominated by the spectrogram's own structure, so the focus-ratio result is a statement about where the energy is, confirmed by the model-agnostic M2 (focus ratio 0.989 [0.917, 1.069] on 100 clips).

### The answer in one paragraph

The speech model barely matters because the classifier is the ceiling. Human transcripts raise Cardiff's macro-F1 by only 0.02 to 0.03, while changing the classifier moves it by about 0.3. The predicted label is the same on ASR text and on gold text for about 92% of clips, and the two Whisper sizes agree with each other on 92% as well. The ASR models do not attend to degraded audio more than its share of the clip (H1), their errors do not concentrate there or avoid the sentiment-bearing words (H2), and perfect text helps only a little (H3, refuted by its own criterion but with a small effect). Making the speech model better, including by weight edits that lower WER, does not move sentiment accuracy.

## What is still open

- T3 to T7 on the course VM: services importing `T1_final/pipeline/assertions.py`, deployment from the registry, the load generator on the untouched load pool, CPU-side latency for the SLA, and the analysis task, whose gold-input ablation is already in `T2_variants/results/gap_ci.csv`.
- WSP: passes 18 to 50, the fp25 and fp50 checkpoints, and the 40-pass floor before a null result can be declared.
- AudioXAI: 300 clips for whisper-base and AventIQ (50 done), M2 and segment replacement on 100 (M2 done on 100, segment replacement on 10).
- MinIO upload of the registry: the server was unreachable.
- The per-clip `.npz` attribution arrays (about 1.9 GB) and the AADD scans are gitignored and exist only on the machine that ran this.

## Per-token attribution: which audio segment each output token used

Added on branch `xai-token-segments` on 2026-11-18, after the overnight run. The question moves from audio quality to alignment: for each token the ASR emits, which 100 ms segment of the clip, and which frequency band, contributed most. The study set is 20 clips, the same 20 for whisper-tiny, whisper-base and AventIQ, reported in two groups so that hard cases are not presented as typical behaviour. Every aggregate number from 10 clips is indicative, not conclusive.

Sampling rule (`Milestone2/AudioXAI/scripts/select_groups.py`, clip ids and strata in `results/token_segments/groups.csv`). The pool is the 50-clip stratified test sample the overnight run used for all three models (`results/test_sample_ids.csv`, drawn by `scan.test_sample(n=50, seed=8001)` from Chen's saved test slice `Milestone2/PipelineEvaluation/test.csv`, 1,500 rows), since it is the only set where all three models have saved attribution arrays. The selector asserts that every pool clip is in that saved test slice and in neither the development slice (500) nor the load pool (1,426, never read), and `results/token_segments/groups_provenance.json` records the SHA-256 of the slice CSV and of the three `*_indices.npy` files, so the 20 clips trace back to the reproducibility splits without any re-splitting. `groups.csv` carries each clip's slice and original index. The hard set is the 10 pool clips with the lowest mean teacher-forced token log-probability under whisper-tiny (uncertainty sampling). The mixed set is 10 clips drawn from the remaining 40 by strata of confidence (above or below the median mean log-probability), clip length (above or below the median duration) and audio quality (AADD class clean against anything else): `numpy.random.default_rng(8001)` permutes each stratum, the 8 strata are visited in fixed alphabetical order taking one clip per visit until 10 are drawn.

Hard set: `id10273_Ed918auNI0Y_00004`, `id10273_E50E5hbggyM_00005`, `id10288_FIxC_PNJNkc_00003`, `id10298_6qFnVechX9o_00012`, `id10287_2iaCNe7ajII_00002`, `id10273_ZpPPaMazpf0_00008`, `id10291_iPZGPcrYIZs_00007`, `id10276_HrA2BLcLApA_00026`, `id10300_8Sz2-IYJ2GA_00013`, `id10271_zWyD72sHVwA_00007` (mean log-probability −3.40 to −0.49, 6 degraded and 4 clean by AADD class, and two of them are clips where whisper-tiny emitted only 1 or 2 tokens, WER 0.95 and 0.97).
Mixed set: `id10273_hW0Q6eHqN-U_00004`, `id10281_1CiBWgsHT4E_00001`, `id10285_uArtiTSTnSU_00014`, `id10309_XMLMvfrgdzY_00009`, `id10276_3vWez3baO60_00009`, `id10273_8fKRoUIpAJY_00012`, `id10287_V0ubcXO71Sg_00001`, `id10304_W-5dGkFnQxM_00001`, `id10287_2iaCNe7ajII_00004`, `id10284_Uzxv7Axh3Z8_00006` (mean log-probability −0.45 to −0.10, with 4 uncertain and 6 confident, 6 long and 4 short, 5 clean and 5 degraded).

Methods. M2 occlusion is the primary method: each 100 ms × 10-mel patch is set to the clip's log-mel floor and the drop in the token's teacher-forced log-probability is recorded. Beside it, M1 (gradient × input with SmoothGrad) appears for comparison only, because it failed the model-randomization check (ρ ≈ 0.92 after re-initializing the top two decoder layers) and tracks input energy (|M1| against |input| correlates at 0.78). M4 cross-attention is not used, since Whisper's token timestamps come from the same alignment heads and the comparison would be circular. The M2 arrays for whisper-base and AventIQ on these 20 clips were computed for this section (`scripts/m2_for_clips.py`, logged in `logs/gpu_jobs.jsonl`). Everything else comes from the saved overnight `.npz` arrays. Code: `scripts/token_segments.py` (tables), `figures/render_xai_tokens_v3.py` (per-clip summary figures, AADD-inspector layout), `figures/render_xai_tokens.py` (the per-token tile figures and the shared loader), `figures/render_xai_token_aggregate.py` (aggregate). Tables: `results/token_segments/tokens.csv`, `summary.json`.

Definitions. A token's time is Whisper's timestamp for it. Its M2 time profile is the positive log-probability drop summed over the 8 bands, per 100 ms patch. The top segment is the patch with the largest drop, the top band the band with the largest drop summed over time. Distance is the gap between the top segment's centre and the token's time. Band shares are each token's positive attribution normalized to 1 and averaged over tokens (no CI). The CIs on distances and fractions are 10,000 bootstrap resamples of clips over per-clip medians, or per-clip means for fractions.

### Per-clip figures

For each of the 20 clips and each model there are two figures in `../figures/xai_tokens/` (120 for the study set, plus the 18 for the three clips of the first pass that are not in the study set, kept as background). `../figures/xai_tokens/MANIFEST.md` lists the arrays each figure read with their SHA-256 and tabulates every token's time, top M2 segment, top M2 band and top M1 segment. Captions name the set the clip belongs to.

- `<model>__<clip>__summary.png`: the headline view, laid out like AADD's own file inspector, one shared time axis. Top to bottom: the waveform coloured per 20 ms frame by AADD's max degradation probability on AADD's blue sequential ramp (light clean, dark degraded, small colorbar), the log-mel in magma, one AADD class row per class that reaches p ≥ 0.5 somewhere in the clip plus a max-p row on the same ramp, a token lane where each output token is a coloured block over its top M2 occlusion segment with its text beside it, the token × time heat strip in greys (rows = tokens, columns = 100 ms patches, cell = that token's M2 scaled to its own maximum, row labels in the token colour), and chips listing the detected degraded intervals (class, start to end, max p). The same token colour outlines the token's top M2 patch on the mel and marks a tick on the waveform, so token and audio are linked by colour with no connector lines. Blues are reserved for audio quality. The token palette (two greens, two golds, a brown, no blue or magma hue) passed the dataviz validator with worst adjacent colorblind ΔE 7.9, which is why the token index is printed in every block. Colours repeat every 5 tokens. M1 is not drawn in this view. Renderer: `figures/render_xai_tokens_v3.py`.
- `<model>__<clip>__tokens.png`: one tile per token, from the first pass. Left half, the log-mel with the M2 occlusion heat at patch resolution and the top patch outlined in orange. Right half, the same with |M1| at full resolution. These were not re-rendered in the redesign.

![Hard set, whisper-tiny, noise clip id10288_FIxC_PNJNkc_00003: AADD-style inspector with the token lane and the token × time heat strip](../figures/xai_tokens/whisper-tiny__id10288_FIxC_PNJNkc_00003__summary.png)

![Mixed set, whisper-tiny, clean clip id10287_2iaCNe7ajII_00004: one tile per token](../figures/xai_tokens/whisper-tiny__id10287_2iaCNe7ajII_00004__tokens.png)

The hard-set example above (WER 0.67, noise, every frame flagged by AADD) shows what "hard" looks like at token level: the heat strip runs along the diagonal for most tokens, each lit at or just after its own timestamp. Two rows break the diagonal. The first token "and" carries the 0.00 s timestamp artefact and its top segment is 0.85 s ahead, and the final "." has its top segment 4.49 s back, spread over the whole clip. The clip's other punctuation sits within 0.4 s of its timestamp, so punctuation as such is not the problem here.

### Aggregate, per group and per model

Ten clips per group, so every number below is indicative, not conclusive. M2 occlusion is the primary method and M1 is the comparison. Same 20 clips for every model.

![Hard set: distance to own timestamp and mel-band share](../figures/xai_tokens/aggregate_distance_bands_hard.png)

Hard set (10 clips, lowest whisper-tiny confidence):

| Model | Tokens | M2 median distance (s) | M2 tokens within 0.3 s | M2 within 0.5 s | M2 top segment's share of the token's attribution | M1 median distance (s) | M1 within 0.3 s | M1 and M2 pick the same top segment |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 175 | 0.30 [0.20, 0.45] | 63% [45, 77] | 72% | 0.27 [0.23, 0.32] | 0.24 [0.17, 0.36] | 66% | 29% [18, 40] |
| whisper-base | 234 | 0.10 [0.07, 0.15] | 82% [73, 89] | 86% | 0.33 [0.27, 0.39] | 0.13 [0.08, 0.21] | 74% | 36% [26, 45] |
| AventIQ | 227 | 0.15 [0.07, 0.27] | 79% [67, 88] | 84% | 0.32 [0.26, 0.37] | 0.13 [0.09, 0.19] | 78% | 34% [28, 41] |

![Mixed set: distance to own timestamp and mel-band share](../figures/xai_tokens/aggregate_distance_bands_mixed.png)

Mixed set (10 clips, stratified, seed 8001):

| Model | Tokens | M2 median distance (s) | M2 tokens within 0.3 s | M2 within 0.5 s | M2 top segment's share of the token's attribution | M1 median distance (s) | M1 within 0.3 s | M1 and M2 pick the same top segment |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 284 | 0.19 [0.17, 0.20] | 77% [74, 81] | 89% | 0.35 [0.31, 0.39] | 0.19 [0.18, 0.21] | 74% | 38% [32, 44] |
| whisper-base | 293 | 0.10 [0.08, 0.13] | 81% [75, 86] | 89% | 0.31 [0.25, 0.36] | 0.17 [0.10, 0.28] | 76% | 33% [26, 39] |
| AventIQ | 301 | 0.10 [0.08, 0.13] | 82% [77, 87] | 90% | 0.27 [0.22, 0.32] | 0.20 [0.12, 0.33] | 73% | 33% [25, 41] |

Mean share of M2 attribution per mel band, hard / mixed: bins 10–19 (about 410–750 Hz) carry the most for every model, 0.24 / 0.27 for whisper-tiny, 0.23 / 0.25 for whisper-base and 0.23 / 0.25 for AventIQ. Bins 0–9 (about 40–370 Hz) come second at 0.15 to 0.21. Bins 20–49 (about 0.8–2.4 kHz) take 0.11 to 0.14 each, and the two bands above 3.7 kHz 0.04 to 0.08 each. M1 instead puts 0.33 to 0.35 on bins 0–9 and 0.20 to 0.21 on bins 10–19 in both groups for every model, which is the profile of a map that follows input energy.

What this says, per model:

- whisper-tiny. In the mixed set three tokens in four draw most on the 100 ms segment at or next to their own timestamp (median gap 0.19 s, 77% within 0.3 s, 89% within 0.5 s). In the hard set that drops to 63% within 0.3 s and the median gap rises to 0.30 s with a wide interval. The hard clips are where the top segment also explains less of the token (0.27 against 0.35), so evidence is more spread out as well as further away.
- whisper-base. Tighter than whisper-tiny in both groups (median gap 0.10 s, 81% to 82% within 0.3 s). Unlike whisper-tiny, it barely moves between the groups: the median gap is the same, 82% against 81% within 0.3 s, and 86% against 89% within 0.5 s. The hard set was selected by whisper-tiny's uncertainty, so this is a statement about whisper-base on clips whisper-tiny found hard, not about whisper-base's own hard cases.
- AventIQ. Within 0.02 of whisper-base in every band share and within 0.05 s of it on median gap in the mixed set, which fits a fine-tune whose weights differ from whisper-base by at most 0.0023. Its hard-set interval is wider (median 0.15 s, CI 0.07 to 0.27).
- Across models. The frequency answer is the same everywhere: tokens are decided in the 410–750 Hz band first and the band below it second, with little weight above 3.7 kHz. M1 and M2 agree on the top segment for only 29% to 38% of tokens even though their distance distributions look alike. M1 lands on the loud frames near the token, M2 on the frames whose removal hurts the token, and those are often a segment apart. That is the randomization failure seen from the other side.

Two things the per-clip figures show that the aggregate hides. First, the first kept token of every clip carries timestamp 0.00 s for every model (all 50 pool clips, all three models), a property of Whisper's token timestamps, not of the attribution, so that token's distance is inflated. Second, the far-off tokens in the hard set are not mainly punctuation. Of the 24 whisper-tiny hard-set tokens whose top M2 segment is more than 0.5 s from their timestamp, 3 are punctuation, 7 are the first token of a clip (the 0.00 s artefact), and 14 are words, among them content words such as "fight" (1.65 s off), "more" (3.03 s), "always" (1.89 s) and "shot" (0.63 s). In the hard set the evidence for a word can sit a second or more away from where Whisper stamps it.

### Background: the full 50-clip pool

The first pass of this work used all 50 pool clips for whisper-tiny and 5 clips for the other two models, then was rescoped to the 20-clip study set above. Those numbers are kept only as context: whisper-tiny M2 on 50 clips gives a median gap of 0.22 s [0.19, 0.26] and 72% of tokens within 0.3 s, and M1 on the same 50 clips gives 73% to 75% within 0.3 s for the three models. With the 23 clips that now have M2 arrays for whisper-base and AventIQ (the 20 study clips plus 3 first-pass clips), both sit at 80% within 0.3 s. The per-clip figures of the three first-pass clips that are not in the study set are kept under the same folder and marked in the manifest.

Limits. Ten clips per group give wide intervals, and the hard set is defined by whisper-tiny, so the other two models' hard-set rows are not their own hard cases. Whisper's timestamps are model output, so "distance to own timestamp" measures agreement between occlusion and the alignment heads, not against a human word alignment. Occlusion patches are 100 ms by 10 mel bins, so sub-patch structure is invisible, and a token whose evidence spans two patches shows a top share near 0.3 by construction.
