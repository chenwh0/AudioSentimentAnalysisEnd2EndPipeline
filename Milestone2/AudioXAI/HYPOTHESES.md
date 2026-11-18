# AudioXAI hypotheses (written 2026-11-18 ~04:37 CST, before any test-slice clip is read)

Question: why does the choice of ASR barely change sentiment results on SLUE-VoxCeleb?
Scope tonight: ASR = openai/whisper-tiny (rev 169d4a4341b33bc18d8881c4b69c2e104e1cc0af),
sentiment = cardiffnlp/twitter-roberta-base-sentiment-latest (rev 3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7),
reported analysis on a 50-clip stratified test sample (default_rng(8001), proportional allocation by label,
>= 3 per class, labels processed alphabetically, rng.permutation within label after sorting test.csv by original_index).
Method development used dev clips only.

Common definitions
- Frame grid: AADD 20 ms segments (aadd --segments --resolution 0.02), cropped to the clip duration.
  A 20 ms frame is "flagged" when any of the 12 AADD class probabilities is >= 0.5.
  Clip-level flag (reported, AADD published default): per-class 95th percentile >= 0.5.
  Clip degradation class = argmax over classes of the per-class 95th percentile.
  AADD was trained on synthetic LibriSpeech degradations; on real VoxCeleb audio its outputs are estimates.
- ASR attribution (M1): gradient x input on Whisper's log-mel, SmoothGrad n=8, sigma = 0.1 x feature std,
  target = teacher-forced log-prob of each generated token given the model's own previous tokens.
  Per-token map |attr| summed over mel bins -> time profile at 10 ms, resampled to 20 ms, normalized to sum 1
  over the unpadded clip; clip map = mean over generated text tokens.
- Focus ratio per clip FR = (attribution mass in flagged frames) / (fraction of frames flagged).
  Clips with zero flagged frames get FR = NaN; they are counted and reported, not silently dropped.
- Bootstrap: 10,000 resamples of clips, numpy.random.default_rng(8001), percentile 95% CI.

H1 (the user's hypothesis). Whisper-tiny's attribution concentrates in AADD-flagged regions more than
their share of duration.
- Statistic: mean FR over clips with >= 1 flagged frame, with bootstrap 95% CI. Also the median and the
  per-degradation-class means (descriptive, small n).
- Holds if the 95% CI of mean FR lies entirely above 1.
- Refuted if the CI includes 1 or lies below 1.
- Secondary (reported, not a verdict): point-biserial correlation of FR with Cardiff correctness, bootstrap CI.

H2. ASR errors concentrate in flagged segments and fall on words with low sentiment importance; sentiment-bearing
words sit in clean audio and survive ASR.
- Word errors: jiwer.process_words on normalize_text(gold) vs normalize_text(hypothesis); substitutions and
  insertions are placed at the hypothesis word's token-timestamp span; deletions at the gap between neighbouring
  hypothesis words. A word is "in flagged audio" if >= 50% of its 20 ms frames are flagged.
- Statistic (a): pooled error rate inside flagged vs outside: errors / reference-aligned words in each region,
  and the paired clip-level difference (inside - outside) with bootstrap CI over clips that have words in both.
- Statistic (b): sentiment-importance share of erroneous hypothesis words (|IG| mass of S/I words / total |IG|)
  vs their word share (count of S/I words / total hypothesis words), paired per clip, bootstrap CI of the
  difference (importance share - word share).
- Holds only if both: error rate inside flagged > outside (CI of difference above 0) AND importance share of
  erroneous words < their word share (CI of difference below 0).
- Refuted if error rate inside flagged <= outside, or if the importance share of erroneous words >= their word share.
  If a CI includes 0 the corresponding half is "not supported".

H3. The bottleneck is the text classifier, not the transcript.
- Decided by Phase 1's gold-text ablation (Milestone2/T1_final results, human SLUE gold transcripts):
  Cardiff macro-F1 on gold text vs on whisper-tiny text over the test slice, paired bootstrap CI of the difference.
- Holds if gold text does not raise Cardiff macro-F1 beyond the paired CI (CI of gold - ASR includes 0 or is below 0).
- Refuted if the paired CI of (gold - ASR) macro-F1 lies entirely above 0.
- On the 50-clip sample AudioXAI also reports Cardiff accuracy on gold vs ASR text (descriptive only).

Faithfulness (prerequisite for trusting H1)
- Deletion: for >= 20 clips, zero (set to the clip's log-mel minimum / noise floor) the top-k% M1 patches
  (100 ms x 10 mel bins), k in {5,10,...,50}, vs random patches of the same count; mean drop in teacher-forced
  sequence log-prob; AUC of both curves. M1 is faithful if AUC_top > AUC_random.
- Model randomization (Adebayo et al. 2018): re-initialize the top 2 decoder layers of a copy, recompute M1
  (same tokens, teacher-forced); Spearman rho between original and randomized time-frequency maps, >= 10 clips.
  PASS only if mean rho < 0.5. Failures are reported as findings.

No test-slice finding changes any reported configuration tonight; any suggested change becomes a recommendation.

## Addendum (2026-11-18 ~04:57 CST, still before TEST_UNLOCKED; based on dev clips only)
Dev observation (AADD orchestrator file, 500 dev clips, 20 ms frames): 54% of frames are flagged under the
any-class >= 0.5 rule; 27% of clips have no flagged frame and 21% have every frame flagged. A clip with every frame
flagged has focus ratio exactly 1 by construction, so it carries no information about H1.
- The primary H1 statistic and verdict stay exactly as defined above (all clips with >= 1 flagged frame).
- Secondary, reported alongside (not a separate verdict): mean focus ratio with bootstrap CI over "informative"
  clips, 0 < fraction flagged < 1; plus the per-clip Spearman correlation between the M1 time profile and the
  per-frame max AADD probability (graded, no threshold).
- Counts of zero-flag and all-flag clips are reported.
