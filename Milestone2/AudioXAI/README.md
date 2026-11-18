# AudioXAI

AudioXAI puts four things on one time axis for each clip, so you can see whether an ASR model's evidence, its errors
and the sentiment decision fall in degraded audio:

1. the waveform and the exact log-mel Whisper sees (WhisperFeatureExtractor: 80 bins, 10 ms hop; cropped to the clip,
   padding frames masked out of every metric);
2. the AADD degradation timeline (12 classes, 20 ms);
3. per-token ASR attribution: M1 SmoothGrad gradient x input on the log-mel (time x frequency) and M4 decoder
   cross-attention averaged over Whisper's alignment heads (time-only);
4. word-level ASR errors against the human SLUE gold transcript (jiwer alignment) and the sentiment model's
   word importance (Integrated Gradients), both placed on the timeline via Whisper token timestamps.

It then aggregates per clip (focus ratio, WER inside vs outside flagged audio, sentiment-importance share in
flagged audio, correctness) with bootstrap CIs, runs faithfulness checks (deletion curves, model randomization),
and writes one self-contained offline HTML report. Hypotheses and refutation criteria are in `HYPOTHESES.md`.

## Usage

```
pip install -e Milestone2/AudioXAI          # into the project venv
audioxai explain --clip <id> --asr openai/whisper-tiny --sentiment cardiffnlp/twitter-roberta-base-sentiment-latest
audioxai scan --slice test --sample 50 [--limit 15]   # resumable; --limit keeps each GPU job short
audioxai summarize --slice test
audioxai report --slice test --out report.html
```

Test clips can only be read once `Milestone2/AudioXAI/TEST_UNLOCKED` exists; load-pool clips are refused in code.
The 50-clip test sample is reproducible: sort test.csv by original_index, proportional allocation by label
(>= 3 per class, largest remainder), labels alphabetically, `numpy.random.default_rng(8001).permutation` within label.

## Outputs (results/)
- `test_sample_ids.csv`, `aggregate.csv` (one row per clip; zero-flag clips kept with NaN focus ratio),
  `faithfulness.csv`, `summary.json` (H1-H3 verdicts and CIs), `npz/<clip>.npz` (float16, cropped; not committed, and neither are results/n100/npz or results/whisper-base/npz),
  `report.html` at the package root.

## Models and versions
- ASR openai/whisper-tiny rev 169d4a4341b33bc18d8881c4b69c2e104e1cc0af (decoded with language=en, task=transcribe,
  greedy; token timestamps from `return_token_timestamps=True` with the checkpoint's alignment heads).
- Sentiment cardiffnlp/twitter-roberta-base-sentiment-latest rev 3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7.
- Captum 0.9.0, transformers 5.5.4, torch 2.11, jiwer, AADD 3.1.0 (model v3).

## Limitations
- AADD is trained on synthetic LibriSpeech degradations; on real VoxCeleb audio its flags are estimates.
- Whisper-tiny DTW token timestamps are coarse (several tokens can share one time, especially at the start).
- Grad x input inherits structure from the input; see the randomization check in `summary.json`.
- Not run tonight: M2 (SPES-style spectrogram occlusion), M3 (Grad-CAM, time-only), end-to-end segment
  replacement, and other ASR variants. Only whisper-tiny + Cardiff are wired.

## Credits
- **AADD** (github.com/gertelrina/aadd, MIT): the degradation detector and its timeline; AudioXAI only reads its output.
  The report layout (waveform over spectrogram over timelines) is modeled on AADD's dataset report.
- **Captum** (Meta, BSD-3): LayerIntegratedGradients for sentiment word importance.
- **Whisper** (OpenAI, MIT) and Hugging Face Transformers: ASR, feature extractor, alignment heads, token timestamps.
- **SPES** (arXiv:2411.01710): the idea of spectrogram-perturbation explanations for speech models, which M2 would
  follow; the deletion check here uses the same patch-occlusion idea.
- **Adebayo et al. 2018** (Sanity Checks for Saliency Maps): the model-randomization test.

What AudioXAI adds: the joining of these pieces on one timeline per clip (attribution vs AADD flags vs word errors vs
sentiment importance), the clip-level metrics that test H1-H3 with bootstrap CIs, the faithfulness checks, and the
offline report. None of the underlying models or detectors are ours.

## Results tonight (whisper-tiny + Cardiff, 50-clip stratified test sample; all numbers from results/summary.json)
- Sample: 37 neutral, 10 positive, 3 negative. AADD flagged 50% of 20 ms frames on average; 16 clips had no flagged
  frame (focus ratio NaN, kept in aggregate.csv) and 16 had every frame flagged (ratio 1 by construction).
- **H1 refuted.** Mean M1 focus ratio 0.991, 95% CI [0.911, 1.061], n=34; informative clips (0 < flagged < 1)
  0.983 [0.832, 1.113], n=18. M3 0.972 [0.923, 1.011], M4 0.967 [0.873, 1.054]. Graded Spearman between the M1 time
  profile and max AADD probability 0.033 [-0.027, 0.094]. Focus ratio does not predict Cardiff correctness
  (point-biserial -0.08 [-0.25, 0.13]).
- **H2 refuted.** Error rate in flagged audio 0.242 (554 ref words) vs clean 0.258 (590). Erroneous words carry
  13.0% of sentiment importance vs 12.4% of words (difference +0.006 [-0.010, 0.024]), so errors do not avoid
  sentiment-bearing words. About half of the sentiment importance (0.50 [0.38, 0.63]) lies in flagged audio.
- **H3 strictly refuted** by Phase 1 (T1_final/results/gap_ci.csv, n=1500): human gold text raises Cardiff macro-F1
  from 0.615 to 0.637, +0.022 [0.003, 0.042]. The gain is small; the classifier on perfect text tops out at 0.637,
  which is the main reason ASR choice matters little. On the sample: accuracy 0.74 on ASR text, 0.78 on gold.
- **Faithfulness.** Deletion (n=20): AUC top-k 1.426 vs random 1.181, difference 0.245 [0.187, 0.303], top > random
  in 19/20 clips, so M1 ranks patches better than random. Model randomization (n=10, top 2 decoder layers
  re-initialized): Spearman 0.916 [0.903, 0.928], **FAIL** (rule: < 0.5); |M1| correlates 0.78 with |input|, i.e. grad x
  input is dominated by the input's own structure. Signed maps do change (rho 0.09). M1 vs M2 (occlusion, all 50
  clips): median per-token time-profile Spearman 0.162 [0.136, 0.189]; M1 vs M4 0.54.
- End-to-end window replacement (item 7, 10 clips, positives/negatives first): replacing a 500 ms window with the
  clip's noise floor changes Cardiff P(gold) by 0.054 [0.023, 0.096] in flagged windows vs 0.126 [0.044, 0.209] in
  clean windows (descriptive, small n).
- 100-clip extension (results/n100; the 50 are a per-label prefix of the same seeded permutations): H1 0.995
  [0.939, 1.049], n=71, refuted; H2 refuted.
- Second ASR variant, whisper-base (rev e37978b90ca9030d5170a5c07aadb050351a65bb), same 50 clips
  (results/whisper-base, report_whisper-base.html): mean WER 0.208 vs 0.243 for tiny, yet Cardiff accuracy is
  0.74 for both (37/50 correct each, 35 in common) and the predicted label matches tiny's on 92% of clips (and
  matches the gold-text prediction on 92%). H1 0.991 [0.912, 1.059], refuted; M4 1.012 [0.938, 1.084]. H2 refuted
  (flagged 0.214 vs clean 0.201; erroneous words carry 16.2% of importance vs 13.1% of words, +[0.003, 0.060]).
  Deletion diff 0.18 [0.126, 0.241]; randomization rho 0.931, FAIL. No Phase 1 gold ablation exists for base, so H3
  is reported only for tiny.

## Answer to the question (tonight's evidence, small n)
ASR choice barely changes sentiment because (a) the sentiment classifier's decision is stable under the word errors
both Whisper sizes make: about 92% of predicted labels are the same on ASR text and on gold text, and (b) even
perfect text only lifts Cardiff macro-F1 by about 2 points (0.615 to 0.637). The user's hypothesis, that the ASR
attends to degraded audio, is not supported: attribution mass in AADD-flagged frames matches their share of duration
(focus ratio about 1, CI includes 1) for M1, M2, M3 and M4, and ASR errors are no more frequent in flagged audio.
Caveats: n=50 (3 negatives), AADD flags are estimates on real audio, and M1 fails the randomization sanity check.
No test-slice finding changed any reported configuration.
