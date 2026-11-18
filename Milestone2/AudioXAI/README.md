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
  `faithfulness.csv`, `summary.json` (H1-H3 verdicts and CIs), `npz/<clip>.npz` (float16, cropped; not committed),
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
