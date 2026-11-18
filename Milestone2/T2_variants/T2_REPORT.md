# T2 variants and registry (Milestone 2, overnight run 2026-11-18)

Every number is read from `Milestone2/T2_variants/results/` (the Phase 1 pipeline re-run over all variants: `run_t1.py --results-dir Milestone2/T2_variants/results --extra-transcripts ...`) or from `Milestone2/T2_variants/registry/`. `whisper-tiny-wsp` (Phase 3's registered checkpoint fp17) was added at the start of Phase 5; the grid below has 6 × 4 = 24 configurations.

## Variants

ASR (speech-to-text): AventIQ-AI/whisper-audio-to-text, facebook/s2t-small-librispeech-asr (batch-1 fix), openai/whisper-tiny, **openai/whisper-base (new)**, **UsefulSensors/moonshine-base (new)**. Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest, tabularisai/multilingual-sentiment-analysis, siebert/sentiment-roberta-large-english, distilbert/distilbert-base-uncased-finetuned-sst-2-english. Every revision is a 40-character commit SHA recorded in each `registry/<kind>/<name>/<revision>/meta.json` together with the SHA-256 of `model.safetensors`.

Mappings (stated before results, per the guide): Cardiff three classes as-is; tabularisai very negative + negative → Negative, positive + very positive → Positive; SiEBERT and DistilBERT Neutral when the score is below τ (dev-tuned; τ = 0.99 for every configuration, a grid-boundary hit, `results/thresholds.json`). Moonshine: English-only, `max_new_tokens` capped at 6.5 tokens per second of audio as its model card asks; its tokenizer has no pad token, so batching sets `pad_token_id = eos`. The Moonshine card cautions against using it for subjective classification; here it only transcribes.

## Offline accuracy on the test slice (1,500 clips)

WER against the SLUE human transcripts (`results/wer_test.csv`; filler-free and pseudo-reference columns in the file):

| ASR variant | WER vs gold | filler-free | vs whisper-large-v3 | loaded size (MB) |
|---|---|---|---|---|
| UsefulSensors/moonshine-base | **0.1612** | 0.1477 | 0.1482 | 248.07 |
| openai/whisper-base | 0.1817 | 0.1650 | 0.1465 | 294.78 |
| AventIQ-AI/whisper-audio-to-text | 0.1831 | 0.1662 | 0.1478 | 147.12 |
| openai/whisper-tiny | 0.2206 | 0.2052 | 0.1988 | 155.43 |
| whisper-tiny-wsp (fp17 patch on whisper-tiny) | 0.2152 | 0.1997 | 0.2100 | 155.43 + 2.0 patch |
| facebook/s2t-small-librispeech-asr (fixed) | 0.5983 | 0.6047 | 0.6518 | 118.84 |
| (whisper-large-v3 pseudo-reference) | 0.1789 | 0.1628 | — | — |

Macro-F1 with each sentiment model downstream (`results/m2_t1_results.csv`, source = full; CIs in the file):

| ASR \ sentiment | Cardiff | DistilBERT | SiEBERT | tabularisai |
|---|---|---|---|---|
| AventIQ | 0.6211 | 0.4580 | 0.3341 | 0.3304 |
| whisper-base | 0.6184 | 0.4599 | 0.3345 | 0.3292 |
| whisper-tiny | 0.6150 | 0.4575 | 0.3228 | 0.3252 |
| s2t-small (fixed) | 0.6032 | 0.4423 | 0.3254 | 0.3344 |
| moonshine-base | 0.6027 | 0.4598 | 0.3282 | 0.3368 |
| whisper-tiny-wsp | 0.6086 | 0.4562 | 0.3239 | 0.3266 |

Cardiff macro-F1 CIs: AventIQ [0.593, 0.649], whisper-base [0.590, 0.646], whisper-tiny [0.587, 0.643], whisper-tiny-wsp [0.580, 0.637], s2t [0.574, 0.632], moonshine [0.574, 0.631]. The wsp row computed here by the T1 pipeline matches Phase 3's locked harness to 4 decimals (WER 0.2152, Cardiff 0.6086), an independent cross-check. Every pair of ASR variants overlaps. Gold text gives Cardiff 0.6371; the paired full-minus-gold gap (`results/gap_ci.csv`) is −0.016 (AventIQ, CI includes 0), −0.019 (whisper-base), −0.022 (whisper-tiny), −0.034 (s2t), −0.035 (moonshine).

Two observations for the ASR-choice question. First, **WER does not order the sentiment result**: moonshine-base has the lowest WER (0.161) and the lowest Cardiff macro-F1 (0.603), while AventIQ at 0.183 WER is the best (0.621). Second, AventIQ is a fine-tune of whisper-base (240 of 245 tensors differ by at most 2.3e-3) and its WER and macro-F1 are within 0.003 of whisper-base, so the "two Whisper fine-tunes" are effectively one variant.

Assertions over all 24 configurations (`results/assertions_test.csv`, `results/correlations.json`): pass rate (A1∧A2∧A3) vs macro-F1 Pearson −0.302 / Spearman −0.125; with A4a −0.121 / −0.022; with A4b −0.367 / −0.299. Same conclusion as T1: pass rates are per-configuration quantities and do not rank configurations.

## Registry

`register_variants.py` writes `registry/<asr|sentiment>/<name>/<revision>/meta.json` with name, version, source repo, revision, weights SHA-256, size on disk, offline accuracy (WER for ASR; test macro-F1 with each sentiment model, with τ and t) and the storage record. **MinIO**: `http://localhost:9000/minio/health/live` → connection refused, no credentials in the environment (`registry/minio_attempt.json`), so the files were written locally and nothing was uploaded. `Task2/variant.ipynb` is left untouched (it hard-codes a MinIO password and does not run); this script supersedes it.

Sizes that `combine_reports.ipynb` got wrong, and why (`meta.json`, `size_on_disk`):

| model | notebook (MB) | loaded files (MB) | all files in the Hub repo (MB) | explanation |
|---|---|---|---|---|
| whisper-tiny | 580.65 | 155.43 | 608.85 | the repo holds four copies of the weights (safetensors, PyTorch, TF, Flax, 151 MB each); a service loads one |
| DistilBERT SST-2 | 1,278.59 | 268.06 | 1,340.70 | five weight formats (safetensors, PyTorch, TF, ONNX, Rust) of 268 MB each |
| SiEBERT | 4,072.68 | 1,422.87 | 4,270.51 | three weight formats of 1.42 GB each |

The notebook summed every file in each Hugging Face repository (all weight formats); the registry counts the single safetensors file plus config, tokenizer and feature-extractor files, which is what the T3 service will load.

## Not done tonight

Quantized / ONNX / OpenVINO variants and input-side knobs (chunk size) were not built; the brief's six ASR variants (with `whisper-tiny-wsp` pending Phase 3) and four sentiment variants satisfy T2's "at least four per scalable stage". No latency numbers: those belong to T5/T6 on the course VM.
