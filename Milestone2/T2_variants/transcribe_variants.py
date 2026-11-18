#!/usr/bin/env python
"""Transcribe dev+test with the new ASR variants (whisper-base, moonshine-base) into the T1 cache. GPU; run under flock."""
import os, sys, pandas as pd
T1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "T1_final"); sys.path.insert(0, T1)
from pipeline.common import load_config, load_slices, repo_root, short
from pipeline.asr import transcribe
cfg = load_config(); root = repo_root(); sl = load_slices(cfg)
files = pd.concat([sl["dev"], sl["test"]])["file"].tolist()
VARIANTS = {"openai/whisper-base": ("e37978b90ca9030d5170a5c07aadb050351a65bb", 8), "UsefulSensors/moonshine-base": ("7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf", 8)}
for name, (rev, bs) in VARIANTS.items():
    cache = os.path.join(root, cfg["cache_dir"], f"transcripts_{short(name)}.csv")
    tr = transcribe(name, rev, files, os.path.join(root, cfg["audio_dir"]), bs, cache, log_path=os.path.join(root, "logs", "gpu_jobs.jsonl"))
    print(name, len(tr), "transcripts; mean ms", round(tr["ms"].mean(), 1), "| example:", tr["transcript"].iloc[0][:100])
