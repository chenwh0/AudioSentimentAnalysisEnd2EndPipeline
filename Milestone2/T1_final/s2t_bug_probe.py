#!/usr/bin/env python
"""Probe the cause of the s2t repetition loops on a dev subset: batch-8 pipeline (as the notebook), batch-1 pipeline,
and batch-8 model.generate with an explicit attention mask. Dev clips only (seed 8001)."""
import os, sys, json, time, numpy as np, pandas as pd, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.common import load_config, load_slices, repo_root, normalize_text, dev_subset_ids, write_json
from pipeline.asr import get_asr, load_wave
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
cfg = load_config(); root = repo_root(); sl = load_slices(cfg); dev = sl["dev"]
N = 48; ids = dev_subset_ids(dev, N); d = dev.set_index("file").loc[ids]
name = "facebook/s2t-small-librispeech-asr"; rev = cfg["asr_models"][name]["revision"]
paths = [os.path.join(root, cfg["audio_dir"], f"{f}.wav") for f in ids]
ref = [normalize_text(x) for x in d[cfg["pseudo_ref_col"]].fillna("")]
def nw(a): return np.array([len(str(x).split()) for x in a])
rw = nw(ref); out = {"n_dev_clips": N, "dev_clip_ids": ids}
p, kw = get_asr(name, rev)
for bs in (8, 1):
    hyp = []
    for i in range(0, N, bs):
        waves = [load_wave(x) for x in paths[i:i+bs]]
        o = p(waves, batch_size=bs, **kw) if bs > 1 else [p(waves[0], **kw)]
        hyp += [x["text"].strip() for x in o]
    hw = nw([normalize_text(x) for x in hyp])
    out[f"pipeline_batch{bs}"] = {"gt_1.5x_pseudo": int((hw > 1.5 * rw).sum()), "mean_words": float(hw.mean()), "ref_mean_words": float(rw.mean()), "example": hyp[0][:160]}
# sort clips by duration descending inside each batch of 8 (less padding)? no: test explicit attention mask instead
proc = Speech2TextProcessor.from_pretrained(name, revision=rev); m = Speech2TextForConditionalGeneration.from_pretrained(name, revision=rev).cuda().eval()
hyp = []
for i in range(0, N, 8):
    waves = [load_wave(x)["raw"] for x in paths[i:i+8]]
    inp = proc(waves, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
    with torch.no_grad():
        g = m.generate(input_features=inp.input_features.cuda(), attention_mask=inp.attention_mask.cuda())
    hyp += [x.strip() for x in proc.batch_decode(g, skip_special_tokens=True)]
hw = nw([normalize_text(x) for x in hyp])
out["generate_batch8_with_attention_mask"] = {"gt_1.5x_pseudo": int((hw > 1.5 * rw).sum()), "mean_words": float(hw.mean()), "example": hyp[0][:160]}
hyp = []
for i in range(0, N, 8):
    waves = [load_wave(x)["raw"] for x in paths[i:i+8]]
    inp = proc(waves, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
    with torch.no_grad():
        g = m.generate(input_features=inp.input_features.cuda())
    hyp += [x.strip() for x in proc.batch_decode(g, skip_special_tokens=True)]
hw = nw([normalize_text(x) for x in hyp])
out["generate_batch8_without_attention_mask"] = {"gt_1.5x_pseudo": int((hw > 1.5 * rw).sum()), "mean_words": float(hw.mean()), "example": hyp[0][:160]}
# does the pipeline pass an attention mask? inspect preprocess output keys
sample = next(iter(p.preprocess(load_wave(paths[0]))))
out["pipeline_preprocess_keys"] = list(sample.keys()) if isinstance(sample, dict) else str(type(sample))
write_json(out, os.path.join(root, cfg["results_dir"], "s2t_bug_probe.json")); print(json.dumps({k: v for k, v in out.items() if k != "dev_clip_ids"}, indent=1))
