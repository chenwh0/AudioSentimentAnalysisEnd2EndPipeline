import os, time, json
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import pipeline

def load_wave(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    return {"raw": x, "sampling_rate": sr}

def get_asr(name, revision, device=0):
    """Same settings as the teammate notebook: chunk_length_s=30 and language/task for multilingual Whisper."""
    p = pipeline("automatic-speech-recognition", model=name, revision=revision, device=device, dtype=torch.float32)
    kw = {}
    arch = type(p.model).__name__.lower()
    if "whisper" in arch:
        kw["chunk_length_s"] = 30
        if getattr(p.model.generation_config, "is_multilingual", False):
            kw["generate_kwargs"] = {"language": "en", "task": "transcribe"}
    if "moonshine" in arch:
        # model card: cap generated tokens at about 6.5 tokens per second of audio to avoid hallucination loops
        if p.tokenizer.pad_token_id is None:
            p.tokenizer.pad_token_id = p.model.config.eos_token_id
        kw["_moonshine_tokens_per_s"] = 6.5
    return p, kw

def transcribe(name, revision, files, audio_dir, batch_size, cache_path, device=0, log_path=None):
    """Transcribe `files` (ids) unless cache_path already holds them. Returns DataFrame(file, transcript, ms)."""
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path).set_index("file")
        missing = [f for f in files if f not in cached.index]
        if not missing: return cached.loc[files].reset_index()
    else:
        cached = pd.DataFrame(columns=["transcript", "ms"]); missing = list(files)
    p, kw = get_asr(name, revision, device)
    tps = kw.pop("_moonshine_tokens_per_s", None)
    rows = []
    t_job = time.time()
    for i in range(0, len(missing), batch_size):
        ids = missing[i:i + batch_size]
        waves = [load_wave(os.path.join(audio_dir, f"{f}.wav")) for f in ids]
        call_kw = dict(kw)
        if tps:
            maxdur = max(w["raw"].size for w in waves) / 16000.0
            call_kw["generate_kwargs"] = {"max_new_tokens": int(maxdur * tps) + 1}
        t0 = time.perf_counter()
        out = p(waves, batch_size=batch_size, **call_kw) if batch_size > 1 else [p(waves[0], **call_kw)]
        ms = (time.perf_counter() - t0) * 1000 / len(ids)
        for f, o in zip(ids, out): rows.append({"file": f, "transcript": o["text"].strip(), "ms": ms})
    new = pd.DataFrame(rows).set_index("file")
    allc = pd.concat([cached, new]) if len(cached) else new
    allc.reset_index().to_csv(cache_path, index=False)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": f"transcribe:{name}", "event": "end",
                                "pid": os.getpid(), "note": f"{len(missing)} clips, batch {batch_size}, {time.time()-t_job:.0f}s"}) + "\n")
    del p; torch.cuda.empty_cache()
    return allc.loc[files].reset_index()
