"""The only way to run whisper-tiny on practice clips during editing.
Each call = one forward pass (<= 64 practice clips), logged to logs/forward_passes.jsonl.
Raises on pass 51 (brief budget) and on pass 21 (tonight's W1 cap)."""
import datetime, hashlib, json, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_harness as H  # import only: reuse normalize_text, load_wave, errs (harness file untouched)

R = H.R; W = H.W
LOG = f"{R}/logs/forward_passes.jsonl"
PRACTICE = set(np.load(f"{W}/splits/wsp_practice_ids.npy").tolist())
BUDGET, TONIGHT_CAP = 50, 20


def passes_used():
    if not os.path.exists(LOG):
        return 0
    n = 0
    for line in open(LOG):
        try:
            if json.loads(line).get("worker") == "W1":
                n += 1
        except Exception:
            pass
    return n


def practice_table():
    df = H.clip_table("dev")
    return df[df["original_index"].isin(PRACTICE)].reset_index(drop=True)


def _log(ids, purpose, hypothesis_id, edit_id):
    if len(ids) > 64:
        raise ValueError("batch > 64 clips refused")
    bad = [i for i in ids if i not in PRACTICE]
    if bad:
        raise ValueError(f"non-practice ids refused: {bad[:5]}")
    n = passes_used() + 1
    if n > BUDGET:
        raise RuntimeError("pass 51: forward-pass budget exhausted")
    if n > TONIGHT_CAP:
        raise RuntimeError("pass 21: tonight's W1 cap of 20 exhausted")
    rec = {"pass_id": n, "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "n_clips": len(ids), "clip_ids_sha256": hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest(),
           "purpose": purpose, "hypothesis_id": hypothesis_id, "edit_id": edit_id, "worker": "W1"}
    with open(LOG, "a") as f:  # log before running so a crash still counts
        f.write(json.dumps(rec) + "\n")
    return n


def teacher_forced_pass(model, proc, df, texts, purpose, hypothesis_id="", edit_id=""):
    """One forward pass: teacher-force `texts` (one per clip) after the fixed English/transcribe/notimestamps
    prompt; returns per-clip (token_ids, logits[float16 cpu], final-LN hidden[float32 cpu]) and pass id."""
    ids = [int(i) for i in df["original_index"]]
    n = _log(ids, purpose, hypothesis_id, edit_id)
    tok = proc.tokenizer
    prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
    res = []
    with torch.no_grad():
        for p, t in zip(df["wav"].tolist(), texts):
            w = H.load_wave(p)
            feats = proc.feature_extractor(w["raw"], sampling_rate=16000, return_tensors="pt").input_features.cuda()
            body = tok.encode(" " + t.strip(), add_special_tokens=False)[:440]
            dec = torch.tensor([prefix + body], device="cuda")
            out = model(input_features=feats, decoder_input_ids=dec, output_hidden_states=True)
            hid = out.decoder_hidden_states[-1][0]
            res.append((prefix + body, out.logits[0].half().cpu(), hid.float().cpu()))
    return res, n


def run_pass(model, proc, df, purpose, hypothesis_id="", edit_id="", hooks=None):
    """df: rows of practice_table(). hooks: list of (module, hook_fn) forward hooks active for the pass."""
    ids = [int(i) for i in df["original_index"]]
    if len(ids) > 64:
        raise ValueError("batch > 64 clips refused")
    bad = [i for i in ids if i not in PRACTICE]
    if bad:
        raise ValueError(f"non-practice ids refused: {bad[:5]}")
    n = _log(ids, purpose, hypothesis_id, edit_id)
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                   feature_extractor=proc.feature_extractor, device=0, dtype=torch.float32)
    kw = {"chunk_length_s": 30, "generate_kwargs": {"language": "en", "task": "transcribe"}}
    handles = [m.register_forward_hook(fn) for m, fn in (hooks or [])]
    texts = []
    try:
        paths = df["wav"].tolist()
        for i in range(0, len(paths), 8):
            out = asr([H.load_wave(p) for p in paths[i:i + 8]], batch_size=8, **kw)
            texts += [o["text"].strip() for o in out]
    finally:
        for h in handles:
            h.remove()
    out = df.copy()
    out["transcript"] = texts
    e = [H.errs(r, h) for r, h in zip(out["gold"], out["transcript"])]
    out["err_gold"] = [x[0] for x in e]; out["nref_gold"] = [x[1] for x in e]
    wer = out["err_gold"].sum() / max(out["nref_gold"].sum(), 1)
    return out, float(wer), n
