"""Per-clip AudioXAI computation for Whisper + Cardiff.

M1: SmoothGrad(gradient x input) on Whisper's log-mel, target = teacher-forced log-prob of each generated
    token given the model's own previous tokens. M4: decoder cross-attention averaged over alignment heads
    (time-only). Sentiment: captum LayerIntegratedGradients on Cardiff's embedding layer.
"""
import re
from functools import lru_cache

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from . import paths

SR = 16000
HOP_S = 0.01          # log-mel hop
SEG_S = 0.02          # AADD / analysis grid
CLASSES = ['noise', 'music', 'reverb', 'clipping', 'lowpass', 'highpass', 'codec', 'packet_loss',
           'hum', 'clicks', 'pitch_shift', 'time_stretch']
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(text):
    text = str(text).lower().replace("—", " ").replace("–", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- data guards
@lru_cache(None)
def slices():
    dev = pd.read_csv(paths.PE / "development.csv", usecols=["file", "label", "original_index", "duration"])
    test = pd.read_csv(paths.PE / "test.csv", usecols=["file", "label", "original_index", "duration"])
    load = pd.read_csv(paths.PE / "load_pool.csv", usecols=["file"])
    assert not (set(dev.file) & set(test.file)) and not (set(dev.file) & set(load.file)) \
        and not (set(test.file) & set(load.file)), "slices overlap"
    return dev, test, set(load.file)


def clip_meta(clip):
    dev, test, load = slices()
    if clip in load:
        raise PermissionError(f"{clip} is in the load pool; AudioXAI never reads it")
    for name, df in (("dev", dev), ("test", test)):
        r = df[df.file == clip]
        if len(r):
            if name == "test" and not paths.TEST_UNLOCKED.exists():
                raise PermissionError("test audio is locked until Milestone2/AudioXAI/TEST_UNLOCKED exists")
            r = r.iloc[0]
            return dict(file=clip, slice=name, label=r.label, original_index=int(r.original_index),
                        duration=float(r.duration))
    raise KeyError(clip)


@lru_cache(None)
def gold_table():
    g = pd.read_csv(paths.GOLD_TSV, sep="\t", usecols=["id", "normalized_text"], keep_default_na=False)
    return dict(zip(g.id, g.normalized_text))


def load_audio(clip):
    meta = clip_meta(clip)  # enforces the guards
    wav, sr = sf.read(paths.AUDIO / f"{clip}.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == SR, sr
    return wav, meta


# ---------------------------------------------------------------- AADD
def aadd_frames(clip, seg_csv_cache={}):
    """Return (n_seg, 12) probabilities on the 20 ms grid for this clip, from the orchestrator's file if
    AADD_READY exists, else from our dev-only scans under results/dev/."""
    srcs = []
    if paths.AADD_READY.exists():
        srcs.append(paths.AADD_SEG)
    srcs += sorted((paths.X / "results/dev").glob("aadd_*seg20ms.csv"))
    for s in srcs:
        if s not in seg_csv_cache:
            df = pd.read_csv(s)
            df["clip"] = df.path.map(lambda p: p.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            seg_csv_cache[s] = {k: g.sort_values("start_s") for k, g in df.groupby("clip")}
        d = seg_csv_cache[s]
        if clip in d:
            g = d[clip]
            return g[[f"p_{c}" for c in CLASSES]].to_numpy(np.float32), str(s)
    return None, None


# ---------------------------------------------------------------- models
@lru_cache(None)
def asr():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(paths.ASR_ID, revision=paths.ASR_REV)
    m = WhisperForConditionalGeneration.from_pretrained(paths.ASR_ID, revision=paths.ASR_REV,
                                                        attn_implementation="eager").to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return proc, m


@lru_cache(None)
def sentiment():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.SENT_ID, revision=paths.SENT_REV)
    m = AutoModelForSequenceClassification.from_pretrained(paths.SENT_ID, revision=paths.SENT_REV).to(DEV).eval()
    return tok, m


def features(wav):
    proc, _ = asr()
    f = proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
    return f.input_features.to(DEV), f.attention_mask.to(DEV)  # (1,80,3000), (1,3000)


def transcribe(feats, amask):
    proc, m = asr()
    with torch.no_grad():
        out = m.generate(feats, attention_mask=amask, language="en", task="transcribe", return_token_timestamps=True,
                         return_dict_in_generate=True)
    seq = out["sequences"][0]
    ts = out["token_timestamps"][0].float().cpu().numpy()
    return seq, ts


def teacher_forced_logp(m, feats, seq, keep):
    """feats (B,80,3000), seq (L,) -> per-position log p(seq[t] | seq[:t]) for positions keep (B,K)."""
    dec_in = seq[:-1].unsqueeze(0).expand(feats.shape[0], -1)
    out = m(input_features=feats, decoder_input_ids=dec_in)
    lp = torch.log_softmax(out.logits.float(), -1)
    tgt = seq[1:]
    lp = lp[:, torch.arange(len(tgt), device=lp.device), tgt]  # (B, L-1)
    return lp[:, keep]


def text_positions(seq):
    """Indices t (into seq[1:]) whose target token is a text token (not special)."""
    proc, m = asr()
    eot = proc.tokenizer.eos_token_id
    tgt = seq[1:].tolist()
    return [i for i, t in enumerate(tgt) if t < eot]


def m1_smoothgrad(m, feats, seq, keep, n=8, sigma_frac=0.1, seed=8001, n_frames=None):
    """Return (K, 80, n_frames) SmoothGrad gradient x input maps (float32 numpy) and the clean logp (K,)."""
    n_frames = n_frames or feats.shape[-1]
    std = feats[..., :n_frames].std()
    g = torch.Generator(device=feats.device).manual_seed(seed)
    noise = torch.randn((n,) + feats.shape[1:], generator=g, device=feats.device) * sigma_frac * std
    x = (feats.expand(n, -1, -1) + noise).detach().requires_grad_(True)
    lp = teacher_forced_logp(m, x, seq, keep)  # (n,K)
    maps = np.zeros((len(keep), feats.shape[1], n_frames), np.float32)
    for k in range(len(keep)):
        gk, = torch.autograd.grad(lp[:, k].sum(), x, retain_graph=k < len(keep) - 1)
        maps[k] = (gk * x).mean(0)[:, :n_frames].detach().float().cpu().numpy()
    with torch.no_grad():
        lp0 = teacher_forced_logp(m, feats, seq, keep)[0].float().cpu().numpy()
    return maps, lp0


def m3_gradcam(m, feats, seq, keep, n_enc):
    """Grad-CAM on the last encoder layer output (time-only: the Conv1d front end leaves no frequency axis).
    alpha_c = mean_t dlogp/dA[t,c]; cam_t = ReLU(sum_c alpha_c A[t,c]). Same formula as captum LayerGradCam."""
    store = {}
    layer = m.model.encoder.layers[-1]

    def hook(mod, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        a.retain_grad(); store["a"] = a
    h = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = feats.detach().clone().requires_grad_(True)  # makes the graph reach the encoder
            lp = teacher_forced_logp(m, x, seq, keep)[0]
            A = store["a"]
            cams = np.zeros((len(keep), n_enc), np.float32)
            for k in range(len(keep)):
                g, = torch.autograd.grad(lp[k], A, retain_graph=k < len(keep) - 1)
                alpha = g[0].mean(0)  # (C,)
                cam = torch.relu((A[0] * alpha).sum(-1))[:n_enc]
                cams[k] = cam.detach().float().cpu().numpy()
    finally:
        h.remove()
    return cams / np.maximum(cams.sum(1, keepdims=True), 1e-12)


def m4_crossattn(m, feats, seq, keep, n_enc):
    heads = m.generation_config.alignment_heads
    with torch.no_grad():
        out = m(input_features=feats, decoder_input_ids=seq[:-1].unsqueeze(0), output_attentions=True)
    ca = out.cross_attentions  # tuple layers of (1, H, L-1, 1500)
    a = torch.stack([ca[l][0, h] for l, h in heads]).mean(0)  # (L-1, 1500)
    a = a[keep][:, :n_enc].float().cpu().numpy()
    return a / np.maximum(a.sum(1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- words, errors
def words_from_tokens(seq, ts, keep):
    """Merge text tokens into words. Returns text and list of dicts (word, norm, t0, t1, c0, c1, toks)."""
    proc, _ = asr()
    tok = proc.tokenizer
    ids = seq.tolist()
    pieces = [(tok.decode([ids[t + 1]]), t) for t in keep]  # keep indexes seq[1:], token id = ids[t+1]
    text, words = "", []
    for s, t in pieces:
        start_new = (not words) or s.startswith(" ")
        c0 = len(text)
        text += s
        tt0 = float(ts[t + 1])
        if start_new:
            words.append(dict(word=s, c0=c0, c1=len(text), t0=tt0, toks=[t]))
        else:
            words[-1]["word"] += s
            words[-1]["c1"] = len(text)
            words[-1]["toks"].append(t)
    # end time = timestamp of the next token after the word's last token
    for w in words:
        last = w["toks"][-1] + 2
        w["t1"] = float(ts[min(last, len(ts) - 1)])
        if w["t1"] < w["t0"]:
            w["t1"] = w["t0"]
        w["word"] = w["word"].strip()
        w["norm"] = normalize_text(w["word"])
    lead = len(text) - len(text.lstrip())
    for w in words:
        w["c0"] -= lead; w["c1"] -= lead
    return text.strip(), words


def align_errors(gold, words):
    """jiwer word alignment of normalize_text(gold) vs normalized hypothesis words.
    Returns per-hyp-word tag list (aligned to `words`) and a list of reference events (time, is_error, kind)."""
    hyp_tokens, owner = [], []
    for i, w in enumerate(words):
        for piece in w["norm"].split():
            hyp_tokens.append(piece); owner.append(i)
    ref = normalize_text(gold)
    tags = ["ok"] * len(words)
    events = []
    if not ref.split():
        return tags, events, dict(S=0, D=0, I=len(hyp_tokens), H=0, N=0)
    if not hyp_tokens:
        return tags, [(None, True, "del")] * len(ref.split()), dict(S=0, D=len(ref.split()), I=0, H=0, N=len(ref.split()))
    o = jiwer.process_words(ref, " ".join(hyp_tokens))
    cnt = dict(S=o.substitutions, D=o.deletions, I=o.insertions, H=o.hits, N=len(ref.split()))

    def wtime(j):
        w = words[owner[j]]
        return 0.5 * (w["t0"] + w["t1"])

    for ch in o.alignments[0]:
        if ch.type == "equal":
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                events.append((wtime(j), False, "ok"))
        elif ch.type == "substitute":
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                tags[owner[j]] = "sub"; events.append((wtime(j), True, "sub"))
        elif ch.type == "insert":
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                tags[owner[j]] = "ins"; events.append((wtime(j), True, "ins"))
        elif ch.type == "delete":
            j = ch.hyp_start_idx
            if j < len(hyp_tokens):
                t = words[owner[j]]["t0"]
                if j > 0:
                    t = 0.5 * (words[owner[j - 1]]["t1"] + t)
            else:
                t = words[owner[-1]]["t1"]
            for _ in range(ch.ref_start_idx, ch.ref_end_idx):
                events.append((t, True, "del"))
    return tags, events, cnt


# ---------------------------------------------------------------- sentiment
LABELS3 = ["Negative", "Neutral", "Positive"]


def sentiment_predict(texts):
    tok, m = sentiment()
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
    with torch.no_grad():
        p = torch.softmax(m(**enc).logits.float(), -1).cpu().numpy()
    id2 = m.config.id2label
    order = [LABELS3.index(id2[i].capitalize()) for i in range(p.shape[1])]
    q = np.zeros_like(p); q[:, order] = p
    return q  # columns Negative, Neutral, Positive


def sentiment_ig(text, words, n_steps=32):
    """LayerIntegratedGradients on Cardiff embeddings for the predicted class. Returns per-word signed and abs
    importance (aligned to words), predicted probs."""
    from captum.attr import LayerIntegratedGradients
    tok, m = sentiment()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(DEV); am = enc["attention_mask"].to(DEV)
    probs = sentiment_predict([text])[0]
    with torch.no_grad():
        pred_raw = int(m(input_ids=ids, attention_mask=am).logits.argmax(-1))
    base = ids.clone()
    special = torch.tensor([t in tok.all_special_ids for t in ids[0].tolist()], device=DEV)
    base[0, ~special] = tok.pad_token_id

    def f(inp, mask):
        return m(input_ids=inp, attention_mask=mask).logits

    lig = LayerIntegratedGradients(f, m.roberta.embeddings)
    attr = lig.attribute(ids, baselines=base, target=pred_raw, additional_forward_args=(am,), n_steps=n_steps,
                         internal_batch_size=n_steps)
    a = attr.sum(-1)[0].detach().float().cpu().numpy()
    sig = np.zeros(len(words)); ab = np.zeros(len(words))
    for (c0, c1), v in zip(offs, a):
        if c1 <= c0:
            continue
        for i, w in enumerate(words):
            if c0 < w["c1"] and c1 > w["c0"]:
                sig[i] += v; ab[i] += abs(v)
                break
    return sig, ab, probs


# ---------------------------------------------------------------- per-clip driver
def flagged_mask(aadd, n_seg):
    if aadd is None:
        return None
    a = aadd[:n_seg]
    if len(a) < n_seg:
        a = np.vstack([a, np.zeros((n_seg - len(a), a.shape[1]), np.float32)])
    return (a >= 0.5).any(1)


def to_seg(profile_10ms, n_seg):
    """(K, n_frames@10ms) -> (K, n_seg@20ms) by summing pairs."""
    K, F = profile_10ms.shape
    pad = 2 * n_seg - F
    if pad > 0:
        profile_10ms = np.concatenate([profile_10ms, np.zeros((K, pad), profile_10ms.dtype)], 1)
    return profile_10ms[:, :2 * n_seg].reshape(K, n_seg, 2).sum(-1)


def explain_clip(clip, smooth_n=8, save_dir=None):
    proc, m = asr()
    wav, meta = load_audio(clip)
    dur = len(wav) / SR
    n_frames = min(3000, int(np.ceil(dur / HOP_S)))
    n_seg = int(np.ceil(n_frames / 2))
    feats, amask = features(wav)
    seq, ts = transcribe(feats, amask)
    assert len(ts) == len(seq), (len(ts), len(seq))
    keep = text_positions(seq)
    text, words = words_from_tokens(seq, ts, keep)
    if keep:
        maps, lp0 = m1_smoothgrad(m, feats, seq, keep, n=smooth_n, n_frames=n_frames)
        m4 = m4_crossattn(m, feats, seq, keep, n_seg)
        m3 = m3_gradcam(m, feats, seq, keep, n_seg)
    else:
        m3 = np.zeros((0, n_seg))
        maps = np.zeros((0, 80, n_frames), np.float32); lp0 = np.zeros(0); m4 = np.zeros((0, n_seg))
    gold = gold_table().get(clip, "")
    tags, events, cnt = align_errors(gold, words)
    if words and text:
        sig, ab, probs = sentiment_ig(text, words)
    else:
        sig = ab = np.zeros(0); probs = sentiment_predict([text or " "])[0]
    pg = sentiment_predict([gold or " "])[0]
    aadd, aadd_src = aadd_frames(clip)
    res = dict(meta=meta, dur=dur, n_frames=n_frames, n_seg=n_seg, seq=seq.cpu().numpy(), ts=ts, keep=np.array(keep),
               text=text, words=words, tags=tags, events=events, cnt=cnt, gold=gold, maps=maps, lp0=lp0, m4=m4, m3=m3,
               ig_sig=sig, ig_abs=ab, probs=probs, probs_gold=pg, aadd=aadd, aadd_src=aadd_src,
               logmel=feats[0, :, :n_frames].float().cpu().numpy(), wav=wav)
    if save_dir is not None:
        save_npz(res, save_dir)
    return res


def save_npz(res, save_dir):
    from pathlib import Path
    import json
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    clip = res["meta"]["file"]
    np.savez_compressed(
        Path(save_dir) / f"{clip}.npz",
        logmel=res["logmel"].astype(np.float16), m1=res["maps"].astype(np.float16), m4=res["m4"].astype(np.float16), m3=res["m3"].astype(np.float16),
        aadd=(res["aadd"][:res["n_seg"]] if res["aadd"] is not None else np.zeros((0, 12))).astype(np.float16),
        seq=res["seq"], token_ts=res["ts"].astype(np.float32), keep=res["keep"], lp0=res["lp0"].astype(np.float32),
        ig_sig=np.asarray(res["ig_sig"], np.float32), ig_abs=np.asarray(res["ig_abs"], np.float32),
        probs=res["probs"].astype(np.float32), probs_gold=res["probs_gold"].astype(np.float32),
        meta=json.dumps(dict(meta=res["meta"], text=res["text"], gold=res["gold"], tags=res["tags"], cnt=res["cnt"],
                             words=[{k: v for k, v in w.items()} for w in res["words"]], events=res["events"],
                             aadd_src=res["aadd_src"], dur=res["dur"], n_frames=res["n_frames"], n_seg=res["n_seg"],
                             asr=paths.ASR_ID, asr_rev=paths.ASR_REV, sent=paths.SENT_ID, sent_rev=paths.SENT_REV)))
