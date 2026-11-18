"""LOCKED evaluation harness for WSP-Whisper (do not edit; SHA-256 in wsp/HARNESS_LOCK.txt).

Usage: python eval_harness.py <checkpoint> <split>
  checkpoint: 'base' or a patch name (edits/patch_<name>.safetensors, rebuilt via apply_edits.py)
  split: holdout | test | dev
Settings (notebook): transformers ASR pipeline, chunk_length_s=30,
generate_kwargs={"language":"en","task":"transcribe"}, batch_size=8, fp32, soundfile mono float32.
Guards: base test runs once only; an edited checkpoint on test requires results/REGISTERED.txt.
Prints summary numbers only.
"""
import datetime, json, os, re, sys
import numpy as np, pandas as pd, soundfile, torch, jiwer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_edits as AE

R = "/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline"
PE = f"{R}/Milestone2/PipelineEvaluation"
W = f"{R}/Milestone2/WSP_Whisper"
REF = "openai/whisper-large-v3_transcript"


def normalize_text(text):
    text = str(text).lower().replace("—", " ").replace("–", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_wave(path):
    x, sr = soundfile.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return {"raw": x, "sampling_rate": sr}


def clip_table(split):
    if split == "test":
        df = pd.read_csv(f"{PE}/test.csv")
    else:
        df = pd.read_csv(f"{PE}/development.csv")
        if split == "holdout":
            ids = set(np.load(f"{W}/splits/wsp_holdout_ids.npy").tolist())
            df = df[df["original_index"].isin(ids)]
        elif split != "dev":
            raise ValueError(split)
    load = set(np.load(f"{PE}/load_pool_indices.npy").tolist())
    assert not (set(df["original_index"].tolist()) & load)
    gold = pd.read_csv(f"{R}/Milestone2/T1_final/data/slue_voxceleb_test.tsv", sep="\t").set_index("id")["normalized_text"]
    df = df.sort_values("original_index").reset_index(drop=True)
    df["gold"] = df["file"].map(gold).fillna("")
    df["wav"] = df["file"].map(lambda f: f"{PE}/data/audio/{f}.wav")
    return df


def errs(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    n = len(r.split())
    if n == 0:
        return 0, 0
    o = jiwer.process_words(r, h if h else "")
    return o.substitutions + o.deletions + o.insertions, n


def corpus_wer(refs, hyps):
    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps) if normalize_text(r)]
    return jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])


def main(ckpt, split):
    out_csv = f"{W}/results/eval_{ckpt}_{split}.csv"
    if split == "test":
        if ckpt == "base" and os.path.exists(out_csv):
            sys.exit("base test run already exists; never rerun it")
        if ckpt != "base" and not os.path.exists(f"{W}/results/REGISTERED.txt"):
            sys.exit("register a checkpoint (results/REGISTERED.txt) before any edited test run")
        if ckpt != "base" and os.path.exists(out_csv):
            sys.exit("this checkpoint already has a test run")
    df = clip_table(split)
    from transformers import pipeline
    model = AE.build(ckpt)
    proc = AE.load_processor()
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                   feature_extractor=proc.feature_extractor, device=0 if torch.cuda.is_available() else -1,
                   dtype=torch.float32)
    kw = {"chunk_length_s": 30, "generate_kwargs": {"language": "en", "task": "transcribe"}}
    texts, bs, paths = [], 8, df["wav"].tolist()
    for i in range(0, len(paths), bs):
        batch = [load_wave(p) for p in paths[i:i + bs]]
        out = asr(batch, batch_size=bs, **kw)
        texts += [o["text"].strip() for o in out]
    df["transcript"] = texts
    eg = [errs(r, h) for r, h in zip(df["gold"], df["transcript"])]
    ep = [errs(r, h) for r, h in zip(df[REF].fillna(""), df["transcript"])]
    df["err_gold"], df["nref_gold"] = [e[0] for e in eg], [e[1] for e in eg]
    df["err_pseudo"], df["nref_pseudo"] = [e[0] for e in ep], [e[1] for e in ep]
    df["wer_gold_clip"] = np.where(df["nref_gold"] > 0, df["err_gold"] / df["nref_gold"].clip(lower=1), np.nan)
    df["wer_pseudo_clip"] = np.where(df["nref_pseudo"] > 0, df["err_pseudo"] / df["nref_pseudo"].clip(lower=1), np.nan)
    hw = df["transcript"].map(lambda t: len(normalize_text(t).split()))
    df["word_ratio"] = np.where(df["nref_gold"] > 0, hw / df["nref_gold"].clip(lower=1), np.nan)
    raw_words = df["transcript"].map(lambda t: len(str(t).split()))
    df["A1"] = raw_words >= 2
    rate = raw_words / df["duration"]
    df["A2"] = (rate >= 0.5) & (rate <= 5.0)
    df["long"] = df["word_ratio"] > 1.5
    df["short"] = df["word_ratio"] < 0.5
    keep = ["file", "original_index", "label", "duration", "transcript", "gold", REF, "err_gold", "nref_gold",
            "err_pseudo", "nref_pseudo", "wer_gold_clip", "wer_pseudo_clip", "word_ratio", "A1", "A2", "long", "short"]
    df[keep].to_csv(out_csv, index=False)
    rec = {"ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "checkpoint": ckpt,
           "split": split, "n": int(len(df)),
           "wer_gold": float(corpus_wer(df["gold"], df["transcript"])),
           "wer_pseudo": float(corpus_wer(df[REF].fillna(""), df["transcript"])),
           "a1": float(df["A1"].mean()), "a2": float(df["A2"].mean()),
           "long": int(df["long"].sum()), "short": int(df["short"].sum())}
    with open(f"{R}/logs/eval_runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
