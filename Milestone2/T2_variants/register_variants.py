#!/usr/bin/env python
"""T2 registry: one meta.json per ASR and sentiment variant (replaces Task2/variant.ipynb, which is left in place).
Size on disk counts only the files a service loads: one weight format (safetensors) plus config, generation config,
feature-extractor/preprocessor config and tokenizer files. Offline accuracy comes from the Phase 2 grid results
(Milestone2/T2_variants/results, produced by T1_final/run_t1.py). MinIO credentials are read from the environment only.
Usage: register_variants.py [--results-dir Milestone2/T2_variants/results] [--wsp-transcripts path]"""
import os, sys, json, time, argparse, hashlib, urllib.request
import pandas as pd
T1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "T1_final"); sys.path.insert(0, T1)
from pipeline.common import load_config, repo_root, short, sha256_file

ASR = {"AventIQ-AI/whisper-audio-to-text": "8d9feea90b485a705473d4388250d0e5bf6663f6", "facebook/s2t-small-librispeech-asr": "256e91a603c013ae09aa1e4c09cfc3f8acdf8d19",
       "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
       "UsefulSensors/moonshine-base": "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf"}
SENT = {"cardiffnlp/twitter-roberta-base-sentiment-latest": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7", "tabularisai/multilingual-sentiment-analysis": "eea032081f8d247b4303ef3565e7cec1b6f201c9",
        "siebert/sentiment-roberta-large-english": "74cea614e245b0832c770ec9aa51bd58df965b9c", "distilbert/distilbert-base-uncased-finetuned-sst-2-english": "714eb0fa89d2f80546fda750413ed43d93601a13"}
LOADED = {"model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json", "feature_extractor_config.json", "tokenizer.json", "tokenizer_config.json",
          "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "normalizer.json", "sentencepiece.bpe.model", "spm_char.model", "vocab.txt"}
WEIGHT_FORMATS = (".safetensors", ".bin", ".h5", ".msgpack", ".onnx", ".ot", ".ckpt", ".pt")

def snapshot_dir(name, rev):
    return os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/{rev}")

def size_loaded(d):
    files = {}
    for root_, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root_, f); rel = os.path.relpath(p, d)
            if rel in LOADED or (rel.endswith(".model") and "/" not in rel): files[rel] = os.path.getsize(os.path.realpath(p))
    if "model.safetensors" not in files:   # fall back to pytorch_model.bin when the repo has no safetensors
        pb = os.path.join(d, "pytorch_model.bin")
        if os.path.exists(pb): files["pytorch_model.bin"] = os.path.getsize(os.path.realpath(pb))
    return files

def hub_all_formats(name):
    """What combine_reports.ipynb most likely summed: every file in the Hub repo, all weight formats included."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(name, files_metadata=True)
        tot = sum((s.size or 0) for s in info.siblings); w = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(WEIGHT_FORMATS)}
        return tot, w
    except Exception as e:
        return None, {"error": str(e)[:120]}

def minio_check():
    url = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"); ak = os.environ.get("MINIO_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER"); sk = os.environ.get("MINIO_SECRET_KEY") or os.environ.get("MINIO_ROOT_PASSWORD")
    rec = {"endpoint": url, "credentials_in_env": bool(ak and sk), "reachable": False, "error": None, "uploaded": False}
    try:
        with urllib.request.urlopen(url + "/minio/health/live", timeout=3) as r: rec["reachable"] = r.status == 200
    except Exception as e: rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    return rec

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="Milestone2/T2_variants/results"); ap.add_argument("--wsp-transcripts", default=None); ap.add_argument("--out", default="Milestone2/T2_variants/registry")
    a = ap.parse_args(); root = repo_root(); RD = os.path.join(root, a.results_dir); OUT = os.path.join(root, a.out)
    res = pd.read_csv(os.path.join(RD, "m2_t1_results.csv")); wer = pd.read_csv(os.path.join(RD, "wer_test.csv")).set_index("asr"); thr = json.load(open(os.path.join(RD, "thresholds.json")))
    full = res[res["source"] == "full"]
    minio = minio_check(); print("MinIO:", minio)
    index = []
    def write(kind, name, version, meta):
        d = os.path.join(OUT, kind, short(name), version); os.makedirs(d, exist_ok=True)
        meta.update({"kind": kind, "name": name, "short_name": short(name), "version": version, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "storage": {"minio": minio, "local_path": os.path.relpath(d, root), "note": "weights are not copied into the repo; the path below is the Hugging Face cache snapshot"}})
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, default=float); index.append({"kind": kind, "name": name, "version": version, "size_loaded_MB": meta["size_on_disk"]["loaded_files_MB"], "path": os.path.relpath(d, root)})
    for name, rev in ASR.items():
        d = snapshot_dir(name, rev); files = size_loaded(d); tot_hub, wf = hub_all_formats(name)
        rows = full[full["asr"] == short(name)]
        card = rows[rows["sentiment"] == "cardiff"].iloc[0] if (rows["sentiment"] == "cardiff").any() else None
        meta = {"source_repo": f"https://huggingface.co/{name}", "revision": rev, "snapshot": d, "weights_sha256": sha256_file(os.path.join(d, "model.safetensors")) if os.path.exists(os.path.join(d, "model.safetensors")) else None,
                "size_on_disk": {"loaded_files": files, "loaded_files_MB": round(sum(files.values()) / 1e6, 2), "hub_repo_all_files_MB": round(tot_hub / 1e6, 2) if tot_hub else None, "hub_weight_files_MB": {k: round(v / 1e6, 2) for k, v in wf.items()} if isinstance(wf, dict) and "error" not in wf else wf},
                "offline_accuracy_test": {"n": 1500, "wer_gold": float(wer.loc[name, "wer_gold"]), "wer_gold_nofillers": float(wer.loc[name, "wer_gold_nofillers"]), "wer_pseudo_reference": float(wer.loc[name, "wer_pseudo_ref"]), "reference": "SLUE-VoxCeleb human transcripts (gold); whisper-large-v3 pseudo-reference for comparability",
                                          "cardiff_macro_f1": float(card["macro_f1"]) if card is not None else None, "cardiff_macro_f1_ci": [float(card["macro_f1_ci_lo"]), float(card["macro_f1_ci_hi"])] if card is not None else None,
                                          "cardiff_t": thr.get(f"{short(name)}:full|cardiff", {}).get("t"), "cardiff_tau": None},
                "all_sentiment_downstream": {r["sentiment"]: {"macro_f1": float(r["macro_f1"]), "ci": [float(r["macro_f1_ci_lo"]), float(r["macro_f1_ci_hi"])], "tau": (None if pd.isna(r["tau"]) else float(r["tau"])), "t": thr.get(f"{short(name)}:full|{r['sentiment']}", {}).get("t")} for _, r in rows.iterrows()},
                "notes": {"AventIQ-AI/whisper-audio-to-text": "fine-tune of openai/whisper-base (same architecture, 240/245 tensors differ by ≤2.3e-3); transcripts near-identical to whisper-base",
                          "facebook/s2t-small-librispeech-asr": "fixed variant: batch size 1 (see T1_final/results/s2t_bug_probe.json)",
                          "UsefulSensors/moonshine-base": "English-only; max_new_tokens capped at 6.5 tokens/s per the model card; the card cautions against using Moonshine for subjective classification (it only transcribes here)"}.get(name, "")}
        write("asr", name, rev, meta)
    if a.wsp_transcripts and os.path.exists(os.path.join(root, a.wsp_transcripts)):
        rows = full[full["asr"] == "whisper-tiny-wsp"]; card = rows[rows["sentiment"] == "cardiff"].iloc[0] if len(rows) else None
        meta = {"source_repo": "openai/whisper-tiny + Milestone2/WSP_Whisper/edits (sparse patch)", "revision": "base 169d4a43 + registered WSP patch", "size_on_disk": {"loaded_files": {"base": "see openai/whisper-tiny", "patch": os.path.relpath(a.wsp_transcripts, root)}, "loaded_files_MB": None},
                "offline_accuracy_test": {"wer_gold": float(wer.loc["whisper-tiny-wsp", "wer_gold"]) if "whisper-tiny-wsp" in wer.index else None, "cardiff_macro_f1": float(card["macro_f1"]) if card is not None else None}}
        write("asr", "whisper-tiny-wsp", "wsp", meta)
    for name, rev in SENT.items():
        d = snapshot_dir(name, rev); files = size_loaded(d); tot_hub, wf = hub_all_formats(name)
        rows = full[full["sentiment"] == short(name)]
        meta = {"source_repo": f"https://huggingface.co/{name}", "revision": rev, "snapshot": d, "weights_sha256": sha256_file(os.path.join(d, "model.safetensors")) if os.path.exists(os.path.join(d, "model.safetensors")) else None,
                "size_on_disk": {"loaded_files": files, "loaded_files_MB": round(sum(files.values()) / 1e6, 2), "hub_repo_all_files_MB": round(tot_hub / 1e6, 2) if tot_hub else None, "hub_weight_files_MB": {k: round(v / 1e6, 2) for k, v in wf.items()} if isinstance(wf, dict) and "error" not in wf else wf},
                "label_mapping": {"cardiff": "three classes as-is", "tabularisai": "very negative+negative→Negative, positive+very positive→Positive", "siebert": "Neutral when score<τ", "distilbert": "Neutral when score<τ"}[short(name)],
                "offline_accuracy_test": {f"upstream={r['asr']}": {"macro_f1": float(r["macro_f1"]), "ci": [float(r["macro_f1_ci_lo"]), float(r["macro_f1_ci_hi"])], "tau": (None if pd.isna(r["tau"]) else float(r["tau"])), "t": thr.get(f"{r['asr']}:full|{short(name)}", {}).get("t")} for _, r in rows.iterrows()},
                "gold_text_macro_f1": {r["asr"]: float(r["macro_f1"]) for _, r in res[(res["source"] == "gold") & (res["sentiment"] == short(name))].iterrows()}}
        write("sentiment", name, rev, meta)
    pd.DataFrame(index).to_csv(os.path.join(OUT, "registry_index.csv"), index=False); json.dump(minio, open(os.path.join(OUT, "minio_attempt.json"), "w"), indent=1)
    print(pd.DataFrame(index).to_string(index=False))

if __name__ == "__main__":
    main()
