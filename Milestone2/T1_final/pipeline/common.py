import os, re, json, hashlib
import numpy as np, pandas as pd, yaml

LABELS = ["Negative", "Neutral", "Positive"]
FILLERS = {"uh", "um", "uhm", "mm", "hmm"}
SHORT = {
    "AventIQ-AI/whisper-audio-to-text": "AventIQ", "facebook/s2t-small-librispeech-asr": "s2t-small",
    "openai/whisper-tiny": "whisper-tiny", "openai/whisper-base": "whisper-base",
    "UsefulSensors/moonshine-base": "moonshine-base", "whisper-tiny-wsp": "whisper-tiny-wsp",
    "openai/whisper-large-v3": "whisper-large-v3",
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "cardiff", "tabularisai/multilingual-sentiment-analysis": "tabularisai",
    "siebert/sentiment-roberta-large-english": "siebert", "distilbert/distilbert-base-uncased-finetuned-sst-2-english": "distilbert",
}
def short(name): return SHORT.get(name, name.split("/")[-1])

def normalize_text(text):
    """The teammate notebook's normalization (kept verbatim so WER stays comparable)."""
    text = str(text).lower().replace("—", " ").replace("–", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_no_fillers(text):
    return " ".join(w for w in normalize_text(text).split() if w not in FILLERS)

def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def load_config(path=None):
    path = path or os.path.join(repo_root(), "Milestone2", "T1_final", "config.yaml")
    with open(path) as f: return yaml.safe_load(f)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def load_slices(cfg):
    """Rebuild the dev/test/load-pool slices from the saved CSVs and index files; Linux audio paths from `file`."""
    root = repo_root(); sd = os.path.join(root, cfg["slices_dir"])
    out = {}
    for name, csv, npy in [("dev", "development.csv", "development_indices.npy"), ("test", "test.csv", "test_indices.npy"),
                           ("load_pool", "load_pool.csv", "load_pool_indices.npy")]:
        df = pd.read_csv(os.path.join(sd, csv))
        idx = np.load(os.path.join(sd, npy))
        assert set(idx) == set(df["original_index"]), f"{name}: index file and CSV disagree"
        df = df.drop(columns=[c for c in df.columns if c == "audio_path"])
        df["audio_path"] = df["file"].map(lambda f: os.path.join(root, cfg["audio_dir"], f"{f}.wav"))
        df["label"] = df["label"].astype(str)
        out[name] = df.reset_index(drop=True)
    a, b, c = (set(out[k]["original_index"]) for k in ("dev", "test", "load_pool"))
    assert not (a & b) and not (a & c) and not (b & c), "slices overlap"
    assert len(out["dev"]) == 500 and len(out["test"]) == 1500 and len(out["load_pool"]) == 1426
    return out

def load_gold(cfg):
    g = pd.read_csv(os.path.join(repo_root(), cfg["gold_tsv"]), sep="\t", dtype=str)
    g["start_second"] = g["start_second"].astype(float); g["end_second"] = g["end_second"].astype(float)
    return g.set_index("id")

def dev_subset_ids(df_dev, n, seed=8001):
    """A reproducible subset of the dev slice (sorted by original_index) for method development; never test or load pool."""
    d = df_dev.sort_values("original_index").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return d.iloc[rng.choice(len(d), n, replace=False)]["file"].tolist()

def write_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=1, default=float)
