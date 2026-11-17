import os
import json
import io
import torch
import pandas as pd

from pathlib import Path
from transformers import pipeline
from minio import Minio
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score

"""# **Extra models config**"""

ASR_TEST = "data/asr_test.csv"
SENTIMENT_TEST = "data/sentiment_test.csv"

OUTPUT_DIR = "variants"

REF = "openai/whisper-large-v3"
ASR_MODELS = [
    REF,
    "AventIQ-AI/whisper-audio-to-text",
    "facebook/s2t-small-librispeech-asr",
    "openai/whisper-tiny"
]

SENTIMENT_MODELS = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularisai/multilingual-sentiment-analysis",
    "siebert/sentiment-roberta-large-english",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
]

"""# **Connect to MinIO**"""

client = Minio("localhost:9000", access_key="minioadmin", secret_key="ChangeThisPassword123!", secure=False)
BUCKET = "models"

if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)
print("Connected to MinIO!")

def get_model_size(directory):
    return sum(os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(directory)
        for file in files
    )

def upload_model(directory, stage, name, source, metrics):
    size = get_model_size(directory)
    metadata = {
        "name": name,
        "version": "v1",
        "source": source,
        "size_bytes": size,
        "offline_metrics": metrics
    }

    prefix = f"{stage}/{name}/v1"

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            relative = os.path.relpath(path, directory)
            client.fput_object(BUCKET, f"{prefix}/{relative}", path)

    data = json.dumps(metadata, indent=2).encode("utf-8")

    client.put_object(BUCKET, f"{prefix}/meta.json", io.BytesIO(data), len(data), content_type="application/json")

    print(f"Uploaded {name}: {size} bytes")

"""# **ASR Variants**"""

def run_asr():

    df = pd.read_csv(ASR_TEST)
    assert "audio_path" in df.columns
    assert "text" in df.columns
    assert len(df) > 0
    references = df["text"].astype(str).tolist()
    for audio_path in df["audio_path"]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    for source in ASR_MODELS:
        name = source.split("/")[-1]
        print(f"\nRunning ASR: {name}")

        model = pipeline("automatic-speech-recognition", model=source, device=-1)

        output_dir = f"{OUTPUT_DIR}/asr/{name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save model and associated files
        model.model.save_pretrained(output_dir)

        if model.tokenizer:
            model.tokenizer.save_pretrained(output_dir)

        if model.feature_extractor:
            model.feature_extractor.save_pretrained(output_dir)

        predictions = []

        for audio_path in df["audio_path"]:
            audio_path = Path(audio_path)
            result = model(str(audio_path))
            predictions.append(result["text"])

        score = wer(references, predictions)

        metrics = { "wer": float(score) }

        upload_model( output_dir, "asr", name, source, metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""# **Sentiment model**"""

def run_sentiment():

    df = pd.read_csv(SENTIMENT_TEST)
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0

    actual = [str(label).strip().lower() for label in df["label"]]

    
    for source in SENTIMENT_MODELS:
        name = source.split("/")[-1]

        print(f"\nRunning sentiment: {name}")

        model = pipeline("text-classification", model=source, device=-1)

        output_dir = f"{OUTPUT_DIR}/sentiment/{name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save model and tokenizer
        model.model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)

        predictions = []

        for text in df["text"].astype(str):
            result = model(text, truncation=True)[0]
            predictions.append(result["label"].strip().lower())

        assert len(predictions) == len(actual)
        if set(predictions) != set(actual):
            raise ValueError( f"Label mismatch for {name}.\n" f"Model labels: {set(predictions)}\n" f"Test labels: {set(actual)}\n" "Add an appropriate label mapping before evaluation." )
        metrics = {
            "accuracy": float(accuracy_score(actual, predictions)),
            "macro_f1": float(f1_score(actual, predictions, average="macro"))
        }

        upload_model(output_dir, "sentiment", name, source, metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_asr()
    run_sentiment()
    print("\nAll variants evaluated and uploaded!")