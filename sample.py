import datasets
import transformers
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import pandas
from tqdm.auto import tqdm
from torch.utils.flop_counter import FlopCounterMode
import argparse
import gc
import os
import random
import time
import platform

"""# Performance helpers"""

def _cuda_sync():
    """GPU work is asynchronous, so wait for it before reading the clock."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")

def _release_gpu():
    """Call after `del`-ing the last reference to a model: collect cycles, then hand cached
    blocks back to the driver. empty_cache alone cannot free memory a live object still holds,
    and a helper that takes the model as a parameter would only drop its own reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class StageTimer:
    def __enter__(self):
        _cuda_sync()
        self.start = time.perf_counter()
        return self
    def __exit__(self, *exc):
        _cuda_sync()
        self.seconds = time.perf_counter() - self.start

def print_environment():
    print("=== Environment ===")
    print(f"OS:            {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:        {platform.python_version()}")
    print(f"torch:         {torch.__version__}  (built for CUDA {torch.version.cuda})")
    print(f"transformers:  {transformers.__version__}   datasets: {datasets.__version__}   pandas: {pandas.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:           {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
        print(f"cuDNN:         {torch.backends.cudnn.version()}")
        print(f"GPU count:     {torch.cuda.device_count()}")
    else:
        print("GPU:           none detected, running on CPU")
    print(f"CPU threads:   {torch.get_num_threads()}")
    print()

"""# Models and labels"""

ASR_MODELS = [
    "AventIQ-AI/whisper-audio-to-text",
    "facebook/s2t-small-librispeech-asr"
]

SENTIMENT_MODELS = [
    "siebert/sentiment-roberta-large-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularisai/multilingual-sentiment-analysis"
]

# Clips per forward pass. Measured on 64 clips: batching gave no speed-up for Whisper
# (10.2 vs 9.7 clips/s) or SiEBERT (162 vs 86 texts/s) because short inputs get padded to
# the longest in the batch, and it changed 14 of 64 Speech2Text transcripts. Keep at 1.
ASR_BATCH_SIZE = 1
SENTIMENT_BATCH_SIZE = 1

# Every model's label vocabulary, folded onto the dataset's. Lower-cased on both sides.
#   siebert:     NEGATIVE, POSITIVE
#   cardiffnlp:  negative, neutral, positive
#   tabularisai: Very Negative, Negative, Neutral, Positive, Very Positive
#   dataset:     Neutral, Positive, Negative, Disagreement, <mixed>
LABEL_MAP = {
    "very negative": "negative",
    "very positive": "positive",
}

# "disagreement" and "<mixed>" (127 of 3,553 clips) are annotator-level labels no model
# can emit, so every model scores 0 on them. They stay in the scoring; the per-class
# tables make that visible. Per the SLUE paper (arXiv 2111.10367) each test utterance was
# labelled by 5 annotators and "disagreement" marks clips where fewer than 3 agreed.

def normalise_label(label: str) -> str:
    label = str(label).strip().lower()
    return LABEL_MAP.get(label, label)

"""# Iron law of ML inference: T = D_vol / BW + O / (R_peak * eta) [+ L_lat]

D_vol  bytes moved per stage: every parameter tensor is counted each time its module runs
       (so each decoder step re-reads the decoder weights) plus the raw input bytes.
       Activation traffic is not counted, so D_vol is a lower bound.
O      floating-point operations, counted by torch's FlopCounterMode (1 FMA = 2 FLOP).
BW     GPU memory bandwidth; R_peak the peak dense throughput for the dtype the model ran in.
eta    hardware efficiency, solved from the measured time: eta = O / (R_peak * (T - D_vol / BW)).
Both counts are measured on a calibration subset of clips, outside the timed region, and
scaled to the full clip count."""

CALIBRATION_CLIPS = 32

GPU_SPECS = {
    # Substring of torch.cuda.get_device_name() -> bandwidth and peak dense TFLOPS per dtype.
    "RTX 4000 Ada": {
        # NVIDIA RTX 4000 Ada datasheet: 160-bit memory interface, 360 GB/s, 26.7 TFLOPS single
        # precision, "327.6 TFLOPS tensor" with footnote "Effective FP8 TFLOPS using sparsity".
        # The Ada whitepaper's RTX 4090 table puts FP8-sparse at 4x FP16-dense (1321.2 vs 330.3),
        # so FP16 dense = 327.6 / 4 = 81.9 TFLOPS. An 8192^2 fp16 matmul reached 71 TFLOPS here.
        "bandwidth_gbs": 360.0,
        "peak_tflops": {"float32": 26.7, "float16": 81.9, "bfloat16": 81.9},
    },
}

def resolve_gpu_specs(device_name: str, bandwidth_gbs=None, peak_tflops_fp32=None, peak_tflops_fp16=None) -> dict:
    specs = next((dict(v) for k, v in GPU_SPECS.items() if k.lower() in device_name.lower()), None)
    if specs is None:
        specs = {"bandwidth_gbs": float("nan"), "peak_tflops": {}}
        print(f"No GPU_SPECS entry for '{device_name}'. Pass --bandwidth-gbs / --peak-tflops-fp32 / --peak-tflops-fp16.")
    specs["peak_tflops"] = dict(specs["peak_tflops"])
    if bandwidth_gbs:
        specs["bandwidth_gbs"] = bandwidth_gbs
    if peak_tflops_fp32:
        specs["peak_tflops"]["float32"] = peak_tflops_fp32
    if peak_tflops_fp16:
        specs["peak_tflops"]["float16"] = specs["peak_tflops"]["bfloat16"] = peak_tflops_fp16
    return specs

class WorkloadMeter:
    """Counts FLOPs (FlopCounterMode) and bytes of weights read (a forward hook on every module
    that owns parameters) for whatever runs inside the `with` block."""
    def __init__(self, model):
        self.model = model
        self.flops = 0
        self.weight_bytes = 0
        self._hooks = []

    def _hook(self, nbytes):
        def count(module, inputs, output):
            self.weight_bytes += nbytes
        return count

    def __enter__(self):
        for module in self.model.modules():
            direct = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            if direct:
                self._hooks.append(module.register_forward_hook(self._hook(direct)))
        self._counter = FlopCounterMode(display=False)
        self._counter.__enter__()
        return self

    def __exit__(self, *exc):
        self._counter.__exit__(*exc)
        self.flops = self._counter.get_total_flops()
        for hook in self._hooks:
            hook.remove()

def _audio_bytes(audios) -> int:
    return sum(len(audio["array"]) * audio["array"].itemsize for audio in audios)

def _text_bytes(tokenizer, texts) -> int:
    return sum(len(ids) for ids in tokenizer(list(texts))["input_ids"]) * 8   # int64 token ids

def measure_workload(pipe, inputs: list, run, input_bytes, n_total: int) -> dict:
    """Run `run(pipe, subset)` on a calibration subset under the meter, scale to n_total clips."""
    if len(inputs) > CALIBRATION_CLIPS:
        subset = random.Random(0).sample(inputs, CALIBRATION_CLIPS)
    else:
        subset = list(inputs)
    with WorkloadMeter(pipe.model) as meter:
        run(pipe, subset)
    scale = n_total / len(subset)
    return {
        "dtype": str(next(pipe.model.parameters()).dtype).replace("torch.", ""),
        "gflop": meter.flops * scale / 1e9,
        "data_gb": (meter.weight_bytes + input_bytes(subset)) * scale / 1e9,
        "calibration_clips": len(subset),
    }

"""# Data"""

class DatasetLoader:
    def __init__(self, dataset_name="DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb", split="test"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> pandas.DataFrame:
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            print(dataset)
            return pandas.DataFrame(dataset)
        except Exception as error:
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {error}") from error

"""# Inference"""

def get_device() -> int:
    return 0 if torch.cuda.is_available() else -1

def build_recognizer(asr_model: str, device: int):
    """Multilingual Whisper checkpoints run language detection on every clip unless told the
    language. SLUE-VoxCeleb is the English portion of the VoxCeleb1 test set (SLUE paper,
    arXiv 2111.10367), so pin it; other ASR models take no such option."""
    kwargs = {}
    if "whisper" in asr_model.lower():
        kwargs["generate_kwargs"] = {"language": "en", "task": "transcribe"}
    return pipeline("automatic-speech-recognition", model=asr_model, device=device, **kwargs)

def pair_name(asr_model: str, sentiment_model: str) -> str:
    return f"ASR: {asr_model}. Sentiment: {sentiment_model}"

def transcribe(recognizer, audios, batch_size: int, desc: str = "ASR") -> list:
    # A generator input makes the pipeline stream results, so tqdm can show progress.
    outputs = recognizer((audio for audio in audios), batch_size=batch_size)
    return [out["text"] for out in tqdm(outputs, total=len(audios), desc=desc)]

def classify(analyzer, texts, batch_size: int, desc: str = "sentiment") -> list:
    outputs = analyzer((text for text in texts), batch_size=batch_size, truncation=True, max_length=512)
    return [(out["label"], out["score"]) for out in tqdm(outputs, total=len(texts), desc=desc)]

def run_analyzers(limit: int = None) -> dict:
    """Run every ASR x sentiment pair. Each ASR model transcribes the clips once and the
    transcripts are shared by the three sentiment models, so the run costs 2 ASR passes
    instead of 6. Peak GPU memory is measured per stage with only that stage's model loaded."""
    original_dataframe = DatasetLoader().load()
    if limit:
        original_dataframe = original_dataframe.head(limit)
    n = len(original_dataframe)
    device = get_device()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    models_results = {}

    for asr_model in ASR_MODELS:
        print(f"\n=== ASR: {asr_model}  ({n} clips) ===")
        _reset_peak_memory()
        with StageTimer() as asr_load_timer:
            recognizer = build_recognizer(asr_model, device)
        audios = list(original_dataframe["audio"])
        with StageTimer() as asr_timer:
            transcripts = transcribe(recognizer, audios, ASR_BATCH_SIZE)
        asr_peak_mb = _peak_memory_mb()
        asr_work = measure_workload(
            recognizer, audios,
            lambda pipe, subset: transcribe(pipe, subset, ASR_BATCH_SIZE, desc="ASR iron-law calibration"),
            _audio_bytes, n)
        del recognizer
        _release_gpu()

        for sentiment_model in SENTIMENT_MODELS:
            name = pair_name(asr_model, sentiment_model)
            print(f"\n--- {name} ---")
            _reset_peak_memory()
            with StageTimer() as sentiment_load_timer:
                analyzer = pipeline("sentiment-analysis", model=sentiment_model, device=device)
            with StageTimer() as sentiment_timer:
                predictions = classify(analyzer, transcripts, SENTIMENT_BATCH_SIZE)
            sentiment_peak_mb = _peak_memory_mb()
            sentiment_work = measure_workload(
                analyzer, transcripts,
                lambda pipe, subset: classify(pipe, subset, SENTIMENT_BATCH_SIZE, desc="sentiment iron-law calibration"),
                lambda subset: _text_bytes(analyzer.tokenizer, subset), n)
            del analyzer
            _release_gpu()

            results = pandas.DataFrame({
                "audio": original_dataframe["audio"],
                "actual_sentiment": original_dataframe["label"],
                "transcription": transcripts,
                "predicted_sentiment": [label for label, _ in predictions],
                "sentiment_score": [score for _, score in predictions],
            }, index=original_dataframe.index)
            results.attrs["performance"] = {
                "device": device_name,
                "clips": n,
                "asr_batch_size": ASR_BATCH_SIZE,
                "sentiment_batch_size": SENTIMENT_BATCH_SIZE,
                "asr_load_s": asr_load_timer.seconds,
                "asr_s": asr_timer.seconds,
                "asr_clips_per_s": n / asr_timer.seconds if asr_timer.seconds else float("nan"),
                "asr_peak_mb": asr_peak_mb,
                "asr_dtype": asr_work["dtype"],
                "asr_gflop": asr_work["gflop"],
                "asr_data_gb": asr_work["data_gb"],
                "asr_calibration_clips": asr_work["calibration_clips"],
                "sentiment_load_s": sentiment_load_timer.seconds,
                "sentiment_s": sentiment_timer.seconds,
                "sentiment_per_s": n / sentiment_timer.seconds if sentiment_timer.seconds else float("nan"),
                "sentiment_peak_mb": sentiment_peak_mb,
                "sentiment_dtype": sentiment_work["dtype"],
                "sentiment_gflop": sentiment_work["gflop"],
                "sentiment_data_gb": sentiment_work["data_gb"],
                "sentiment_calibration_clips": sentiment_work["calibration_clips"],
                # The ASR stage is shared by the three pairs that use the same ASR model,
                # so total_s is what this pair would cost if run on its own.
                "total_s": asr_load_timer.seconds + asr_timer.seconds
                           + sentiment_load_timer.seconds + sentiment_timer.seconds,
            }
            models_results[name] = results
            print(results[["actual_sentiment", "transcription", "predicted_sentiment", "sentiment_score"]].head())
    return models_results

"""# Results"""

def _add_scoring_columns(dataframe: pandas.DataFrame):
    if "correct" in dataframe:
        return
    dataframe["actual_sentiment_norm"] = dataframe["actual_sentiment"].map(normalise_label)
    dataframe["predicted_sentiment_norm"] = dataframe["predicted_sentiment"].map(normalise_label)
    dataframe["correct"] = dataframe["actual_sentiment_norm"] == dataframe["predicted_sentiment_norm"]

def per_class_table(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """One row per dataset class: precision, recall, F1, accuracy on that class, support."""
    _add_scoring_columns(dataframe)
    actual, predicted = dataframe["actual_sentiment_norm"], dataframe["predicted_sentiment_norm"]
    classes = sorted(actual.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, average=None, zero_division=0)
    table = pandas.DataFrame({
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": dataframe.groupby("actual_sentiment_norm")["correct"].mean().reindex(classes).values,
        "support": support,
    }, index=pandas.Index(classes, name="class"))
    table.loc["macro avg"] = [precision.mean(), recall.mean(), f1.mean(), float("nan"), int(support.sum())]
    table["support"] = table["support"].astype(int)
    return table

def score_pair(dataframe: pandas.DataFrame) -> dict:
    """Pair-level accuracy, macro precision / recall / F1 over all dataset classes, and the
    per-class F1 values, flattened so they fit one row of metrics.csv."""
    table = per_class_table(dataframe)
    macro = table.loc["macro avg"]
    scores = {
        "accuracy": dataframe["correct"].mean(),
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "macro_f1": macro["f1"],
    }
    for label, row in table.drop(index="macro avg").iterrows():
        scores[f"f1_{label}"] = row["f1"]
    return scores

def report_results(models_results: dict, output_path: str = "metrics.csv",
                   per_class_path: str = "per_class_metrics.csv") -> pandas.DataFrame:
    rows, per_class_rows = [], []
    for model_name, dataframe in models_results.items():
        table = per_class_table(dataframe)
        scores = score_pair(dataframe)
        rows.append({"model_pair": model_name, **scores})
        per_class_rows.append(table.assign(model_pair=model_name).reset_index())

        print(f"{model_name}")
        print(f"  accuracy {scores['accuracy']:.2%}   macro precision {scores['precision_macro']:.3f}   "
              f"macro recall {scores['recall_macro']:.3f}   macro F1 {scores['macro_f1']:.3f}")
        with pandas.option_context("display.float_format", "{:.3f}".format):
            print(table.to_string())
        print()

    metrics = pandas.DataFrame(rows).set_index("model_pair")
    metrics.to_csv(output_path)
    print(f"Metrics table saved to {output_path}")
    per_class = pandas.concat(per_class_rows, ignore_index=True)[
        ["model_pair", "class", "precision", "recall", "f1", "accuracy", "support"]]
    per_class.to_csv(per_class_path, index=False)
    print(f"Per-class tables saved to {per_class_path}")
    return metrics

"""# Confidence"""

def report_confidence(models_results: dict, output_path: str = "confidence.csv") -> pandas.DataFrame:
    """Mean sentiment model confidence per pair: the pipeline's softmax probability for the
    predicted label (the `sentiment_score` column), averaged over all clips, and split by
    whether the prediction was correct. A model that is as confident when wrong as when
    right is not giving you a usable confidence signal."""
    rows = []
    for model_name, dataframe in models_results.items():
        _add_scoring_columns(dataframe)
        score = dataframe["sentiment_score"]
        rows.append({
            "model_pair": model_name,
            "mean_confidence": score.mean(),
            "std_confidence": score.std(),
            "mean_confidence_correct": score[dataframe["correct"]].mean(),
            "mean_confidence_incorrect": score[~dataframe["correct"]].mean(),
            "share_above_0.9": (score > 0.9).mean(),
        })
        dataframe.attrs.setdefault("confidence", {}).update(rows[-1])
    confidence = pandas.DataFrame(rows).set_index("model_pair")

    print("=== Mean sentiment model confidence score ===")
    for model_name, row in confidence.iterrows():
        print(f"{model_name}")
        print(f"  mean confidence:         {row['mean_confidence']:.4f}  (std {row['std_confidence']:.4f})")
        print(f"  when correct / incorrect: {row['mean_confidence_correct']:.4f} / {row['mean_confidence_incorrect']:.4f}")
        print(f"  share of clips > 0.90:   {row['share_above_0.9']:.1%}")
        print()
    confidence.to_csv(output_path)
    print(f"Confidence table saved to {output_path}")
    return confidence

"""# Performance"""

PERFORMANCE_COLUMNS = ["device", "clips", "asr_batch_size", "sentiment_batch_size",
                       "asr_load_s", "asr_s", "asr_clips_per_s", "asr_peak_mb",
                       "asr_dtype", "asr_gflop", "asr_data_gb", "asr_calibration_clips",
                       "sentiment_load_s", "sentiment_s", "sentiment_per_s", "sentiment_peak_mb",
                       "sentiment_dtype", "sentiment_gflop", "sentiment_data_gb", "sentiment_calibration_clips",
                       "total_s"]

def report_performance(models_results: dict, output_path: str = "performance.csv") -> pandas.DataFrame:
    rows = []
    for model_name, dataframe in models_results.items():
        perf = dataframe.attrs.get("performance")
        if perf:
            rows.append({"model_pair": model_name, **perf})
    if not rows:
        print("No performance data recorded.")
        return pandas.DataFrame()

    performance = pandas.DataFrame(rows).set_index("model_pair")[PERFORMANCE_COLUMNS]

    print("=== Performance (seconds unless noted; ASR stage shared across pairs with the same ASR model) ===")
    with pandas.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", None,
                               "display.float_format", "{:.2f}".format):
        print(performance)
    print()
    performance.to_csv(output_path)
    print(f"Performance table saved to {output_path}")
    return performance

"""# Iron law report"""

def _iron_law_row(t_s, gflop, data_gb, bandwidth_gbs, peak_tflops) -> dict:
    t_memory = data_gb / bandwidth_gbs
    t_compute_ideal = gflop / 1e3 / peak_tflops
    t_iron_law = t_memory + t_compute_ideal
    intensity = gflop / data_gb if data_gb else float("nan")
    ridge = peak_tflops * 1e3 / bandwidth_gbs
    compute_budget = t_s - t_memory
    return {
        "t_measured_s": t_s,
        "data_gb": data_gb,
        "gflop": gflop,
        "intensity_flop_per_byte": intensity,
        "ridge_point_flop_per_byte": ridge,
        "bound": "memory" if intensity < ridge else "compute",
        "t_memory_s": t_memory,
        "t_compute_ideal_s": t_compute_ideal,
        "t_iron_law_eta1_s": t_iron_law,
        "efficiency_eta": gflop / 1e3 / (peak_tflops * compute_budget) if compute_budget > 0 else float("nan"),
        "achieved_tflops": gflop / 1e3 / t_s if t_s else float("nan"),
        "achieved_gbs": data_gb / t_s if t_s else float("nan"),
        "overhead_vs_eta1_s": t_s - t_iron_law,
    }

def report_iron_law(performance: pandas.DataFrame, specs: dict, output_path: str = "iron_law.csv") -> pandas.DataFrame:
    """Per pair and per stage: the iron-law decomposition of the measured inference time."""
    if performance.empty:
        return pandas.DataFrame()
    bandwidth = specs["bandwidth_gbs"]
    rows = []
    for model_name, perf in performance.iterrows():
        stage_rows = {}
        for stage in ("asr", "sentiment"):
            peak = specs["peak_tflops"].get(perf[f"{stage}_dtype"], float("nan"))
            stage_rows[stage] = {
                "model_pair": model_name, "stage": stage, "dtype": perf[f"{stage}_dtype"],
                "bandwidth_gbs": bandwidth, "peak_tflops": peak,
                **_iron_law_row(perf[f"{stage}_s"], perf[f"{stage}_gflop"], perf[f"{stage}_data_gb"], bandwidth, peak),
            }
        a, b = stage_rows["asr"], stage_rows["sentiment"]
        # The pair as a whole: sum the measured, memory and ideal-compute times of the two stages.
        # Stages may run in different dtypes, so the pair's efficiency is defined through its summed
        # ideal compute time: eta = t_compute_ideal / (t_measured - t_memory).
        t_s, t_mem, t_ideal = (a[k] + b[k] for k in ("t_measured_s", "t_memory_s", "t_compute_ideal_s"))
        total = {
            "model_pair": model_name, "stage": "total", "dtype": f"{a['dtype']}+{b['dtype']}",
            "bandwidth_gbs": bandwidth, "peak_tflops": float("nan"),
            "t_measured_s": t_s, "data_gb": a["data_gb"] + b["data_gb"], "gflop": a["gflop"] + b["gflop"],
            "intensity_flop_per_byte": (a["gflop"] + b["gflop"]) / (a["data_gb"] + b["data_gb"]),
            "ridge_point_flop_per_byte": float("nan"), "bound": "memory" if t_mem > t_ideal else "compute",
            "t_memory_s": t_mem, "t_compute_ideal_s": t_ideal, "t_iron_law_eta1_s": t_mem + t_ideal,
            "efficiency_eta": t_ideal / (t_s - t_mem) if t_s > t_mem else float("nan"),
            "achieved_tflops": (a["gflop"] + b["gflop"]) / 1e3 / t_s,
            "achieved_gbs": (a["data_gb"] + b["data_gb"]) / t_s,
            "overhead_vs_eta1_s": t_s - t_mem - t_ideal,
        }
        rows += [a, b, total]
    iron = pandas.DataFrame(rows).set_index(["model_pair", "stage"])

    print("=== Iron law of inference:  T = D_vol / BW + O / (R_peak * eta) ===")
    print(f"BW = {bandwidth:.0f} GB/s, R_peak = " +
          ", ".join(f"{k} {v:.1f} TFLOPS" for k, v in specs["peak_tflops"].items()) +
          "   (eta solved from the measured time; D_vol excludes activations)")
    show = ["dtype", "t_measured_s", "data_gb", "gflop", "intensity_flop_per_byte", "bound",
            "t_memory_s", "t_compute_ideal_s", "t_iron_law_eta1_s", "efficiency_eta", "achieved_tflops"]
    with pandas.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", None,
                               "display.float_format", "{:.3f}".format):
        shown = iron[show].copy()
        shown.index = pandas.MultiIndex.from_tuples(
            [(_short_model_name(n).replace("\n", " "), st) for n, st in shown.index], names=["model_pair", "stage"])
        print(shown)
    print()
    iron.to_csv(output_path)
    print(f"Iron-law table saved to {output_path}")
    return iron

"""# Figure"""

def _short_model_name(model_names: str) -> str:
    """Turn the long pair name into two short lines for axis labels."""
    asr, sentiment = model_names.split(". Sentiment: ")
    asr = asr.replace("ASR: ", "").split("/")[-1]
    sentiment = sentiment.split("/")[-1]
    return f"{asr}\n+ {sentiment}"

def plot_results(models_results: dict, output_path: str = "results.png"):
    """Accuracy and macro F1 per pair on top, per-class accuracy heatmap below."""
    blue, orange = "#2a78d6", "#eb6834"
    text_primary, text_secondary, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])

    names = list(models_results.keys())
    labels = [_short_model_name(n) for n in names]

    accuracy, macro_f1, per_class = [], [], {}
    for name in names:
        dataframe = models_results[name]
        scores = score_pair(dataframe)
        accuracy.append(scores["accuracy"])
        macro_f1.append(scores["macro_f1"])
        per_class[name] = dataframe.groupby("actual_sentiment_norm")["correct"].mean()

    # Accuracy you would get by always answering the most common label. Every pair
    # scores the same clips, so the true labels of any pair are the dataset's labels.
    actual = models_results[names[0]]["actual_sentiment_norm"]
    class_counts = actual.value_counts()
    baseline = class_counts.iloc[0] / len(actual)
    classes = list(class_counts.index)

    per_class_table = pandas.DataFrame(per_class).T.reindex(index=names, columns=classes)

    fig, (ax_bars, ax_heat) = plt.subplots(
        2, 1, figsize=(10, 4.2 + 0.9 * len(names)),
        gridspec_kw={"height_ratios": [1.1, 1]}, constrained_layout=True
    )
    fig.patch.set_facecolor("#fcfcfb")

    # Top panel
    y = list(range(len(names)))
    height = 0.36
    ax_bars.barh([i + height / 2 for i in y], accuracy, height, color=blue, label="Accuracy (all clips)")
    ax_bars.barh([i - height / 2 for i in y], macro_f1, height, color=orange, label="Macro F1 (all classes)")
    for i, (a, f) in enumerate(zip(accuracy, macro_f1)):
        ax_bars.text(a + 0.01, i + height / 2, f"{a:.1%}", va="center", fontsize=9, color=text_primary)
        ax_bars.text(f + 0.01, i - height / 2, f"{f:.2f}", va="center", fontsize=9, color=text_primary)
    ax_bars.axvline(baseline, color=text_secondary, linestyle="--", linewidth=1)
    ax_bars.text(baseline, len(names) - 0.45, f" majority baseline {baseline:.0%}",
                 fontsize=8.5, color=text_secondary, va="bottom")
    ax_bars.set_yticks(y, labels, fontsize=9, color=text_primary)
    ax_bars.invert_yaxis()
    ax_bars.set_xlim(0, 1.0)
    ax_bars.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], color=text_secondary)
    ax_bars.grid(axis="x", color=grid, linewidth=0.8)
    ax_bars.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_bars.spines[side].set_visible(False)
    ax_bars.spines["bottom"].set_color(grid)
    ax_bars.tick_params(length=0)
    ax_bars.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9, borderaxespad=0.6)
    ax_bars.set_title("Overall accuracy vs macro F1 over all dataset classes (higher is better)",
                      loc="left", fontsize=11, color=text_primary, fontweight="bold", pad=18)

    # Bottom panel
    values = per_class_table.to_numpy(dtype=float)
    ax_heat.imshow(values, cmap=blues, vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(values.shape[1]),
                       [f"{c}\n(n={class_counts[c]})" for c in per_class_table.columns],
                       fontsize=9, color=text_primary)
    ax_heat.set_yticks(range(values.shape[0]), labels, fontsize=9, color=text_primary)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            ax_heat.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=9,
                         color="#ffffff" if v > 0.55 else text_primary)
    # Minor gridlines in the surface colour draw the gaps between cells
    ax_heat.set_xticks([x - 0.5 for x in range(1, values.shape[1])], minor=True)
    ax_heat.set_yticks([y_ - 0.5 for y_ in range(1, values.shape[0])], minor=True)
    ax_heat.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax_heat.tick_params(which="both", length=0)
    for side in ax_heat.spines.values():
        side.set_visible(False)
    ax_heat.set_title("Per-class accuracy after mapping model labels onto dataset labels",
                      loc="left", fontsize=11, color=text_primary, fontweight="bold")

    fig.suptitle("Speech-to-sentiment pipeline: ASR model + sentiment model",
                 fontsize=13, color=text_primary, fontweight="bold", x=0.01, ha="left")
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Figure saved to {output_path}")
    plt.show()
    return fig

def plot_performance(performance, output_path: str = "performance.png"):
    """Runtime by stage, ASR throughput, sentiment throughput, and peak GPU memory per stage.
    Accepts the performance DataFrame or a path to the saved CSV."""
    if isinstance(performance, str):
        performance = pandas.read_csv(performance, index_col="model_pair")
    if performance.empty:
        print("No performance data to plot.")
        return None

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    text_primary, text_secondary, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    labels = [_short_model_name(n) for n in performance.index]
    y = list(range(len(labels)))
    device = str(performance["device"].iloc[0])
    clips = int(performance["clips"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(15, 3.5 + 0.75 * len(labels)), constrained_layout=True)
    fig.patch.set_facecolor(surface)
    ax_time, ax_asr, ax_sent, ax_mem = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def style(ax, xlabel):
        ax.set_yticks(y, labels, fontsize=8.5, color=text_primary)
        ax.invert_yaxis()
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(length=0, labelsize=8.5, colors=text_secondary)
        ax.set_xlabel(xlabel, fontsize=9, color=text_secondary)

    # Runtime by stage
    load_m = (performance["asr_load_s"] + performance["sentiment_load_s"]) / 60
    asr_m, sent_m = (performance[c] / 60 for c in ("asr_s", "sentiment_s"))
    total_m = load_m + asr_m + sent_m
    ax_time.barh(y, asr_m, 0.6, color=blue, label="ASR (transcription)", edgecolor=surface, linewidth=2)
    ax_time.barh(y, sent_m, 0.6, left=asr_m, color=orange, label="Sentiment", edgecolor=surface, linewidth=2)
    ax_time.barh(y, load_m, 0.6, left=asr_m + sent_m, color=aqua, label="Model load", edgecolor=surface, linewidth=2)
    offset = total_m.max() * 0.02
    for i in y:
        share = asr_m.iloc[i] / total_m.iloc[i]
        ax_time.text(total_m.iloc[i] + offset, i, f"{total_m.iloc[i]:.1f} min  (ASR {share:.0%})",
                     va="center", fontsize=8.5, color=text_primary)
    ax_time.set_xlim(0, total_m.max() * 1.45)
    style(ax_time, f"minutes for {clips:,} clips (ASR stage shared by pairs with the same ASR model)")
    ax_time.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    ax_time.set_title("Runtime per model pair, by stage", loc="left", fontsize=11,
                      color=text_primary, fontweight="bold", pad=18)

    # ASR throughput
    asr_batch = int(performance["asr_batch_size"].iloc[0])
    ax_asr.barh(y, performance["asr_clips_per_s"], 0.6, color=blue)
    for i, v in enumerate(performance["asr_clips_per_s"]):
        ax_asr.text(v + performance["asr_clips_per_s"].max() * 0.015, i, f"{v:.1f}", va="center", fontsize=8.5, color=text_primary)
    ax_asr.set_xlim(0, performance["asr_clips_per_s"].max() * 1.2)
    style(ax_asr, f"audio clips transcribed per second (batch size {asr_batch})")
    ax_asr.set_title("ASR throughput (higher is faster)", loc="left", fontsize=11,
                     color=text_primary, fontweight="bold")

    # Sentiment throughput
    sent_batch = int(performance["sentiment_batch_size"].iloc[0])
    ax_sent.barh(y, performance["sentiment_per_s"], 0.6, color=orange)
    for i, v in enumerate(performance["sentiment_per_s"]):
        ax_sent.text(v + performance["sentiment_per_s"].max() * 0.015, i, f"{v:.0f}", va="center", fontsize=8.5, color=text_primary)
    ax_sent.set_xlim(0, performance["sentiment_per_s"].max() * 1.2)
    style(ax_sent, f"transcripts classified per second (batch size {sent_batch})")
    ax_sent.set_title("Sentiment throughput (higher is faster)", loc="left", fontsize=11,
                      color=text_primary, fontweight="bold")

    # Peak GPU memory, one bar per stage
    asr_gb = performance["asr_peak_mb"] / 1024
    sent_gb = performance["sentiment_peak_mb"] / 1024
    if asr_gb.notna().any() or sent_gb.notna().any():
        h = 0.36
        ax_mem.barh([i - h / 2 for i in y], asr_gb, h, color=blue, label="ASR stage")
        ax_mem.barh([i + h / 2 for i in y], sent_gb, h, color=orange, label="Sentiment stage")
        for i, (a, s) in enumerate(zip(asr_gb, sent_gb)):
            ax_mem.text(a + 0.04, i - h / 2, f"{a:.2f} GB", va="center", fontsize=8, color=text_primary)
            ax_mem.text(s + 0.04, i + h / 2, f"{s:.2f} GB", va="center", fontsize=8, color=text_primary)
        ax_mem.set_xlim(0, max(asr_gb.max(), sent_gb.max()) * 1.25)
        style(ax_mem, "peak GPU memory allocated (GB), that stage's model only")
        ax_mem.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
        ax_mem.set_title("Peak GPU memory per stage", loc="left", fontsize=11,
                         color=text_primary, fontweight="bold", pad=18)
    else:
        ax_mem.axis("off")
        ax_mem.text(0.0, 0.5, "No GPU memory data (run on CPU)", fontsize=10, color=text_secondary)

    fig.suptitle(f"Pipeline performance on {device}", fontsize=13, color=text_primary,
                 fontweight="bold", x=0.01, ha="left")
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Figure saved to {output_path}")
    plt.show()
    return fig

"""# Main"""

def main():
    parser = argparse.ArgumentParser(description="Benchmark every ASR x sentiment pair on SLUE-VoxCeleb.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N clips (smoke test); omit for the full test split")
    parser.add_argument("--output-dir", default=".",
                        help="where metrics.csv, confidence.csv, performance.csv, iron_law.csv and the figures go")
    parser.add_argument("--bandwidth-gbs", type=float, default=None, help="override GPU memory bandwidth (GB/s)")
    parser.add_argument("--peak-tflops-fp32", type=float, default=None, help="override peak FP32 TFLOPS")
    parser.add_argument("--peak-tflops-fp16", type=float, default=None, help="override peak dense FP16/BF16 TFLOPS")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda filename: os.path.join(args.output_dir, filename)

    print_environment()
    with StageTimer() as wall:
        models_results = run_analyzers(limit=args.limit)
    print(f"\nWall-clock for all pairs, including dataset load and model loads: {wall.seconds / 60:.1f} min\n")
    report_results(models_results, out("metrics.csv"), out("per_class_metrics.csv"))
    report_confidence(models_results, out("confidence.csv"))
    performance = report_performance(models_results, out("performance.csv"))
    if not performance.empty:
        specs = resolve_gpu_specs(str(performance["device"].iloc[0]), args.bandwidth_gbs,
                                  args.peak_tflops_fp32, args.peak_tflops_fp16)
        report_iron_law(performance, specs, out("iron_law.csv"))
    plot_results(models_results, out("results.png"))
    plot_performance(performance, out("performance.png"))

if __name__ == "__main__":
    main()
