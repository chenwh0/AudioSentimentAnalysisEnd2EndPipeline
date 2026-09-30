

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import pandas
from tqdm.auto import tqdm
import time
import platform

"""# Performance helpers (work on any CUDA-enabled Linux box, and fall back cleanly on CPU)"""

def _cuda_sync():
    """GPU work is asynchronous; wait for it so timings measure real work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")

class StageTimer:
    """with StageTimer() as t: ...; then t.seconds"""
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
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:           {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
        print(f"cuDNN:         {torch.backends.cudnn.version()}")
        print(f"GPU count:     {torch.cuda.device_count()}")
    else:
        print("GPU:           none detected, running on CPU")
    print(f"CPU threads:   {torch.get_num_threads()}")
    print()

"""# Check if Cuda is available"""

device = 0 if torch.cuda.is_available() else -1

"""# Load the Dataset"""

try:
    dataset = load_dataset("DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb", split="test")
    print(dataset)
    original_dataframe = pandas.DataFrame(dataset)
    print(original_dataframe)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

# All possible label classes
print(original_dataframe["label"].unique())

"""# Load the Models"""

asr_models = [
    "AventIQ-AI/whisper-audio-to-text",
    "facebook/s2t-small-librispeech-asr"
]

sentiment_analysis_models = [
    "siebert/sentiment-roberta-large-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularisai/multilingual-sentiment-analysis"
]

asr = pipeline(
    "automatic-speech-recognition",
    model=asr_models[0],
    device=device
)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=sentiment_analysis_models[0],
    device=device
)

"""# Inference"""

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

class AudioSentimentAnalyzer:
    def __init__(self, asr_model_name: str, sentiment_model_name: str):
        self.asr_model_name = asr_model_name
        self.sentiment_model_name = sentiment_model_name

        self.device = self._get_device()
        with StageTimer() as load_timer:
            self.speech_recognizer = self._build_speech_recognizer()
            self.sentiment_analyzer = self._build_sentiment_analyzer()
        self.performance = {"load_s": load_timer.seconds}

    @staticmethod
    def _get_device() -> int:
        return 0 if torch.cuda.is_available() else -1
    def _build_speech_recognizer(self):
        return pipeline(
            "automatic-speech-recognition",
            model=self.asr_model_name,
            device=self.device
        )
    def _build_sentiment_analyzer(self):
        return pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )

    def transcribe(self, audio):
        return self.speech_recognizer(audio)["text"]

    def predict_sentiment(self, transcription: str):
        result = self.sentiment_analyzer(transcription, truncation=True, max_length=512)[0]
        return result["label"], result["score"]

    def analyze(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:

        results = pandas.DataFrame({"audio": dataframe["audio"], "actual_sentiment": dataframe["label"]})

        tqdm.pandas()
        n = len(results)

        _reset_peak_memory()
        with StageTimer() as asr_timer:
            results["transcription"] = results["audio"].progress_apply(self.transcribe)
        with StageTimer() as sentiment_timer:
            predictions = results["transcription"].progress_apply(self.predict_sentiment)
        results[["predicted_sentiment", "sentiment_score"]] = pandas.DataFrame(predictions.tolist(), index=results.index)

        self.performance.update({
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "clips": n,
            "asr_s": asr_timer.seconds,
            "asr_clips_per_s": n / asr_timer.seconds if asr_timer.seconds else float("nan"),
            "sentiment_s": sentiment_timer.seconds,
            "sentiment_per_s": n / sentiment_timer.seconds if sentiment_timer.seconds else float("nan"),
            "total_s": self.performance["load_s"] + asr_timer.seconds + sentiment_timer.seconds,
            "peak_gpu_mem_mb": _peak_memory_mb(),
        })
        results.attrs["performance"] = dict(self.performance)
        return results

ASR_MODELS = [
    "AventIQ-AI/whisper-audio-to-text",
    "facebook/s2t-small-librispeech-asr"
]

SENTIMENT_MODELS = [
    "siebert/sentiment-roberta-large-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularisai/multilingual-sentiment-analysis"
]

def run_analyzers():
    dataset_loader = DatasetLoader()
    original_dataframe = dataset_loader.load()
    models_results = {}
    for asr_model in ASR_MODELS:
        for sentiment_model in SENTIMENT_MODELS:
            model_names = f"ASR: {asr_model}. Sentiment: {sentiment_model}"
            print(model_names)
            analyzer = AudioSentimentAnalyzer(asr_model, sentiment_model)
            results = analyzer.analyze(original_dataframe)
            models_results[model_names] = results
            print(results)
            del analyzer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return models_results

"""# Results"""

def report_results(models_results: dict):
    # Compare labels case-insensitively so "Positive" and "POSITIVE" count as the same class
    for model_name, dataframe in models_results.items():
        dataframe["actual_sentiment_norm"] = dataframe["actual_sentiment"].str.lower()
        dataframe["predicted_sentiment_norm"] = dataframe["predicted_sentiment"].str.lower()
        dataframe["correct"] = (
            dataframe["actual_sentiment_norm"] == dataframe["predicted_sentiment_norm"]
        )

    for model_name, dataframe in models_results.items():
        accuracy = dataframe["correct"].mean()

        print(f"{model_name}")
        print(f"Accuracy: {accuracy:.2%}")
        print()

    """## Per-class accuracy (share of each true class that was predicted correctly)"""

    for model_name, dataframe in models_results.items():
        per_class_accuracy = dataframe.groupby("actual_sentiment")["correct"].agg(["mean", "size"])
        per_class_accuracy.columns = ["accuracy", "count"]

        print(f"{model_name}")
        for label, row in per_class_accuracy.iterrows():
            print(f"  {label:14s} {row['accuracy']:8.2%}  (n={int(row['count'])})")
        print()

    """## Classification report (precision / recall / F1 per class)"""

    for model_name, dataframe in models_results.items():
        print(f"\n{model_name}")
        print(classification_report(
            dataframe["actual_sentiment_norm"],
            dataframe["predicted_sentiment_norm"],
            zero_division=0,
        ))

"""# Performance"""

def report_performance(models_results: dict, output_path: str = "performance.csv") -> pandas.DataFrame:
    rows = []
    for model_name, dataframe in models_results.items():
        perf = dataframe.attrs.get("performance")
        if perf:
            rows.append({"model_pair": model_name, **perf})
    if not rows:
        print("No performance data recorded.")
        return pandas.DataFrame()

    performance = pandas.DataFrame(rows).set_index("model_pair")
    columns = ["device", "clips", "load_s", "asr_s", "asr_clips_per_s",
               "sentiment_s", "sentiment_per_s", "total_s", "peak_gpu_mem_mb"]
    performance = performance[columns]

    print("=== Performance (seconds unless noted) ===")
    with pandas.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", None,
                               "display.float_format", "{:.2f}".format):
        print(performance)
    print()
    performance.to_csv(output_path)
    print(f"Performance table saved to {output_path}")
    return performance

"""# Figure"""

def _short_model_name(model_names: str) -> str:
    """'ASR: org/whisper-x. Sentiment: org/roberta-y' -> 'whisper-x + roberta-y'"""
    asr, sentiment = model_names.split(". Sentiment: ")
    asr = asr.replace("ASR: ", "").split("/")[-1]
    sentiment = sentiment.split("/")[-1]
    return f"{asr}\n+ {sentiment}"

def plot_results(models_results: dict, output_path: str = "results.png"):
    """Two panels: overall accuracy vs macro F1 per model pair, and per-class accuracy heatmap."""
    # Palette (validated categorical + sequential blue ramp) and text tokens
    blue, orange = "#2a78d6", "#eb6834"
    text_primary, text_secondary, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])

    names = list(models_results.keys())
    labels = [_short_model_name(n) for n in names]

    accuracy, macro_f1, per_class = [], [], {}
    for name in names:
        dataframe = models_results[name]
        actual = dataframe["actual_sentiment"].str.lower()
        predicted = dataframe["predicted_sentiment"].str.lower()
        correct = actual == predicted
        accuracy.append(correct.mean())
        macro_f1.append(f1_score(actual, predicted, average="macro", zero_division=0))
        per_class[name] = correct.groupby(dataframe["actual_sentiment"]).mean()

    # Majority-class baseline: always predicting the most common true label
    first = models_results[names[0]]["actual_sentiment"]
    baseline = first.value_counts(normalize=True).iloc[0]

    per_class_table = pandas.DataFrame(per_class).T.loc[names]        # rows = models, cols = classes
    per_class_table = per_class_table[first.value_counts().index]     # order classes by frequency

    fig, (ax_bars, ax_heat) = plt.subplots(
        2, 1, figsize=(10, 4.2 + 0.9 * len(names)),
        gridspec_kw={"height_ratios": [1.1, 1]}, constrained_layout=True
    )
    fig.patch.set_facecolor("#fcfcfb")

    # --- Panel 1: overall accuracy vs macro F1, horizontal grouped bars
    y = list(range(len(names)))
    height = 0.36
    ax_bars.barh([i + height / 2 for i in y], accuracy, height, color=blue, label="Accuracy")
    ax_bars.barh([i - height / 2 for i in y], macro_f1, height, color=orange, label="Macro F1")
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
    ax_bars.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax_bars.set_title("Overall accuracy vs macro F1 (higher is better)",
                      loc="left", fontsize=11, color=text_primary, fontweight="bold")

    # --- Panel 2: per-class accuracy heatmap (rows = model pairs, cols = true class)
    values = per_class_table.to_numpy(dtype=float)
    ax_heat.imshow(values, cmap=blues, vmin=0, vmax=1, aspect="auto")
    class_counts = first.value_counts()
    ax_heat.set_xticks(range(values.shape[1]),
                       [f"{c}\n(n={class_counts[c]})" for c in per_class_table.columns],
                       fontsize=9, color=text_primary)
    ax_heat.set_yticks(range(values.shape[0]), labels, fontsize=9, color=text_primary)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            ax_heat.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=9,
                         color="#ffffff" if v > 0.55 else text_primary)
    # 2px surface gap between cells
    ax_heat.set_xticks([x - 0.5 for x in range(1, values.shape[1])], minor=True)
    ax_heat.set_yticks([y_ - 0.5 for y_ in range(1, values.shape[0])], minor=True)
    ax_heat.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax_heat.tick_params(which="both", length=0)
    for side in ax_heat.spines.values():
        side.set_visible(False)
    ax_heat.set_title("Per-class accuracy: share of each true class predicted correctly",
                      loc="left", fontsize=11, color=text_primary, fontweight="bold")

    fig.suptitle("Speech-to-sentiment pipeline: ASR model + sentiment model",
                 fontsize=13, color=text_primary, fontweight="bold", x=0.01, ha="left")
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Figure saved to {output_path}")
    plt.show()
    return fig

def plot_performance(performance, output_path: str = "performance.png"):
    """Four panels from the performance table (DataFrame or path to performance.csv):
    time breakdown per pair, ASR throughput, sentiment throughput, peak GPU memory."""
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

    # --- (a) time breakdown: stacked load / ASR / sentiment, in minutes
    load_m, asr_m, sent_m = (performance[c] / 60 for c in ("load_s", "asr_s", "sentiment_s"))
    total_m = load_m + asr_m + sent_m
    ax_time.barh(y, asr_m, 0.6, color=blue, label="ASR (transcription)", edgecolor=surface, linewidth=2)
    ax_time.barh(y, sent_m, 0.6, left=asr_m, color=orange, label="Sentiment", edgecolor=surface, linewidth=2)
    ax_time.barh(y, load_m, 0.6, left=asr_m + sent_m, color=aqua, label="Model load", edgecolor=surface, linewidth=2)
    for i in y:
        share = asr_m.iloc[i] / total_m.iloc[i]
        ax_time.text(total_m.iloc[i] + 0.1, i, f"{total_m.iloc[i]:.1f} min  (ASR {share:.0%})",
                     va="center", fontsize=8.5, color=text_primary)
    ax_time.set_xlim(0, total_m.max() * 1.45)
    style(ax_time, f"minutes for {clips:,} clips")
    ax_time.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    ax_time.set_title("Runtime per model pair, by stage", loc="left", fontsize=11,
                      color=text_primary, fontweight="bold", pad=18)

    # --- (b) ASR throughput
    ax_asr.barh(y, performance["asr_clips_per_s"], 0.6, color=blue)
    for i, v in enumerate(performance["asr_clips_per_s"]):
        ax_asr.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8.5, color=text_primary)
    ax_asr.set_xlim(0, performance["asr_clips_per_s"].max() * 1.2)
    style(ax_asr, "audio clips transcribed per second")
    ax_asr.set_title("ASR throughput (higher is faster)", loc="left", fontsize=11,
                     color=text_primary, fontweight="bold")

    # --- (c) sentiment throughput
    ax_sent.barh(y, performance["sentiment_per_s"], 0.6, color=orange)
    for i, v in enumerate(performance["sentiment_per_s"]):
        ax_sent.text(v + 8, i, f"{v:.0f}", va="center", fontsize=8.5, color=text_primary)
    ax_sent.set_xlim(0, performance["sentiment_per_s"].max() * 1.2)
    style(ax_sent, "transcripts classified per second")
    ax_sent.set_title("Sentiment throughput (higher is faster)", loc="left", fontsize=11,
                      color=text_primary, fontweight="bold")

    # --- (d) peak GPU memory
    mem_gb = performance["peak_gpu_mem_mb"] / 1024
    if mem_gb.notna().any():
        ax_mem.barh(y, mem_gb, 0.6, color=aqua)
        for i, v in enumerate(mem_gb):
            ax_mem.text(v + 0.04, i, f"{v:.2f} GB", va="center", fontsize=8.5, color=text_primary)
        ax_mem.set_xlim(0, mem_gb.max() * 1.25)
        style(ax_mem, "peak GPU memory allocated (GB)")
        ax_mem.set_title("Peak GPU memory per model pair", loc="left", fontsize=11,
                         color=text_primary, fontweight="bold")
    else:
        ax_mem.axis("off")
        ax_mem.text(0.0, 0.5, "No GPU memory data (run on CPU)", fontsize=10, color=text_secondary)

    fig.suptitle(f"Pipeline performance on {device}", fontsize=13, color=text_primary,
                 fontweight="bold", x=0.01, ha="left")
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Figure saved to {output_path}")
    plt.show()
    return fig

if __name__ == "__main__":
    print_environment()
    models_results = run_analyzers()
    report_results(models_results)
    performance = report_performance(models_results)
    plot_results(models_results)
    plot_performance(performance)
