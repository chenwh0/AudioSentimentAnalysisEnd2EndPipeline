"""Shared style for the T1 figures (palette validated with the dataviz validator, --pairs all)."""
import hashlib, json, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent

# Okabe-Ito, fixed order cardiff, distilbert, siebert, tabularisai (all-pairs CVD dE >= 8.6, normal >= 15.6)
SENT_ORDER = ["cardiff", "distilbert", "siebert", "tabularisai"]
SENT_COLOR = {"cardiff": "#0072B2", "distilbert": "#D55E00", "siebert": "#009E73", "tabularisai": "#E69F00"}
SENT_LABEL = {"cardiff": "Cardiff (Twitter RoBERTa)", "distilbert": "DistilBERT (SST-2)",
              "siebert": "SiEBERT (RoBERTa-large)", "tabularisai": "tabularisai (multilingual)"}
ASR_ORDER = ["AventIQ", "s2t-small", "whisper-tiny"]
ASR_MARKER = {"AventIQ": "o", "s2t-small": "s", "whisper-tiny": "^"}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MACRO_F1_NOTE = ("Macro-F1 = the plain average of the F1 score for each of the three sentiment classes "
                 "(negative, neutral, positive), so the rare negative class counts as much as the common neutral one.")


def load_spec(name):
    spec = json.loads((R / "figures" / "specs" / f"{name}.json").read_text())
    data = R / spec["data_file"]
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != spec["data_sha256"]:
        raise SystemExit(f"sha256 mismatch for {data}: {digest} != {spec['data_sha256']}")
    return spec, data


def apply_rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "svg.fonttype": "none",
    })


def wrap(s, width):
    return "\n".join(textwrap.wrap(s, width))


def finish(fig, spec, wrap_width, note=MACRO_F1_NOTE):
    """Caption as sub-caption title; footnote (default: macro-F1 definition) at the bottom."""
    fig.suptitle(wrap(spec["caption"], wrap_width), x=0.01, ha="left", fontsize=10, color=INK, y=0.995)
    fig.text(0.01, 0.005, wrap(note, int(wrap_width * 1.15)), ha="left", va="bottom", fontsize=7.5, color=MUTED)
    out = R / spec["output"]
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    print("wrote", out, out.with_suffix(".svg"))


# T2: one color per speech-to-text variant (Okabe-Ito colors not used for the sentiment models where possible),
# fixed order AventIQ, s2t-small, whisper-tiny, whisper-base, moonshine-base. Markers double-encode identity.
T2_ORDER = ["AventIQ", "s2t-small", "whisper-tiny", "whisper-base", "moonshine-base"]
T2_COLOR = {"AventIQ": "#CC79A7", "s2t-small": "#56B4E9", "whisper-tiny": "#009E73",
            "whisper-base": "#D55E00", "moonshine-base": "#000000"}
T2_MARKER = {"AventIQ": "o", "s2t-small": "s", "whisper-tiny": "^", "whisper-base": "D", "moonshine-base": "v"}
WER_NOTE = ("Word error rate = (words substituted + words dropped + words inserted) / words in the reference transcript; "
            "0.18 means about 18 errors per 100 reference words.")

# XAI figures: AADD dominant degradation class, fixed order by clip count in Milestone2/AudioXAI/results/aggregate.csv
# (ties broken by summary.json degradation_class_counts order). Six Okabe-Ito hues validated --pairs all (worst CVD
# dE 7.6 -> markers double-encode identity); "clean" is deliberately neutral gray with an open marker.
XAI_CLASS_ORDER = ["clean", "noise", "reverb", "music", "codec", "clipping", "time_stretch"]
XAI_CLASS_COLOR = {"clean": "#898781", "noise": "#0072B2", "reverb": "#E69F00", "music": "#009E73",
                   "codec": "#CC79A7", "clipping": "#56B4E9", "time_stretch": "#D55E00"}
XAI_CLASS_MARKER = {"clean": "o", "noise": "o", "reverb": "s", "music": "^", "codec": "D", "clipping": "v",
                    "time_stretch": "P"}
XAI_CLASS_LABEL = {"clean": "clean", "noise": "noise", "reverb": "reverb (echo)", "music": "music",
                   "codec": "codec artefacts", "clipping": "clipping", "time_stretch": "time stretch"}
# Cardiff sentiment correct / wrong (pair validated: CVD dE 21.9, normal 31.2)
CORRECT_COLOR = {1: "#0072B2", 0: "#D55E00"}


def boot_ci(x, n=10_000, seed=8001, stat=None):
    """Percentile 95% bootstrap CI of the mean (numpy default_rng(seed)); x is 1-D or (clips, k) for per-column CIs."""
    import numpy as np
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5, axis=0), np.percentile(m, 97.5, axis=0)
