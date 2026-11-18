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


def finish(fig, spec, wrap_width):
    """Caption as sub-caption title; macro-F1 footnote at the bottom."""
    fig.suptitle(wrap(spec["caption"], wrap_width), x=0.01, ha="left", fontsize=10, color=INK, y=0.995)
    fig.text(0.01, 0.005, wrap(MACRO_F1_NOTE, int(wrap_width * 1.15)), ha="left", va="bottom", fontsize=7.5, color=MUTED)
    out = R / spec["output"]
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    print("wrote", out, out.with_suffix(".svg"))
