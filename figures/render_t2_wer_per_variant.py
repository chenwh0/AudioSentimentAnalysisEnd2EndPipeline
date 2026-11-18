import numpy as np, pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from _t1_style import *

spec, data = load_spec("t2_wer_per_variant")
apply_rc()
plt.rcParams["hatch.linewidth"] = 0.8
df = pd.read_csv(data)
REF = "openai/whisper-large-v3"
ref_wer = float(df.loc[df.asr == REF, "wer_gold"].iloc[0])
df = df[df.asr != REF].copy()
SHORT = {"UsefulSensors/moonshine-base": "moonshine-base", "openai/whisper-base": "whisper-base",
         "AventIQ-AI/whisper-audio-to-text": "AventIQ (whisper-base fine-tune)", "openai/whisper-tiny": "whisper-tiny",
         "facebook/s2t-small-librispeech-asr": "s2t-small (batch-1 fix)",
         "s2t-small-OLD(batch8)": "s2t-small, before the batch fix"}
OLD = "s2t-small-OLD(batch8)"
df["name"] = df.asr.map(SHORT)
df = df.sort_values("wer_gold").reset_index(drop=True)   # ascending; plotted top to bottom

SER = ["wer_gold", "wer_gold_nofillers", "wer_pseudo_ref"]
SER_LABEL = {"wer_gold": "vs human (SLUE) transcripts",
             "wer_gold_nofillers": "vs human transcripts, filler words removed",
             "wer_pseudo_ref": "vs whisper-large-v3 machine transcript (the old stand-in)"}
HUE = "#0072B2"


def tint(c, f):
    return tuple(1 - (1 - v) * f for v in to_rgb(c))


def style(s, old=False):
    hue = MUTED if old else HUE
    if s == "wer_gold":
        st = dict(color=hue)
    elif s == "wer_gold_nofillers":
        st = dict(color=tint(hue, 0.45))
    else:
        st = dict(facecolor=SURFACE, hatch="////", edgecolor=hue, linewidth=0.8)
    return st


fig, ax = plt.subplots(figsize=(10, 6.4))
fig.subplots_adjust(top=0.84, bottom=0.2, left=0.25, right=0.97)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
h, gap = 0.24, 0.03
y = np.arange(len(df))
for i, s in enumerate(SER):
    ys = y + (i - 1) * (h + gap)
    for yy, (_, r) in zip(ys, df.iterrows()):
        old = r.asr == OLD
        ax.barh(yy, r[s], height=h, zorder=2, **style(s, old))
        ax.text(r[s] + 0.008, yy, f"{r[s]:.3f}", va="center", ha="left", fontsize=7.5,
                color=MUTED if old else INK2, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
ax.axvline(ref_wer, color=INK, ls="--", lw=1, zorder=3)
ax.text(ref_wer + 0.006, -0.72, f"whisper-large-v3 (the pseudo-reference) vs gold: {ref_wer:.3f}",
        fontsize=8, color=INK, va="bottom", ha="left")
ax.set_yticks(y, df.name)
for t, a in zip(ax.get_yticklabels(), df.asr):
    if a == OLD:
        t.set_color(MUTED); t.set_fontstyle("italic")
ax.set_ylim(len(df) - 0.5, -0.85)
ax.set_xlim(0, 1.0)
ax.set_xlabel(spec["axes"]["x"])
ax.set_ylabel(spec["axes"]["y"])
handles = [Patch(label=SER_LABEL[s], **style(s)) for s in SER]
handles.append(Patch(label="gray bars: superseded run (batched s2t-small, before the fix)", color=MUTED))
handles.append(Line2D([], [], color=INK, ls="--", lw=1, label="whisper-large-v3 vs human transcripts"))
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.25, 0.045), ncol=2, fontsize=8)
finish(fig, spec, 120, note=WER_NOTE)
