import json
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from _t1_style import *

spec, data = load_spec("xai_deletion_curves")
apply_rc()
df = pd.read_csv(data)
DL = json.loads((data.parent / "summary.json").read_text())["deletion"]
assert len(df) == DL["n"] == 20
ks = np.arange(5, 51, 5)
SER = {"top": ("drop_top_", "#D55E00", "o", "-", "most-attributed patches removed first"),
       "rand": ("drop_rand_", "#0072B2", "s", "--", "random patches removed")}

fig, ax = plt.subplots(figsize=(8.6, 5.9))
fig.subplots_adjust(top=0.85, bottom=0.235, left=0.12, right=0.97)
ax.grid(axis="x", visible=True)
handles, labels = [], []
for key, (pre, col, mk, ls, lab) in SER.items():
    Y = df[[f"{pre}{k}" for k in ks]].to_numpy(float)
    m = Y.mean(0)
    lo, hi = boot_ci(Y)
    ax.fill_between(ks, lo, hi, color=col, alpha=0.14, lw=0, zorder=1)
    ax.plot(ks, m, color=col, ls=ls, lw=2, marker=mk, ms=6, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    handles.append((Patch(color=col, alpha=0.25, lw=0), Line2D([], [], color=col, ls=ls, lw=2, marker=mk, ms=6)))
    labels.append(lab + " (band: 95% CI)")
ax.set_xticks(ks, [f"{k}%" for k in ks])
ax.set_xlim(3, 52)
ax.set_ylim(0, None)
ax.set_xlabel(spec["axes"]["x"]); ax.set_ylabel("drop in the model's log-probability of its own\ntranscript (nats per token, mean over clips)")
f = lambda o: f"{o['mean']:.3f} (95% CI {o['ci'][0]:.3f} to {o['ci'][1]:.3f})"
box = (f"Area under the curve, mean over {DL['n']} clips\n"
       f"top:        {DL['auc_top']['mean']:.3f}\n"
       f"random:     {DL['auc_random']['mean']:.3f}\n"
       f"difference: {f(DL['auc_diff'])}\n"
       f"top above random in {DL['frac_clips_top_gt_random']:.0%} of clips")
ax.text(0.98, 0.04, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2, multialignment="left", family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
from matplotlib.legend_handler import HandlerTuple
fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="lower left",
           bbox_to_anchor=(0.11, 0.105), ncol=2, fontsize=8, handlelength=3)
note = ("Patches are tiles of the log-mel spectrogram ranked by the attribution map (gradient x input with SmoothGrad); the drop is "
        "the mean per-token fall in log-probability (nats) of the model's own transcript after the patches are removed. "
        "Area under the curve is taken over the removed fraction 0.05 to 0.50. CIs: 10,000 resamples of the 20 clips, numpy default_rng(8001).")
finish(fig, spec, 105, note=note)
