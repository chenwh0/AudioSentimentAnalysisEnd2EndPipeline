"""Aggregate figure for the per-token study: distance from each token's top-attributed 100 ms segment to the token's own
timestamp (left) and the share of attribution per mel band (right), per model, M2 occlusion (primary) and M1 (comparison).
Data: Milestone2/AudioXAI/results/token_segments/tokens.csv and summary.json (CPU only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
T = R / "Milestone2/AudioXAI/results/token_segments"
GROUP = sys.argv[1] if len(sys.argv) > 1 else "all50"
df = pd.read_csv(T / "tokens.csv"); S = json.load(open(T / "summary.json"))
if GROUP != "all50":
    df = df[df.group == GROUP]
S = {m: S[m][GROUP] for m in S if isinstance(S[m], dict) and GROUP in S[m]}
GROUP_TXT = {"hard": "Hard set: the 10 clips with the lowest mean whisper-tiny token log-probability among the 50-clip shared test sample (uncertainty sampling).", "mixed": "Mixed set: 10 clips drawn from the other 40 by strata of confidence x clip length x clean/degraded (AADD class), default_rng(8001), one per stratum in fixed order until 10.", "all50": "All 50 clips of the shared stratified test sample (background numbers)."}[GROUP]
BG, GRID, BORDER, TXT, TXT2, MUTED = "#fcfcfb", "#ecebe7", "#e4e3df", "#0b0b0b", "#52514e", "#77766f"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODELS = [("whisper-tiny", "whisper-tiny"), ("whisper-base", "whisper-base"), ("aventiq", "AventIQ (whisper-base fine-tune)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BORDER, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT, "figure.facecolor": BG, "axes.facecolor": BG})
fig, axes = plt.subplots(3, 2, figsize=(11, 8.2), gridspec_kw=dict(width_ratios=[1.25, 1], hspace=0.55, wspace=0.28))
bins = np.arange(0, 2.05, 0.1)
for i, (m, name) in enumerate(MODELS):
    sub = df[df.model == m]; ax = axes[i, 0]
    for meth, col, lab in (("m2", BLUE, "M2 occlusion"), ("m1", ORANGE, "M1 gradient × input")):
        d = sub[f"{meth}_dist"].dropna()
        if len(d) == 0: continue
        h, _ = np.histogram(np.clip(d, 0, 2), bins=bins); h = h / h.sum()
        st = S[m][f"{meth}_dist_median_s"]
        ax.step(bins[:-1], h, where="post", color=col, lw=1.8,
                label=f"{lab}: {len(d)} tokens, {st['n_clips']} clips; median {st['mean_of_clip_medians']:.2f} s [{st['ci'][0]:.2f}, {st['ci'][1]:.2f}]")
    ax.axvline(0.3, color=MUTED, lw=0.8, ls="--"); ax.text(0.31, 0.01, "0.3 s", color=MUTED, va="bottom", fontsize=8)
    ax.set_title(f"{name}: |top segment − token time|", loc="left", fontsize=10)
    ax.set_xlabel("seconds (last bin collects every distance of 1.9 s or more)"); ax.set_ylabel("share of tokens"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right"); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
    ax = axes[i, 1]; x = np.arange(8); w = 0.38
    for j, (meth, col, lab) in enumerate((("m2", BLUE, "M2"), ("m1", ORANGE, "M1"))):
        if f"{meth}_band_mass_mean" not in S[m]: continue
        ax.bar(x + (j - 0.5) * w, S[m][f"{meth}_band_mass_mean"], w, color=col, label=lab, edgecolor=BG, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{b*10}–{b*10+9}" for b in range(8)], fontsize=7.5)
    ax.set_xlabel("mel bins (0 = lowest frequency)"); ax.set_ylabel("mean share of attribution"); ax.set_ylim(0, 0.42)
    ax.set_title(f"{name}: attribution by mel band", loc="left", fontsize=10); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
fig.suptitle(f"[{GROUP} set] Where each output token's attribution sits: distance to its own timestamp, and mel-band share, per model", x=0.01, ha="left", fontsize=11, color=TXT)
fig.text(0.01, 0.005, GROUP_TXT + " Numbers from 10 clips are indicative, not conclusive. M2 (occlusion of 100 ms × 10-mel patches) is the primary method; M1 (gradient × input, SmoothGrad) is shown for comparison because it failed the "
         "model-randomization check (ρ ≈ 0.92) and tracks input energy (|M1| vs |input| 0.78). M1 rows use the 50-clip test sample for every model; "
         "CIs: 10,000 bootstrap resamples of clips. Clip ids in Milestone2/AudioXAI/results/token_segments/groups.csv.",
         fontsize=7.5, color=TXT2, wrap=True)
out = R / f"figures/xai_tokens/aggregate_distance_bands_{GROUP}"
fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight"); fig.savefig(str(out) + ".svg", bbox_inches="tight")
print("wrote", out)
