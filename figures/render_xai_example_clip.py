import json
import numpy as np, pandas as pd, soundfile as sf
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from _t1_style import *

CLIP = "id10273_ZpPPaMazpf0_00003"
spec, data = load_spec("xai_example_clip")
apply_rc()
CLASSES = ['noise', 'music', 'reverb', 'clipping', 'lowpass', 'highpass', 'codec', 'packet_loss',
           'hum', 'clicks', 'pitch_shift', 'time_stretch']   # audioxai.core.CLASSES (AADD v3 column order)
z = np.load(data, allow_pickle=False)
meta = json.loads(str(z["meta"]))
dur = float(meta["dur"]); nf = int(meta["n_frames"]); nseg = int(meta["n_seg"])
wav, sr = sf.read(R / "Milestone2/PipelineEvaluation/data/audio" / f"{CLIP}.wav", dtype="float32")
assert sr == 16000 and wav.ndim == 1
seg = pd.read_csv(R / "Milestone2/T1_final/results/aadd_devtest_seg20ms.csv")
seg = seg[seg.path.str.endswith(f"/{CLIP}.wav")].sort_values("start_s")
P = seg[[f"p_{c}" for c in CLASSES]].to_numpy(np.float32)[:nseg]
assert np.allclose(P, z["aadd"].astype(np.float32), atol=2e-3), "AADD csv rows differ from the npz"
agg = pd.read_csv(R / "Milestone2/AudioXAI/results/aggregate.csv").set_index("file").loc[CLIP]

# M1: |gradient x input| per token, each token map normalised to unit mass, averaged over tokens -> clip map
m1 = np.abs(z["m1"].astype(np.float32))
m1 = (m1 / m1.sum((1, 2), keepdims=True)).mean(0)                  # (80, nf), sums to 1
k = np.ones(3) / 3                                                   # light 3x3 box smoothing for legibility
m1s = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, m1)
m1s = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, m1s)
alpha = np.clip(m1s / np.percentile(m1s, 99.5), 0, 1)
mel = z["logmel"].astype(np.float32)

words, tags, events = meta["words"], meta["tags"], meta["events"]
ig = z["ig_abs"].astype(float); share = ig / ig.sum(); sgn = z["ig_sig"].astype(float)
assert len(words) == len(tags) == len(share)
LAB = ["Negative", "Neutral", "Positive"]
pred = LAB[int(np.argmax(z["probs"]))]
ERR = {"ok": ("#52514e", "correct"), "sub": ("#D55E00", "substitution"), "ins": ("#CC79A7", "insertion")}
HEAT = "#E69F00"

fig, axs = plt.subplots(4, 1, figsize=(11, 10.2), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 1.7, 0.9, 1.5]))
fig.subplots_adjust(top=0.875, bottom=0.165, left=0.09, right=0.98, hspace=0.12)
a0, a1, a2, a3 = axs
t = np.arange(len(wav)) / sr
a0.plot(t, wav, color=INK2, lw=0.4)
a0.set_ylabel(spec["axes"]["panel_y"][0]); a0.grid(axis="y", visible=False)
lim = np.abs(wav).max() * 1.05; a0.set_ylim(-lim, lim)

ext = [0, nf * 0.01, 0, 80]
a1.imshow(mel, origin="lower", aspect="auto", extent=ext, cmap="Greys_r", interpolation="nearest")
rgba = np.zeros(mel.shape + (4,)); rgba[..., :3] = to_rgb(HEAT); rgba[..., 3] = 0.9 * alpha
a1.imshow(rgba, origin="lower", aspect="auto", extent=ext, interpolation="nearest")
a1.set_ylabel("mel bin\n(80, low to high frequency)"); a1.grid(False)
a1.set_yticks([0, 20, 40, 60, 80])
a1.text(0.005, 0.96, "orange = where whisper-tiny's attribution falls (|gradient x input|, SmoothGrad, mean over "
        f"{len(z['keep'])} tokens); more opaque = more attribution", transform=a1.transAxes, fontsize=7.5,
        color=INK, va="top", bbox=dict(boxstyle="square,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))

mx = P.max(1); t20 = np.arange(len(mx)) * 0.02
flag = mx >= 0.5
dom = pd.Series(np.array(CLASSES)[P.argmax(1)][flag]).value_counts()
a2.fill_between(t20, 0, flag * 1.0, step="post", color=XAI_CLASS_COLOR["time_stretch"], alpha=0.12, lw=0)
a2.fill_between(t20, 0, mx, step="post", color="#0072B2", alpha=0.35, lw=0)
a2.step(t20, mx, where="post", color="#0072B2", lw=1.2)
a2.axhline(0.5, color=INK, ls="--", lw=0.9)
a2.text(dur - 0.02, 0.52, "flag threshold 0.5", ha="right", va="bottom", fontsize=7.5, color=INK)
a2.set_ylim(0, 1.05); a2.set_ylabel("degradation\nprobability")
a2.text(0.005, 0.97, f"flagged {flag.mean():.0%} of 20 ms frames (shaded); most likely class there: "
        + ", ".join(f"{c.replace('_', ' ')} {n}" for c, n in dom.items()) + " frames",
        transform=a2.transAxes, fontsize=7.5, color=INK, va="top",
        bbox=dict(boxstyle="square,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))

# words: spread label positions so they do not collide, leader line back to the true time
cx = np.array([0.5 * (w["t0"] + w["t1"]) for w in words])
gap = 0.105
x = cx.copy()
for _ in range(500):
    moved = False
    for i in range(1, len(x)):
        d = x[i] - x[i - 1]
        if d < gap:
            sh = (gap - d) / 2; x[i - 1] -= sh; x[i] += sh; moved = True
    x = np.clip(x, 0.05, dur - 0.03)
    if not moved:
        break
bw = 0.075
top = share.max() * 1.05
for i, w in enumerate(words):
    col = ERR[tags[i]][0]
    a3.bar(x[i], share[i], width=bw, color=col, zorder=3)
    if sgn[i] < 0:
        a3.bar(x[i], share[i], width=bw, facecolor="none", edgecolor=SURFACE, hatch="////", lw=0, zorder=4)
    if abs(x[i] - cx[i]) > 0.01:
        a3.plot([cx[i], x[i]], [-0.06 * top, 0], color=MUTED, lw=0.6, zorder=2)
    a3.plot([cx[i]], [-0.06 * top], marker="|", color=MUTED, ms=5)
    a3.text(x[i], share[i] + 0.04 * top, w["word"], rotation=90, ha="center", va="bottom", fontsize=8.5, color=col,
            fontweight="bold" if tags[i] != "ok" else "normal")
dels = [e[0] for e in events if e[2] == "del"]
a3.scatter(dels, [-0.06 * top] * len(dels), marker="v", s=46, color=INK, zorder=5)
a3.set_ylim(-0.1 * top, top * 1.75)
a3.set_yticks([0, 0.05, 0.10])
a3.axhline(0, color=AXIS, lw=0.8)
a3.set_ylabel("sentiment importance\n(Integrated Gradients,\nshare)")
a3.set_xlabel(spec["axes"]["x"])
a3.set_xlim(0, dur + 0.1)
for a in axs:
    a.grid(axis="x", visible=False)

cnt = meta["cnt"]
wer = (cnt["S"] + cnt["D"] + cnt["I"]) / cnt["N"]
head = (f"Clip {CLIP}  |  gold sentiment label: {meta['meta']['label']}  |  predicted from the transcript: {pred} "
        f"(p = {z['probs'].max():.2f})\nword error rate {wer:.2f} ({cnt['S']} substituted, {cnt['D']} dropped, "
        f"{cnt['I']} inserted of {cnt['N']} words)")
fig.text(0.01, 0.885, head, fontsize=8.5, color=INK, ha="left", va="bottom", fontweight="bold")
fig.text(0.09, 0.125, f"whisper-tiny transcript: \"{meta['text']}\"\nhuman reference:         \"{meta['gold']}\"",
         fontsize=8, color=INK2, ha="left", va="top", family="DejaVu Sans Mono")
handles = [Patch(color=ERR[k][0], label=f"word {ERR[k][1]}") for k in ("ok", "sub", "ins")]
handles += [Line2D([], [], ls="", marker="v", color=INK, ms=7, label="dropped word (missing from transcript)"),
            Patch(facecolor=MUTED, hatch="////", edgecolor=SURFACE, label="word pushes against the predicted label")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.085, 0.068), ncol=5, fontsize=7.8, handlelength=1.4)
note = ("Waveform and spectrogram from the 16 kHz clip; degradation probability = highest of AADD's 12 class probabilities per 20 ms frame. "
        "Importance share = |Integrated Gradients| of the Cardiff sentiment model for each word / sum over words. "
        "Ticks under the baseline mark each word's true time; words are spread apart only to keep labels readable.")
finish(fig, spec, 150, note=note)
