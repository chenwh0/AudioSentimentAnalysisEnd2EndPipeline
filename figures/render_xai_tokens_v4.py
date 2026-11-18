"""Presentation figures per (model, clip), each 10 x 8 in at 300 dpi, fonts >= 12 pt, one shared time axis per figure.
  A  <model>__<clip>__quality            waveform -> mel (magma) -> AADD class rows + max p -> degraded-interval chips. No tokens.
  B  <model>__<clip>__tokens_on_audio    thin AADD max-p strip -> mel with each token's top M2 patch boxed in the token's colour
                                         -> token lane with the token text on the segment that mattered most to it.
  C  <model>__<clip>__token_time_heatmap token x time heat map (rows = tokens, columns = 100 ms, row scaled to its max).
Token colours: 5 slots with no blue or magma hue (validated, CVD worst adjacent dE 7.9); the index is printed with every token.
Usage: python figures/render_xai_tokens_v4.py --model <whisper-tiny|whisper-base|aventiq> --clip <id> [--cls text]"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_xai_tokens as v1
from render_xai_tokens_v3 import CLASSES, BLUE, CAT, intervals, SEG, PT

OUT = v1.OUT; AUDIO = v1.ROOT / "Milestone2/PipelineEvaluation/data/audio"
BG, GRID, BORDER, TXT, TXT2, MUTED = "#fcfcfb", "#ecebe7", "#e4e3df", "#0b0b0b", "#52514e", "#77766f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "figure.facecolor": BG,
                     "axes.facecolor": BG, "axes.edgecolor": BORDER, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.labelcolor": TXT2, "svg.fonttype": "none"})
DPI = 300

def col(k): return CAT[k % len(CAT)]

def title_of(D, model, clip, cls):
    cnt = D["meta"]["cnt"]; wer = (cnt["S"] + cnt["D"] + cnt["I"]) / max(1, cnt["N"])
    name = {"whisper-tiny": "whisper-tiny", "whisper-base": "whisper-base", "aventiq": "AventIQ (whisper-base fine-tune)"}.get(model, model)
    return f"{clip}\n{name}  ·  degradation: {cls or 'unknown'}  ·  WER {wer:.0%}"

def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=12, color=TXT2, pad=4)

def tidy(ax, dur, bottom=False):
    ax.set_xlim(0, dur)
    for s in ax.spines.values(): s.set_color(BORDER)
    ax.tick_params(labelbottom=bottom, colors=TXT2)
    if bottom: ax.set_xlabel("time (s)")

def load_wave(clip):
    wav, sr = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    step = max(1, sr // 4000); w = wav[::step]; return w, np.arange(len(w)) * step / sr

def draw_wave(ax, w, tw, pmax):
    pts = np.column_stack([tw, w]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
    fi = np.minimum((tw[:-1] / SEG).astype(int), len(pmax) - 1)
    ax.add_collection(LineCollection(segs, colors=BLUE(Normalize(0, 1)(pmax[fi])), linewidths=0.8))
    amp = float(np.abs(w).max() + 1e-6); ax.set_ylim(-amp * 1.1, amp * 1.1); ax.set_yticks([])

def draw_mel(ax, L, dur):
    nf = L.shape[1]
    ax.imshow(L, origin="lower", aspect="auto", cmap="magma", extent=[0, nf * 0.01, 0, 80], interpolation="nearest",
              vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80]); ax.set_yticklabels(["0", "1.9 k", "8 k"]); ax.set_ylabel("Hz (mel)")

def class_rows(ax, P, active, pmax, dur):
    nrows = len(active) + 1; ax.set_ylim(0, nrows); ax.set_yticks([])
    tseg = np.arange(len(P)) * SEG
    for ri, j in enumerate(list(active) + [None]):
        y = nrows - 1 - ri; p = pmax if j is None else P[:, j]
        ax.bar(tseg, np.ones(len(P)) * 0.84, width=SEG, bottom=y + 0.08, color=BLUE(np.clip(p, 0, 1)), align="edge", lw=0)
        ax.text(-0.01 * dur, y + 0.5, "max" if j is None else CLASSES[j].replace("_", " "), ha="right", va="center",
                fontsize=12, color=TXT, fontweight="bold" if j is None else "normal")
    if not active: ax.text(dur / 2, nrows - 0.5, "no class reaches 0.5 in this clip", ha="center", va="center", color=MUTED)

def colorbar(fig, x, y, w=0.16, h=0.014, label="AADD degradation probability"):
    cax = fig.add_axes([x, y, w, h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=BLUE), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cax.tick_params(labelsize=10, colors=TXT2, length=2); cax.set_title(label, fontsize=10, color=TXT2, pad=2)

def fig_quality(D, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32); dur = D["meta"]["dur"]
    P = D["aadd"].astype(np.float32)
    if len(P) == 0: P = np.zeros((int(np.ceil(dur / SEG)), len(CLASSES)), np.float32)
    pmax = P.max(1); active = [j for j in range(len(CLASSES)) if P[:, j].max() >= 0.5]
    w, tw = load_wave(clip)
    ivs = intervals(P); nchip_lines = max(1, int(np.ceil(len(ivs) / 2))) if ivs else 1
    hs = [1.2, 2.6, 0.42 * (len(active) + 1) + 0.2]
    extra = 0.3 * max(0, nchip_lines - 2)
    fig = plt.figure(figsize=(10, 8.6 + extra))
    bottom = 0.27 * 8.6 / (8.6 + extra)
    gs = fig.add_gridspec(3, 1, height_ratios=hs, hspace=0.42, left=0.16, right=0.97, top=0.84, bottom=bottom)
    axw, axm, axc = [fig.add_subplot(gs[i]) for i in range(3)]
    draw_wave(axw, w, tw, pmax); tidy(axw, dur); panel_label(axw, "Waveform, coloured by how degraded AADD rates each 20 ms frame")
    draw_mel(axm, L, dur); tidy(axm, dur); panel_label(axm, "Mel spectrogram, the input the speech model sees")
    class_rows(axc, P, active, pmax, dur); tidy(axc, dur, bottom=True)
    panel_label(axc, "AADD class probability per 20 ms frame (classes that reach 0.5, and the maximum)")
    colorbar(fig, 0.79, 0.905)
    chips = [f"{n.replace('_', ' ')}  {a:.1f}–{b:.1f} s  peak {m:.2f}" for n, a, b, m in ivs] or ["AADD finds no degraded interval in this clip."]
    ytop = bottom - 0.085
    fig.text(0.16, ytop, "Degraded intervals AADD found (class, time span, peak probability)", fontsize=12, color=TXT2, va="bottom")
    for i, c in enumerate(chips[:12]):
        fig.text(0.16 + (i % 2) * 0.41, ytop - 0.03 - (i // 2) * 0.042 * 8.6 / (8.6 + extra), c, fontsize=11, color=TXT, va="top",
                 bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.9", fc=BG, ec=BORDER, lw=0.9))
    fig.suptitle(title_of(D, model, clip, cls), x=0.16, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.975)
    fig.text(0.16, 0.012, "Light blue = clean, dark blue = degraded (AADD 3.1.0, 20 ms frames, flag at probability ≥ 0.5). Audio quality is context for figure B.",
             fontsize=10, color=TXT2, va="bottom", wrap=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI); fig.savefig(OUT / f"{stem}.svg"); plt.close(fig)

def assign_lanes(rows, dur, axis_in=8.1, char_in=0.095):
    lanes = []
    for r in rows:
        if "m2_seg" not in r: r["lane"] = 0; continue
        x0 = r["m2_seg"] * PT; width = PT + (len(r["tok"].strip() or "␣") + 3) * char_in * dur / axis_in + 0.02 * dur
        r["x0"], r["x1"] = x0, x0 + width
        for li in range(40):
            while len(lanes) <= li: lanes.append([])
            if all(r["x1"] <= a or r["x0"] >= b for a, b in lanes[li]):
                lanes[li].append((r["x0"], r["x1"])); r["lane"] = li; break
    return max(1, max(r.get("lane", 0) for r in rows) + 1)

def fig_tokens_on_audio(D, rows, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32); dur = D["meta"]["dur"]
    P = D["aadd"].astype(np.float32); pmax = P.max(1) if len(P) else np.zeros(int(np.ceil(dur / SEG)))
    nl = assign_lanes(rows, dur)
    lane_h = 0.34 * nl
    hs = [0.25, 2.4, lane_h]
    H = 8.0 + max(0, lane_h - 2.4) * 0.9
    fig = plt.figure(figsize=(10, min(H, 10.5)))
    gs = fig.add_gridspec(3, 1, height_ratios=hs, hspace=0.3, left=0.16, right=0.97, top=0.86, bottom=0.14)
    axq, axm, axt = [fig.add_subplot(gs[i]) for i in range(3)]
    tseg = np.arange(len(pmax)) * SEG
    axq.bar(tseg, np.ones(len(pmax)), width=SEG, color=BLUE(np.clip(pmax, 0, 1)), align="edge", lw=0); axq.set_ylim(0, 1); axq.set_yticks([])
    tidy(axq, dur); panel_label(axq, "Audio quality: AADD maximum degradation probability (dark = degraded)")
    draw_mel(axm, L, dur); tidy(axm, dur)
    for r in rows:
        if "m2_patch" not in r: continue
        b, t = r["m2_patch"]
        axm.add_patch(Rectangle((t * PT, b * 10), PT, 10, fill=False, edgecolor=col(r["k"]), lw=2.0, zorder=3))
        above = b < 7
        axm.text(t * PT + PT / 2, b * 10 + (10.5 if above else -0.5), str(r["k"]), color=col(r["k"]), fontsize=10, ha="center", va="bottom" if above else "top", fontweight="bold", zorder=4, clip_on=True,
                 bbox=dict(boxstyle="square,pad=0.05", fc=BG, ec="none", alpha=0.7))
    panel_label(axm, "Mel spectrogram; each box is the patch that mattered most to one token (occlusion)")
    axt.set_ylim(0, nl); axt.set_yticks([]); axt.grid(axis="x", color=GRID, lw=0.7); tidy(axt, dur, bottom=True)
    for r in rows:
        if "m2_seg" not in r: continue
        y = nl - 1 - r["lane"]; x0 = r["m2_seg"] * PT
        axt.add_patch(Rectangle((x0, y + 0.12), PT, 0.76, facecolor=col(r["k"]), edgecolor="white", lw=0.6))
        axt.text(x0 + PT + 0.012 * dur, y + 0.5, f'{r["k"]} {r["tok"].strip() or "␣"}', color=col(r["k"]), fontsize=12, ha="left", va="center", fontweight="bold")
    panel_label(axt, "Tokens, placed on the audio segment that mattered most to each one")
    fig.suptitle(title_of(D, model, clip, cls), x=0.16, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.975)
    fig.text(0.16, 0.012, "Same colour = same token (index, then text). Colours repeat every 5 tokens. Box = the 100 ms × 10-mel patch whose removal hurts that token most.",
             fontsize=10, color=TXT2, va="bottom", wrap=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI); fig.savefig(OUT / f"{stem}.svg"); plt.close(fig)

def fig_heatmap(D, rows, model, clip, cls, stem):
    dur = D["meta"]["dur"]; nf = D["logmel"].shape[1]; npatch = int(np.ceil(nf / 10)); K = len(rows)
    H = np.zeros((K, npatch), np.float32)
    if D["m2"] is not None:
        for r in rows:
            p = np.clip(D["m2"][r["k"]], 0, None).sum(0); H[r["k"]] = p / (p.max() + 1e-12)
    fig = plt.figure(figsize=(10, max(6.5, min(10.5, 2.6 + 0.26 * K))))
    ax = fig.add_axes([0.2, 0.12, 0.76, 0.76])
    ax.imshow(H, aspect="auto", cmap="Greys", extent=[0, npatch * PT, K - 0.5, -0.5], interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(K)); ax.set_yticklabels([f'{r["k"]} {r["tok"].strip() or "␣"}' for r in rows], fontsize=max(8, min(12, 300 / max(K, 1))))
    tidy(ax, dur, bottom=True); panel_label(ax, "How much each 100 ms of audio mattered to each token (occlusion, row scaled to its own maximum)")
    for lab, r in zip(ax.get_yticklabels(), rows): lab.set_color(col(r["k"]))
    fig.suptitle(title_of(D, model, clip, cls), x=0.2, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.975)
    fig.text(0.2, 0.012, "Darker = larger drop in the token's log-probability when that 100 ms patch is removed. Row labels share the token colours of figure B.",
             fontsize=10, color=TXT2, va="bottom", wrap=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI); fig.savefig(OUT / f"{stem}.svg"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--cls", default=None)
    a = ap.parse_args()
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
    D = v1.load(a.model, a.clip); rows = v1.analyse(D, tok); cls = a.cls or v1.KNOWN_CLS.get(a.clip)
    base = f"{a.model}__{a.clip}"
    fig_quality(D, a.model, a.clip, cls, f"{base}__quality")
    fig_tokens_on_audio(D, rows, a.model, a.clip, cls, f"{base}__tokens_on_audio")
    fig_heatmap(D, rows, a.model, a.clip, cls, f"{base}__token_time_heatmap")
    v1.write_sidecar(D, rows, a.model, a.clip, cls, f"{base}__tokens_on_audio")
    print(base, "v4", len(rows), "tokens")

if __name__ == "__main__":
    main()
