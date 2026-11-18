"""Per-clip token-to-audio summary figure, v3, laid out like AADD's file inspector (docs/img/report_inspector_light.png):
one shared time axis, top to bottom: (1) waveform coloured by AADD max degradation probability per 20 ms frame on AADD's
blue sequential ramp, with a small colorbar; (2) log-mel spectrogram in magma; (3) AADD class timeline rows (only classes
that reach p >= 0.5 somewhere in the clip) plus a "max p" row, blue ramp by probability; (4) a token lane: each output token
as a coloured block over its top M2 occlusion segment with the token text, the same colour as a thin outline on the mel and a
tick on the waveform; (5) token x time heat strip (rows = tokens, columns = 100 ms patches, cell = M2 scaled to row max);
(6) chips with the detected degraded intervals (class, start-end, max p). Token palette: 5 slots with no blue or magma hue
(validated: CVD worst adjacent dE 7.9, so the token index is printed in every block). Colours repeat every 5 tokens.
Usage: python figures/render_xai_tokens_v3.py --model <whisper-tiny|whisper-base|aventiq> --clip <id> [--cls text]"""
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

OUT = v1.OUT; AUDIO = v1.ROOT / "Milestone2/PipelineEvaluation/data/audio"
CLASSES = ['noise', 'music', 'reverb', 'clipping', 'lowpass', 'highpass', 'codec', 'packet_loss', 'hum', 'clicks', 'pitch_shift', 'time_stretch']
BG, GRID, BORDER, TXT, TXT2, MUTED = "#fcfcfb", "#ecebe7", "#e4e3df", "#0b0b0b", "#52514e", "#77766f"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("aadd_seq", SEQ)
CAT = ["#008300", "#c48a00", "#1baf7a", "#8f5a1a", "#5a9e00"]
SEG, PT = 0.02, 0.1
CAPTION = ("Layout follows AADD's file inspector. Blues are audio quality only: the waveform and the class rows use AADD's sequential "
           "ramp (light = clean, dark = degraded; a frame counts as degraded at max class probability ≥ 0.5). Token colours (greens, "
           "golds, brown, no blue or magma) link each output token to the audio it used: a block over its top M2 occlusion segment in "
           "the token lane, an outline of its top M2 patch on the mel, and a tick on the waveform. Colours repeat every 5 tokens, so "
           "the token index is printed in each block. M2 (occlusion of 100 ms × 10-mel patches) is the primary method; M1 (gradient × "
           "input) is not drawn here because it failed the model-randomization check (ρ ≈ 0.92) and tracks input energy (0.78).")

def col(k): return CAT[k % len(CAT)]

def intervals(P, thr=0.5, gap=0.5):
    out = []
    for j, name in enumerate(CLASSES):
        runs, a, m = [], -1, 0.0
        p = P[:, j]
        for i in range(len(p) + 1):
            v = p[i] if i < len(p) else 0.0
            if v >= thr:
                if a < 0: a = i
                m = max(m, v)
            elif a >= 0:
                runs.append([a * SEG, i * SEG, m]); a, m = -1, 0.0
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] < gap:
                merged[-1][1] = r[1]; merged[-1][2] = max(merged[-1][2], r[2])
            else:
                merged.append(r)
        out += [(name, *r) for r in merged]
    return sorted(out, key=lambda x: x[1])

def fig_v3(D, rows, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32); nf = L.shape[1]; dur = D["meta"]["dur"]; npatch = int(np.ceil(nf / 10))
    P = D["aadd"].astype(np.float32)
    if len(P) == 0: P = np.zeros((int(np.ceil(dur / SEG)), len(CLASSES)), np.float32)
    pmax = P.max(1); active = [j for j in range(len(CLASSES)) if P[:, j].max() >= 0.5]
    wav, sr = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    step = max(1, sr // 4000); w = wav[::step]; tw = np.arange(len(w)) * step / sr
    K = len(rows)
    # token lane sub-rows (greedy, no horizontal overlap)
    lanes = []
    for r in rows:
        if "m2_seg" not in r: r["lane"] = 0; continue
        x0 = r["m2_seg"] * PT
        for li in range(8):
            while len(lanes) <= li: lanes.append([])
            if all(abs(x0 - x) >= max(PT, 0.085 * dur) for x in lanes[li]):
                lanes[li].append(x0); r["lane"] = li; break
        else:
            r["lane"] = 7
    nl = max(1, max(r.get("lane", 0) for r in rows) + 1)
    nrows = len(active) + 1
    hs = [1.0, 2.0, 0.22 * nrows, 0.34 * nl + 0.3, max(0.9, 0.11 * K)]
    fig = plt.figure(figsize=(13.5, 2.6 + sum(hs) * 1.2), facecolor=BG)
    gs = fig.add_gridspec(5, 1, height_ratios=hs, hspace=0.1, left=0.11, right=0.975, top=0.92, bottom=0.14)
    axw, axm, axc, axt, axh = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in (axw, axm, axc, axt, axh):
        ax.set_facecolor(BG); ax.set_xlim(0, dur)
        for s in ax.spines.values(): s.set_color(BORDER)
    # 1 waveform coloured by AADD max p
    pts = np.column_stack([tw, w]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
    fi = np.minimum((tw[:-1] / SEG).astype(int), len(pmax) - 1)
    lc = LineCollection(segs, colors=BLUE(Normalize(0, 1)(pmax[fi])), linewidths=0.7); axw.add_collection(lc)
    amp = float(np.abs(w).max() + 1e-6); axw.set_ylim(-amp * 1.2, amp * 1.2); axw.set_yticks([]); axw.tick_params(labelbottom=False)
    axw.set_ylabel("waveform", color=TXT2, fontsize=9, rotation=0, ha="right", va="center")
    for r in rows:
        if "m2_seg" in r:
            axw.plot([(r["m2_seg"] + 0.5) * PT], [amp * 1.08], marker="v", color=col(r["k"]), ms=5, clip_on=False, zorder=5)
    cax = fig.add_axes([0.885, 0.955, 0.09, 0.012]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=BLUE), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cax.tick_params(labelsize=6.5, colors=TXT2, length=2); cax.set_title("AADD max p", fontsize=7, color=TXT2, pad=2)
    # 2 mel in magma + token outlines
    axm.imshow(L, origin="lower", aspect="auto", cmap="magma", extent=[0, nf * 0.01, 0, 80], interpolation="nearest",
               vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    v1.mel_axis(axm, dur, xlabels=False); axm.set_ylabel("mel", color=TXT2, fontsize=9, rotation=0, ha="right", va="center")
    for r in rows:
        if "m2_patch" not in r: continue
        b, t = r["m2_patch"]
        axm.add_patch(Rectangle((t * PT, b * 10), PT, 10, fill=False, edgecolor=col(r["k"]), lw=1.4, zorder=3))
        axm.text(t * PT + PT / 2, b * 10 + 10.8, str(r["k"]), color=col(r["k"]), fontsize=6, ha="center", va="bottom", fontweight="bold", zorder=4)
    # 3 AADD class rows + max p
    axc.set_ylim(0, nrows); axc.set_xlim(0, dur); axc.tick_params(labelbottom=False); axc.set_yticks([])
    tseg = np.arange(len(P)) * SEG
    for ri, j in enumerate(list(active) + [None]):
        y = nrows - 1 - ri; p = pmax if j is None else P[:, j]
        axc.bar(tseg, np.ones(len(P)) * 0.86, width=SEG, bottom=y + 0.07, color=BLUE(np.clip(p, 0, 1)), align="edge", lw=0)
        axc.text(-0.012 * dur, y + 0.5, "max p" if j is None else CLASSES[j], ha="right", va="center", fontsize=7.5,
                 color=TXT if j is None or P[:, j].max() >= 0.5 else TXT2, fontweight="bold" if j is None else "normal")
        axc.axhline(y, color=GRID, lw=0.5)
    if not active: axc.text(dur / 2, nrows - 0.5, "no class reaches p ≥ 0.5 in this clip", ha="center", va="center", fontsize=7.5, color=MUTED)
    # 4 token lane
    axt.set_ylim(0, nl); axt.set_yticks([]); axt.tick_params(labelbottom=False); axt.grid(axis="x", color=GRID, lw=0.6)
    axt.set_ylabel("tokens on\ntop M2\nsegment", color=TXT2, fontsize=7.5, rotation=0, ha="right", va="center")
    for r in rows:
        if "m2_seg" not in r: continue
        x0 = r["m2_seg"] * PT; y = nl - 1 - r["lane"]
        axt.add_patch(Rectangle((x0, y + 0.1), PT, 0.8, facecolor=col(r["k"]), edgecolor="white", lw=0.5))
        axt.text(x0 + PT / 2, y + 0.5, str(r["k"]), color="white", fontsize=6, ha="center", va="center", fontweight="bold")
        axt.text(x0 + PT + 0.01 * dur, y + 0.5, r["tok"].strip() or "␣", color=col(r["k"]), fontsize=7.5, ha="left", va="center", fontweight="bold")
    # 5 heat strip
    H = np.zeros((K, npatch), np.float32)
    if D["m2"] is not None:
        for r in rows:
            p = np.clip(D["m2"][r["k"]], 0, None).sum(0); H[r["k"]] = p / (p.max() + 1e-12)
    axh.imshow(H, aspect="auto", cmap="Greys", extent=[0, npatch * PT, K - 0.5, -0.5], interpolation="nearest", vmin=0, vmax=1)
    axh.set_xlim(0, dur); axh.set_yticks(range(K))
    axh.set_yticklabels([f'{r["k"]} {r["tok"].strip() or "␣"}' for r in rows], fontsize=max(5, min(7.5, 160 / max(K, 1))))
    for lab, r in zip(axh.get_yticklabels(), rows): lab.set_color(col(r["k"]))
    axh.set_xlabel("time (s)", color=TXT2, labelpad=3); axh.tick_params(axis="x", colors=TXT2)
    axh.set_ylabel("M2 per token\n(row max = 1)", color=TXT2, fontsize=7.5, rotation=0, ha="right", va="center")
    # 6 chips: detected degraded intervals
    ivs = intervals(P)
    chips = [f"{n}  {a:.1f}–{b:.1f} s  max {m:.2f}" for n, a, b, m in ivs] or ["No degradation above the threshold."]
    x, y, line = 0.11, 0.095, 0
    for c in chips[:24]:
        t = fig.text(x, y - line * 0.022, c, fontsize=7, color=TXT2, va="top",
                     bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8", fc=BG, ec=BORDER, lw=0.8))
        x += 0.0062 * len(c) + 0.018
        if x > 0.86: x, line = 0.11, line + 1
    if len(chips) > 24: fig.text(0.11, y - (line + 1) * 0.022, f"… {len(chips) - 24} more intervals", fontsize=7, color=MUTED, va="top")
    title, _ = v1.header(D, model, clip, cls)
    fig.suptitle(title, x=0.11, ha="left", fontsize=11, fontweight="bold", color=TXT)
    fig.text(0.11, 0.004, CAPTION, fontsize=7, color=TXT2, wrap=True, va="bottom")
    fig.savefig(OUT / f"{stem}.png", dpi=160, facecolor=BG); fig.savefig(OUT / f"{stem}.svg", facecolor=BG); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--cls", default=None)
    a = ap.parse_args()
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
    D = v1.load(a.model, a.clip); rows = v1.analyse(D, tok)
    stem = f"{a.model}__{a.clip}__summary"
    fig_v3(D, rows, a.model, a.clip, a.cls or v1.KNOWN_CLS.get(a.clip), stem)
    v1.write_sidecar(D, rows, a.model, a.clip, a.cls, stem)
    print(stem, "v3", len(rows), "tokens")

if __name__ == "__main__":
    main()
