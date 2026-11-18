"""Per-clip token-to-audio summary figure, v2 (colour links, no connector lines).
Panels, top to bottom, sharing the time axis:
  1. the waveform, coloured per 20 ms frame by the AADD degradation flag (clean grey, degraded charcoal), with each
     token's top M2 occlusion segment painted over it in the token's colour;
  2. the log-mel spectrogram the model saw, with each token's top M2 patch (100 ms x 10 mel) filled in the token's colour;
  3. a thin strip with each token's top M1 (gradient x input) 100 ms segment in the same colour;
  4. the token labels at their own timestamps, in their colours;
  5. a token x time heat strip: rows = tokens, columns = 100 ms patches, cell = that token's M2 attribution normalized to its max.
Colours: the dataviz reference categorical palette (8 slots, validated), cycled; the token index is printed in every region so a
repeated colour is never ambiguous. Usage: python figures/render_xai_tokens_v2.py --model whisper-tiny --clip <id> [--cls text]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_xai_tokens as v1

ROOT = v1.ROOT; OUT = v1.OUT
AUDIO = ROOT / "Milestone2/PipelineEvaluation/data/audio"
BG, GRID, BORDER = "#fcfcfb", "#ecebe7", "#e4e3df"
TXT, TXT2, MUTED = "#0b0b0b", "#52514e", "#77766f"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLEAN, DEGR = "#b5b9b4", "#2e2e2c"
HEAT = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
HOP, SEG, PT = 0.01, 0.02, 0.1
CAPTION = ("Colour links each output token to the audio it used: the same colour marks the token label, its top M2 occlusion "
           "patch on the spectrogram and on the waveform, and its top M1 segment in the thin strip. Colours repeat every 8 tokens, "
           "so the token index is printed in every region. M2 (occlusion of 100 ms × 10-mel patches) is the primary method; M1 "
           "(gradient × input, SmoothGrad) is shown for comparison only, because it failed the model-randomization check "
           "(ρ ≈ 0.92) and tracks input energy (|M1| vs |input| 0.78). Waveform colour is context only: charcoal = AADD flags "
           "the 20 ms frame as degraded (max class probability ≥ 0.5), grey = clean.")

def col(k): return CAT[k % len(CAT)]

def fig_summary_v2(D, rows, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32); nf = L.shape[1]; dur = D["meta"]["dur"]
    npatch = int(np.ceil(nf / 10))
    wav, sr = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    step = max(1, sr // 4000); w = wav[::step]; tw = np.arange(len(w)) * step / sr
    aadd = D["aadd"].astype(np.float32); flag = (aadd.max(1) >= 0.5) if len(aadd) else np.zeros(int(np.ceil(dur / SEG)), bool)
    fidx = np.minimum((tw / SEG).astype(int), len(flag) - 1) if len(flag) else np.zeros(len(tw), int)
    deg = flag[fidx] if len(flag) else np.zeros(len(tw), bool)
    K = len(rows)
    hs = [1.0, 2.2, 0.28, 0.75, max(0.9, 0.11 * K)]
    fig = plt.figure(figsize=(13.5, 1.9 + sum(hs) * 1.25), facecolor=BG)
    gs = fig.add_gridspec(5, 1, height_ratios=hs, hspace=0.12, left=0.105, right=0.985, top=0.9, bottom=0.1 + 0.02 * (K > 30))
    axw, axm, ax1, axt, axh = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in (axw, axm, ax1, axt, axh):
        ax.set_facecolor(BG); ax.set_xlim(0, dur)
        for s in ax.spines.values(): s.set_color(BORDER)
    # 1 waveform coloured by AADD flag
    wc = np.where(deg, np.nan, w); wd = np.where(deg, w, np.nan)
    axw.plot(tw, wc, color=CLEAN, lw=0.6, zorder=2); axw.plot(tw, wd, color=DEGR, lw=0.6, zorder=3)
    amp = float(np.abs(w).max() + 1e-6); axw.set_ylim(-amp * 1.15, amp * 1.15); axw.set_yticks([])
    axw.tick_params(labelbottom=False); axw.grid(axis="x", color=GRID, lw=0.6)
    axw.set_ylabel("waveform", color=TXT2, fontsize=9, rotation=0, ha="right", va="center")
    for r in rows:
        if "m2_seg" not in r: continue
        x0 = r["m2_seg"] * PT
        axw.axvspan(x0, x0 + PT, color=col(r["k"]), alpha=0.32, lw=0, zorder=1)
        axw.text(x0 + PT / 2, amp * 1.0, str(r["k"]), color=col(r["k"]), fontsize=6.5, ha="center", va="top", fontweight="bold", zorder=4)
    axw.legend(handles=[Patch(color=CLEAN, label="clean"), Patch(color=DEGR, label="degraded, AADD p ≥ 0.5")], loc="upper right",
               frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(1.0, 1.28), handlelength=1.2)
    # 2 spectrogram with M2 top patches
    v1.draw_logmel(axm, L, dur); v1.mel_axis(axm, dur, xlabels=False)
    axm.set_ylabel("mel", color=TXT2, fontsize=9, rotation=0, ha="right", va="center")
    for r in rows:
        if "m2_patch" not in r: continue
        b, t = r["m2_patch"]
        axm.add_patch(Rectangle((t * PT, b * 10), PT, 10, facecolor=col(r["k"]), edgecolor="white", lw=0.6, alpha=0.85, zorder=3))
        axm.text(t * PT + PT / 2, b * 10 + 5, str(r["k"]), color="white", fontsize=6, ha="center", va="center", fontweight="bold", zorder=4)
    axm.set_title("Top M2 occlusion patch per token, on the log-mel the model saw", loc="left", fontsize=9, color=TXT2, pad=6)
    # 3 M1 top-segment strip
    ax1.set_ylim(0, 1); ax1.set_yticks([]); ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("M1 top\nsegment", color=TXT2, fontsize=7.5, rotation=0, ha="right", va="center")
    for r in rows:
        x0 = r["m1_seg"] * PT
        ax1.add_patch(Rectangle((x0, 0.1), PT, 0.8, facecolor=col(r["k"]), edgecolor="white", lw=0.4, alpha=0.9))
        ax1.text(x0 + PT / 2, 0.5, str(r["k"]), color="white", fontsize=5.5, ha="center", va="center", fontweight="bold")
    # 4 token labels at their timestamps
    axt.set_ylim(0, 1); axt.set_yticks([]); axt.tick_params(labelbottom=False); axt.grid(axis="x", color=GRID, lw=0.6)
    axt.set_ylabel("tokens at\ntimestamp", color=TXT2, fontsize=7.5, rotation=0, ha="right", va="center")
    for i, r in enumerate(rows):
        y = 0.72 if i % 2 == 0 else 0.25
        axt.plot([r["t"]], [y + 0.14], marker="v", color=col(r["k"]), ms=3.5, clip_on=False)
        axt.text(r["t"], y - 0.02, f'{r["k"]} {r["tok"].strip() or "␣"}', color=col(r["k"]), fontsize=7.5, ha="center", va="top", fontweight="bold")
    # 5 token x time heat strip
    H = np.zeros((K, npatch), np.float32)
    for r in rows:
        if D["m2"] is None: break
        p = np.clip(D["m2"][r["k"]], 0, None).sum(0); H[r["k"]] = p / (p.max() + 1e-12)
    axh.imshow(H, aspect="auto", cmap=HEAT, extent=[0, npatch * PT, K - 0.5, -0.5], interpolation="nearest", vmin=0, vmax=1)
    axh.set_yticks(range(K)); axh.set_yticklabels([f'{r["k"]} {r["tok"].strip() or "␣"}' for r in rows], fontsize=max(5, min(7.5, 160 / max(K, 1))))
    for lab, r in zip(axh.get_yticklabels(), rows): lab.set_color(col(r["k"]))
    axh.set_xlabel("time (s)", color=TXT2, labelpad=4); axh.set_ylabel("M2 per token\n(row max = 1)", color=TXT2, fontsize=7.5, rotation=0, ha="right", va="center")
    axh.tick_params(axis="x", colors=TXT2); axh.set_xlim(0, dur)
    title, _ = v1.header(D, model, clip, cls)
    fig.suptitle(title, x=0.105, ha="left", fontsize=11, fontweight="bold", color=TXT)
    fig.text(0.105, 0.006, CAPTION, fontsize=7.2, color=TXT2, wrap=True, va="bottom")
    fig.savefig(OUT / f"{stem}.png", dpi=160, facecolor=BG); fig.savefig(OUT / f"{stem}.svg", facecolor=BG); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--cls", default=None)
    a = ap.parse_args()
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
    D = v1.load(a.model, a.clip); rows = v1.analyse(D, tok)
    cls = a.cls or v1.KNOWN_CLS.get(a.clip)
    stem = f"{a.model}__{a.clip}__summary"
    fig_summary_v2(D, rows, a.model, a.clip, cls, stem)
    v1.write_sidecar(D, rows, a.model, a.clip, cls, stem)
    print(stem, "v2", len(rows), "tokens")

if __name__ == "__main__":
    main()
