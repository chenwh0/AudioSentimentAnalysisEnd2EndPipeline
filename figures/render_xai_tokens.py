"""Per-token XAI figures (M2 occlusion primary, M1 grad x input for comparison).

python figures/render_xai_tokens.py --model whisper-tiny --clip <id> [--cls noise]
python figures/render_xai_tokens.py --manifest      # rebuild xai_tokens/MANIFEST.md from sidecars
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "Milestone2/AudioXAI/results"
OUT = ROOT / "figures/xai_tokens"
MODEL_DIR = {"whisper-tiny": RES / "npz", "whisper-base": RES / "whisper-base/npz",
             "aventiq": RES / "aventiq/npz"}
KNOWN_CLS = {"id10287_2iaCNe7ajII_00004": "clean", "id10285_uArtiTSTnSU_00014": "clean",
             "id10296_f_k09R8r_cA_00004": "noise", "id10288_R4S2c-NiPfA_00001": "reverb",
             "id10273_ZpPPaMazpf0_00003": "time-stretch"}
TOK_ID, TOK_REV = "openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

SURFACE, GRID, BORDER = "#fcfcfb", "#ecebe7", "#e4e3df"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#77766f"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c62f2f"
RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HEAT = LinearSegmentedColormap.from_list("aadd_blue", RAMP)
CAPTION = ("M2 (occlusion) is the primary method. M1 (gradient × input, SmoothGrad) is shown for "
           "comparison: it failed the model-randomization check (ρ ≈ 0.92 after re-initializing "
           "the top decoder layers) and tracks input energy (|M1| vs |input| correlation 0.78).")
CAPTION2 = "Lines link each token to the 100 ms segment whose occlusion hurts that token most."
HOP, PATCH = 0.01, 0.1
NEAR = 0.3


def rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "axes.titlecolor": TEXT, "grid.color": GRID, "grid.linewidth": 0.6, "axes.linewidth": 0.6,
        "xtick.major.size": 2, "ytick.major.size": 2, "svg.fonttype": "none"})


def esc(s):
    return s.replace("$", r"\$")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(model, clip):
    d = MODEL_DIR[model]
    p1, p2 = d / f"{clip}.npz", d / f"{clip}_m2.npz"
    z = np.load(p1)
    D = {k: z[k] for k in ("logmel", "m1", "aadd", "seq", "token_ts", "keep")}
    D["meta"] = json.loads(str(z["meta"]))
    D["m2"] = np.load(p2)["m2"].astype(np.float32) if p2.exists() else None
    D["files"] = [p1] + ([p2] if p2.exists() else [])
    return D


def analyse(D, tok):
    keep, seq, ts = D["keep"], D["seq"], D["token_ts"]
    m1 = np.abs(D["m1"].astype(np.float32))
    K, nmel, nf = m1.shape
    npatch = int(np.ceil(nf / 10))
    pad = npatch * 10 - nf
    rows = []
    for k in range(K):
        pos = int(keep[k]) + 1
        r = {"k": k, "id": int(seq[pos]), "tok": tok.decode([int(seq[pos])]), "t": float(ts[pos])}
        prof = np.pad(m1[k].sum(0), (0, pad)).reshape(npatch, 10).sum(1)
        r["m1_seg"] = int(np.argmax(prof))
        r["m1_band"] = int(np.argmax(m1[k].reshape(8, 10, nf).sum((1, 2))))
        if D["m2"] is not None:
            p = np.clip(D["m2"][k], 0, None)
            r["m2_seg"] = int(np.argmax(p.sum(0)))
            r["m2_band"] = int(np.argmax(p.sum(1)))
            b, t = np.unravel_index(int(np.argmax(p)), p.shape)
            r["m2_patch"] = (int(b), int(t))
        rows.append(r)
    return rows


def mel_axis(ax, dur, xlabels=True):
    ax.set_ylim(0, 80)
    ax.set_xlim(0, dur)
    ax.set_yticks([0, 80], ["0 Hz", "8k"])
    for s in ax.spines.values():
        s.set_color(BORDER)
    if not xlabels:
        ax.tick_params(labelbottom=False)


def overlay(ax, A, extent, gamma=1.0):
    A = A / (A.max() + 1e-12)
    rgba = HEAT(A)
    rgba[..., 3] = np.clip(A, 0, 1) ** gamma * 0.9
    ax.imshow(rgba, origin="lower", aspect="auto", extent=extent, interpolation="nearest", zorder=2)


def draw_logmel(ax, L, dur):
    ax.imshow(L, origin="lower", aspect="auto", cmap="gray_r", extent=[0, L.shape[1] * HOP, 0, 80],
              interpolation="nearest", vmin=np.percentile(L, 5), vmax=np.percentile(L, 99.5),
              alpha=0.6, zorder=1)


def tok_tick(ax, t):
    ax.axvline(t, color=TEXT, lw=0.7, ls=(0, (1, 1.5)), zorder=5)
    ax.plot([t], [80], marker="v", color=TEXT, ms=4, clip_on=False, zorder=6)


def header(D, model, clip, cls):
    c = D["meta"].get("cnt", {})
    wer = (c.get("S", 0) + c.get("D", 0) + c.get("I", 0)) / max(c.get("N", 1), 1)
    parts = [clip, model]
    if cls:
        parts.append(f"degradation: {cls}")
    parts.append(f"WER {wer:.1%} (S{c.get('S',0)} D{c.get('D',0)} I{c.get('I',0)} / N{c.get('N',0)})")
    return "  ·  ".join(parts), wer


def fig_tokens(D, rows, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32)
    nf = L.shape[1]
    dur = nf * HOP
    m1 = np.abs(D["m1"].astype(np.float32))
    K = len(rows)
    has2 = D["m2"] is not None
    nr = int(np.ceil(K / 4))
    fig = plt.figure(figsize=(17, 1.8 * nr + 1.5))
    top = 1 - 0.95 / fig.get_figheight()
    bot = 0.75 / fig.get_figheight()
    gs = fig.add_gridspec(nr, 11, width_ratios=[1, 1, 0.14, 1, 1, 0.14, 1, 1, 0.14, 1, 1],
                          left=0.035, right=0.995, top=top, bottom=bot, hspace=0.8, wspace=0.07)
    starts = [0, 3, 6, 9]
    for r in rows:
        k = r["k"]
        gr, gc = divmod(k, 4)
        for j, meth in enumerate(("M2", "M1")):
            ax = fig.add_subplot(gs[gr, starts[gc] + j])
            draw_logmel(ax, L, dur)
            if meth == "M2" and has2:
                p = np.clip(D["m2"][k], 0, None)
                overlay(ax, p, [0, p.shape[1] * PATCH, 0, 80], gamma=0.6)
                b, t = r["m2_patch"]
                ax.axvspan(r["m2_seg"] * PATCH, (r["m2_seg"] + 1) * PATCH, color=ORANGE, alpha=0.18, lw=0, zorder=3)
                ax.add_patch(Rectangle((t * PATCH, b * 10), PATCH, 10, fill=False, ec=ORANGE, lw=1.4, zorder=6))
            elif meth == "M2":
                ax.text(0.5, 0.5, "M2 not available", transform=ax.transAxes, ha="center", va="center",
                        color=TEXT2, fontsize=8, zorder=7,
                        bbox=dict(fc=SURFACE, ec=BORDER, lw=0.5))
            else:
                overlay(ax, np.minimum(m1[k], np.percentile(m1[k], 99.5)), [0, nf * HOP, 0, 80], gamma=0.8)
                ax.axvspan(r["m1_seg"] * PATCH, (r["m1_seg"] + 1) * PATCH, fc="none", ec=BLUE, ls="--", lw=0.8, zorder=4)
            tok_tick(ax, r["t"])
            mel_axis(ax, dur)
            if j == 1:
                ax.tick_params(labelleft=False)
            ax.set_title(("M2 occlusion" if meth == "M2" else "M1 |grad×input|"), fontsize=7.5,
                         color=TEXT2, pad=2)
            if j == 0:
                ax.text(1.0, 1.0 + 0.28, f'{k}: "{esc(r["tok"])}"  t={r["t"]:.2f}s', transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, color=TEXT, fontweight="bold")
            if gr == nr - 1 or k + 4 >= K:
                ax.set_xlabel("time (s)", fontsize=7.5, labelpad=1)
    title, _ = header(D, model, clip, cls)
    if not has2:
        title += "  ·  M1 only (no _m2.npz)"
    fig.text(0.035, 1 - 0.25 / fig.get_figheight(), title, ha="left", va="center", fontsize=11, fontweight="bold")
    fig.text(0.035, 0.32 / fig.get_figheight(), CAPTION, ha="left", va="center", fontsize=8.5, color=TEXT2, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}__tokens.{ext}", dpi=160)
    plt.close(fig)


def fig_summary(D, rows, model, clip, cls, stem):
    L = D["logmel"].astype(np.float32)
    nf = L.shape[1]
    dur = nf * HOP
    has2 = D["m2"] is not None
    W = max(9.0, min(18.0, 1.9 * dur + 2))
    fig = plt.figure(figsize=(W, 6.3))
    H = fig.get_figheight()
    gs = fig.add_gridspec(3, 1, height_ratios=[0.25, 2.6, 1.3], left=0.07, right=0.985,
                          top=1 - 0.75 / H, bottom=0.95 / H, hspace=0.0)
    axA = fig.add_subplot(gs[0])
    axM = fig.add_subplot(gs[1], sharex=axA)
    axT = fig.add_subplot(gs[2], sharex=axA)
    pos = axA.get_position()
    axA.set_position([pos.x0, pos.y0 + 0.02, pos.width, pos.height])
    # AADD strip
    pm = D["aadd"].astype(np.float32).max(1)
    rgba = np.zeros((1, len(pm), 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(ORANGE)
    rgba[0, :, 3] = np.clip(pm, 0, 1)
    axA.imshow(rgba, aspect="auto", extent=[0, len(pm) * 0.02, 0, 1], interpolation="nearest")
    axA.set_yticks([])
    axA.set_ylabel("AADD\np(max)", rotation=0, ha="right", va="center", fontsize=7.5, color=MUTED)
    axA.tick_params(labelbottom=False, bottom=False)
    for s in axA.spines.values():
        s.set_color(BORDER)
    # spectrogram
    draw_logmel(axM, L, dur)
    for r in rows:
        if has2:
            b, t = r["m2_patch"]
            axM.add_patch(Rectangle((t * PATCH, b * 10), PATCH, 10, fill=False, ec=ORANGE, lw=1.3, zorder=5))
        axM.axvspan(r["m1_seg"] * PATCH, (r["m1_seg"] + 1) * PATCH, fc="none", ec=BLUE, ls="--", lw=0.7,
                    alpha=0.8, zorder=4)
    mel_axis(axM, dur, xlabels=False)
    axM.set_ylabel("mel", fontsize=8)
    fig.text(0.07, 1 - 0.5 / H, ("Top M2 (occlusion) patch per token (orange box)   ·   top M1 (grad×input) "
                   "100 ms segment (dashed blue)") if has2 else
                  "Top M1 (grad×input) 100 ms segment per token (dashed blue)  ·  M2 not available",
             fontsize=9, ha="left", va="center", color=TEXT2)
    # tokens
    axT.set_ylim(-1.0, 1.0)
    axT.set_yticks([])
    for s in ("left", "right", "top"):
        axT.spines[s].set_visible(False)
    axT.spines["bottom"].set_color(BORDER)
    axT.axhline(-0.05, color=MUTED, lw=0.6, zorder=1)
    axT.set_xlabel("time (s)")
    axT.grid(axis="x", visible=True)
    axT.set_axisbelow(True)
    for i, r in enumerate(rows):
        y = -0.42 if i % 2 == 0 else -0.8
        axT.plot([r["t"]], [-0.05], marker="|", color=TEXT, ms=6, zorder=3)
        axT.text(r["t"], y, esc(r["tok"]), ha="center", va="center", fontsize=8.5, color=TEXT, zorder=4,
                 bbox=dict(fc=SURFACE, ec="none", pad=0.6))
        if not has2:
            continue
        c = (r["m2_seg"] + 0.5) * PATCH
        dist = c - r["t"]
        col = ORANGE if abs(dist) <= NEAR else RED
        con = ConnectionPatch(xyA=(r["t"], y + 0.1), coordsA=axT.transData, xyB=(c, 0), coordsB=axM.transData,
                              color=col, lw=0.9, alpha=0.85, zorder=2)
        fig.add_artist(con)
        if abs(dist) > NEAR:
            axT.text(r["t"] + 0.03, y - 0.17, f"{dist:+.2f}s", ha="left", va="center", fontsize=6.5, color=RED)
    title, _ = header(D, model, clip, cls)
    if not has2:
        title += "  ·  M1 only (no _m2.npz)"
    fig.text(0.07, 1 - 0.22 / H, title, ha="left", va="center", fontsize=10.5, fontweight="bold")
    fig.text(0.07, 0.3 / H, CAPTION + " " + (CAPTION2 if has2 else ""), ha="left", va="center",
             fontsize=8, color=TEXT2, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}__summary.{ext}", dpi=160)
    plt.close(fig)


def write_sidecar(D, rows, model, clip, cls, stem):
    _, wer = header(D, model, clip, cls)
    side = {"model": model, "asr": D["meta"].get("asr"), "clip": clip, "cls": cls, "wer": wer,
            "has_m2": D["m2"] is not None,
            "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in D["files"]],
            "figures": [f"{stem}__tokens.png", f"{stem}__summary.png"], "tokens": rows}
    (OUT / f"{stem}.json").write_text(json.dumps(side, indent=1))


def manifest():
    lines = ["# xai_tokens figures", "",
             "Generated by `figures/render_xai_tokens.py`. Times in seconds. M2 top segment = argmax_t sum_b clip(m2,0) "
             "(segment start = t*0.1); M2 band b = mel bins 10b..10b+9; M1 top segment = argmax of 100 ms-summed "
             "sum_mel |m1|. Token time = token_ts[keep[k]+1].", "",
             "| figure | model (asr id) | clip | class | npz read (sha256) |", "|---|---|---|---|---|"]
    sides = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    order = {"whisper-tiny": 0, "whisper-base": 1, "aventiq": 2}
    sides.sort(key=lambda s: (order.get(s["model"], 9), s["clip"]))
    for s in sides:
        files = "<br>".join(f"`{f['path']}` `{f['sha256']}`" for f in s["files"])
        for fg in s["figures"]:
            lines.append(f"| `{fg}` | {s['model']} (`{s['asr']}`) | {s['clip']} | {s['cls'] or ''} | {files} |")
    for s in sides:
        lines += ["", f"## {s['model']} / {s['clip']}" + ("" if s["has_m2"] else " (M1 only, no _m2.npz)"),
                  "", f"WER {s['wer']:.3f}", "",
                  "| k | token | token t | M2 top seg start | M2 top band | M1 top seg start |",
                  "|---|---|---|---|---|---|"]
        for r in s["tokens"]:
            tk = r["tok"].replace("|", "\\|")
            m2s = f"{r['m2_seg']*PATCH:.1f}" if "m2_seg" in r else "n/a"
            m2b = str(r["m2_band"]) if "m2_band" in r else "n/a"
            lines.append(f"| {r['k']} | `{tk}` | {r['t']:.2f} | {m2s} | {m2b} | {r['m1_seg']*PATCH:.1f} |")
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_DIR), default="whisper-tiny")
    ap.add_argument("--clip")
    ap.add_argument("--cls", default=None, help="degradation class label for the title")
    ap.add_argument("--manifest", action="store_true", help="rebuild MANIFEST.md from sidecar JSONs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        manifest()
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(TOK_ID, revision=TOK_REV)
    rc()
    cls = a.cls or KNOWN_CLS.get(a.clip)
    D = load(a.model, a.clip)
    rows = analyse(D, tok)
    stem = f"{a.model}__{a.clip}"
    fig_tokens(D, rows, a.model, a.clip, cls, stem)
    fig_summary(D, rows, a.model, a.clip, cls, stem)
    write_sidecar(D, rows, a.model, a.clip, cls, stem)
    print(stem, "m2" if D["m2"] is not None else "M1-only", len(rows), "tokens")


if __name__ == "__main__":
    main()
