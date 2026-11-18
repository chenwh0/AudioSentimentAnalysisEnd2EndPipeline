"""Faithfulness checks for M1: deletion curves and model-randomization (Adebayo et al., 2018)."""
import copy

import numpy as np
import torch
from scipy.stats import spearmanr

from . import core

KS = list(range(5, 55, 5))
PT, PF = 10, 10  # patch = 10 frames (100 ms) x 10 mel bins (1/8 of 80)


def patch_saliency(maps, n_frames):
    a = np.abs(maps)
    a = a / np.maximum(a.sum((1, 2), keepdims=True), 1e-12)
    s = a.sum(0)  # (80, n_frames)
    nt = int(np.ceil(n_frames / PT))
    pad = nt * PT - n_frames
    s = np.pad(s, ((0, 0), (0, pad)))
    return s.reshape(80 // PF, PF, nt, PT).sum((1, 3))  # (8, nt)


def deletion(res, n_random=3, seed=8001, batch=12):
    proc, m = core.asr()
    feats, _ = core.features(res["wav"])
    seq = torch.as_tensor(res["seq"], device=core.DEV)
    keep = list(res["keep"])
    nf = res["n_frames"]
    sal = patch_saliency(res["maps"], nf)
    P = sal.size
    order = np.argsort(-sal.ravel())
    rng = np.random.default_rng(seed)
    fill = float(feats[0, :, :nf].min())
    masks, tags = [], []
    for k in KS:
        n = max(1, int(round(k / 100 * P)))
        masks.append(order[:n]); tags.append(("top", k))
        for r in range(n_random):
            masks.append(rng.permutation(P)[:n]); tags.append(("rand", k))

    def apply(idx):
        x = feats.clone()
        for p in idx:
            fb, tb = divmod(int(p), sal.shape[1])
            x[0, fb * PF:(fb + 1) * PF, tb * PT:min((tb + 1) * PT, nf)] = fill
        return x

    with torch.no_grad():
        lp0 = core.teacher_forced_logp(m, feats, seq, keep).sum(1).item()
        lps = []
        for i in range(0, len(masks), batch):
            x = torch.cat([apply(mk) for mk in masks[i:i + batch]])
            lps += core.teacher_forced_logp(m, x, seq, keep).sum(1).float().cpu().tolist()
    top = {k: 0.0 for k in KS}; rnd = {k: [] for k in KS}
    for (t, k), lp in zip(tags, lps):
        d = (lp0 - lp) / max(len(keep), 1)  # drop in mean per-token log-prob (nats)
        if t == "top": top[k] = d
        else: rnd[k].append(d)
    rnd = {k: float(np.mean(v)) for k, v in rnd.items()}
    x = np.array(KS) / 100
    auc_top = float(np.trapezoid([top[k] for k in KS], x))
    auc_rand = float(np.trapezoid([rnd[k] for k in KS], x))
    out = dict(file=res["meta"]["file"], n_patches=P, n_tokens=len(keep), lp0_per_token=lp0 / max(len(keep), 1),
               auc_top=auc_top, auc_random=auc_rand, auc_diff=auc_top - auc_rand)
    for k in KS:
        out[f"drop_top_{k}"] = top[k]; out[f"drop_rand_{k}"] = rnd[k]
    return out


_RAND = {}


def randomized_model(seed=8001, n_layers=2):
    if "m" not in _RAND:
        _, m = core.asr()
        mr = copy.deepcopy(m)
        torch.manual_seed(seed)
        for layer in mr.model.decoder.layers[-n_layers:]:
            for mod in layer.modules():
                if isinstance(mod, torch.nn.Linear):
                    torch.nn.init.normal_(mod.weight, std=0.02)
                    if mod.bias is not None: torch.nn.init.zeros_(mod.bias)
                elif isinstance(mod, torch.nn.LayerNorm):
                    torch.nn.init.ones_(mod.weight); torch.nn.init.zeros_(mod.bias)
        for p in mr.parameters():
            p.requires_grad_(False)
        _RAND["m"] = mr.eval()
    return _RAND["m"]


def randomization(res, smooth_n=8):
    mr = randomized_model()
    feats, _ = core.features(res["wav"])
    seq = torch.as_tensor(res["seq"], device=core.DEV)
    keep = list(res["keep"])
    maps_r, _ = core.m1_smoothgrad(mr, feats, seq, keep, n=smooth_n, n_frames=res["n_frames"])
    a, b = np.abs(res["maps"]), np.abs(maps_r)
    rho_tf = spearmanr(a.mean(0).ravel(), b.mean(0).ravel()).statistic
    rho_t = spearmanr(a.sum(1).mean(0), b.sum(1).mean(0)).statistic
    rho_signed = spearmanr(res["maps"].mean(0).ravel(), maps_r.mean(0).ravel()).statistic
    # per-token TF rank correlation, median
    per_tok = [spearmanr(a[k].ravel(), b[k].ravel()).statistic for k in range(min(len(keep), 20))]
    # input-only baseline: how much does |x| alone explain the map? (grad x input inherits structure of x)
    x = np.abs(res["logmel"]).ravel()
    rho_input = spearmanr(a.mean(0).ravel(), x).statistic
    return dict(file=res["meta"]["file"], rho_tf_abs=float(rho_tf), rho_time_abs=float(rho_t),
                rho_tf_signed=float(rho_signed), rho_token_median=float(np.median(per_tok)) if per_tok else np.nan,
                rho_map_vs_absinput=float(rho_input), pass_check=bool(rho_tf < 0.5))
