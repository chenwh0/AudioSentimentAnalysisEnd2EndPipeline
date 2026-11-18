"""M2 occlusion for a fixed clip list, rebuilt from the saved .npz (seq, keep, n_frames) plus the WAV.
Usage: AUDIOXAI_ASR=<model id> python scripts/m2_for_clips.py <results subdir> <clip> [<clip> ...]
Writes results/<subdir>/npz/<clip>_m2.npz with m2 (K, 8 bands, n_100ms_patches) float16, same layout as occlusion.m2_occlusion."""
import json, sys, time
from pathlib import Path
import numpy as np
from audioxai import core, occlusion, paths

sub, clips = sys.argv[1], sys.argv[2:]
od = paths.X / "results" / sub / "npz"
for c in clips:
    out = od / f"{c}_m2.npz"
    if out.exists():
        print("skip", c, flush=True); continue
    z = np.load(od / f"{c}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    assert meta["asr"] == paths.ASR_ID, (meta["asr"], paths.ASR_ID)
    wav, _ = core.load_audio(c)
    res = dict(wav=wav, seq=z["seq"], keep=z["keep"], n_frames=int(meta["n_frames"]))
    t0 = time.time()
    m2 = occlusion.m2_occlusion(res)
    np.savez_compressed(out, m2=m2.astype(np.float16))
    print("done", sub, c, m2.shape, f"{time.time()-t0:.0f}s", flush=True)
