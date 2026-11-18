"""Pick the 20-clip study set from the 50-clip stratified test sample shared by whisper-tiny, whisper-base and AventIQ.
Hard set (10): uncertainty sampling on whisper-tiny: the 10 clips with the lowest mean teacher-forced token log-prob (lp0 in the npz).
Mixed set (10): from the remaining 40, strata = (confident / uncertain by median mean log-prob) x (short / long by median duration)
x (clean / degraded by AADD class); default_rng(8001) permutes each stratum; strata are visited in a fixed order, one clip per visit,
until 10 are drawn. Writes results/token_segments/groups.csv."""
import hashlib, json, sys
import numpy as np, pandas as pd
from pathlib import Path
X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X))
from audioxai import core, paths
# Reproducibility slices: Chen's saved splits (never re-split). Every study clip must be in the saved test slice.
dev, test, load = core.slices()
test_idx = test.set_index("file")["original_index"].to_dict()
agg = pd.read_csv(X / "results/aggregate.csv")
assert set(agg.file) <= set(test.file), "50-clip pool is not inside the saved test slice"
assert not (set(agg.file) & set(dev.file)) and not (set(agg.file) & load), "pool touches dev or load pool"
rows = []
for c in agg.file:
    z = np.load(X / "results/npz" / f"{c}.npz")
    rows.append(dict(clip=c, mean_lp0=float(z["lp0"].mean()), n_tok=int(len(z["lp0"]))))
d = pd.DataFrame(rows).merge(agg[["file", "duration", "degradation_class", "label", "wer"]], left_on="clip", right_on="file").drop(columns="file")
d = d.sort_values("mean_lp0").reset_index(drop=True)
hard = list(d["clip"][:10])
rest = d[~d["clip"].isin(hard)].copy()
rest["conf"] = np.where(rest.mean_lp0 >= rest.mean_lp0.median(), "confident", "uncertain")
rest["length"] = np.where(rest.duration >= rest.duration.median(), "long", "short")
rest["quality"] = np.where(rest.degradation_class == "clean", "clean", "degraded")
rng = np.random.default_rng(8001)
strata = {}
for key, g in rest.groupby(["conf", "length", "quality"]):
    strata[key] = list(np.array(g["clip"])[rng.permutation(len(g))])
order = sorted(strata)
mixed = []
while len(mixed) < 10:
    for key in order:
        if strata[key] and len(mixed) < 10:
            mixed.append(strata[key].pop(0))
d["group"] = np.where(d["clip"].isin(hard), "hard", np.where(d["clip"].isin(mixed), "mixed", "background"))
d["slice"] = "test"
d["original_index"] = d["clip"].map(test_idx)
assert d.original_index.notna().all()
d = d.merge(rest[["clip", "conf", "length", "quality"]], on="clip", how="left")
out = X / "results/token_segments"; out.mkdir(exist_ok=True)
d.to_csv(out / "groups.csv", index=False)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
prov = {"pool": "results/test_sample_ids.csv (50-clip stratified sample of the saved test slice, scan.test_sample(n=50, seed=8001))",
        "test_slice_csv": str(paths.PE / "test.csv"), "test_slice_sha256": sha(paths.PE / "test.csv"), "test_slice_rows": int(len(test)),
        "dev_rows": int(len(dev)), "load_pool_rows": int(len(load)), "load_pool_read": False,
        "indices_files": {f.name: sha(f) for f in sorted(paths.PE.glob("*indices*.npy"))},
        "hard_rule": "10 lowest mean whisper-tiny token log-prob (lp0 in results/npz/<clip>.npz)",
        "mixed_rule": "strata conf x length x quality on the remaining 40; default_rng(8001) permutation per stratum; strata in sorted order, one per visit, until 10",
        "seed": 8001}
json.dump(prov, open(out / "groups_provenance.json", "w"), indent=1)
print(d[d.group != "background"].sort_values(["group", "mean_lp0"]).to_string(index=False))
print("strata sizes:", {k: len(v) for k, v in zip(order, [rest[(rest.conf == a) & (rest.length == b) & (rest.quality == c)] for a, b, c in order])})
