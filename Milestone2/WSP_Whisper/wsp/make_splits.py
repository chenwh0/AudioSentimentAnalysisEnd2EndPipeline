"""Step 1: label-stratified practice/holdout split of the 500 dev clips, default_rng(8001)."""
import hashlib, json, os
import numpy as np, pandas as pd
R = "/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline"
PE = f"{R}/Milestone2/PipelineEvaluation"
OUT = f"{R}/Milestone2/WSP_Whisper/splits"
dev = pd.read_csv(f"{PE}/development.csv").sort_values("original_index").reset_index(drop=True)
assert len(dev) == 500
rng = np.random.default_rng(8001)
practice, holdout = [], []
for lab in sorted(dev["label"].unique()):
    pos = np.flatnonzero(dev["label"].values == lab)
    perm = rng.permutation(pos)
    k = len(pos) // 2
    practice += dev.loc[perm[:k], "original_index"].tolist()
    holdout += dev.loc[perm[k:], "original_index"].tolist()
practice = np.array(practice, dtype=np.int64); holdout = np.array(holdout, dtype=np.int64)
test = np.load(f"{PE}/test_indices.npy"); load = np.load(f"{PE}/load_pool_indices.npy")
S = lambda a: set(a.tolist())
assert not (S(practice) & S(holdout)); assert not ((S(practice) | S(holdout)) & S(test))
assert not ((S(practice) | S(holdout)) & S(load)); assert len(S(practice) | S(holdout)) == 500
np.save(f"{OUT}/wsp_practice_ids.npy", practice); np.save(f"{OUT}/wsp_holdout_ids.npy", holdout)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
lc = lambda ids: dev.set_index("original_index").loc[ids, "label"].value_counts().sort_index().to_dict()
man = {"seed": 8001, "source": "Milestone2/PipelineEvaluation/development.csv sorted by original_index",
       "method": "labels alphabetical; rng.permutation of row positions per label; first floor(n/2) -> practice",
       "id_type": "original_index", "n_practice": int(len(practice)), "n_holdout": int(len(holdout)),
       "label_counts_practice": lc(practice), "label_counts_holdout": lc(holdout),
       "sha256": {"wsp_practice_ids.npy": sha(f"{OUT}/wsp_practice_ids.npy"), "wsp_holdout_ids.npy": sha(f"{OUT}/wsp_holdout_ids.npy")},
       "disjoint_checks": "practice/holdout/test_indices/load_pool_indices all pairwise disjoint (asserted)"}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2, default=int)
open(f"{OUT}/READY", "w").write("ready\n")
print(json.dumps(man, indent=1, default=int))
