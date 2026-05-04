# Incremental training guide

`vi_mogp.hierarchical.train` saves enough state on every run that you can add
a future data batch without retraining the 524-expert world from scratch.

## What every `train.py --out <dir>` run writes

| Path | Content |
|:---|:---|
| `<dir>/partition.pkl` | Frozen BGM (K_geo, K_phi, scalers) + cluster assignment for every training row |
| `<dir>/experts/expert_NNNN.pt` | Per-expert GP state_dict |
| `<dir>/baselines/baseline_NN.pt` | Per-geo GRBCM baseline (K_geo of them) |
| `<dir>/clusters/cluster_NNNN.csv` | The rows that fell into each cluster |
| `<dir>/model.pt` | Aggregated checkpoint (used by `predict.py` / `invert_rbcm.py`) |
| `<dir>/meta.json` | Run metadata |
| `<dir>/undersampled_clusters.csv` | Clusters with N < 30 (served by per-geo baseline) |

## Resume rule (train.py:228–231)

On startup, train.py scans `<dir>/experts/expert_*.pt`:

- file **exists** ⇒ expert is **loaded and skipped** (no Adam)
- file **missing** ⇒ expert is trained fresh and saved

`--load-partition <path-to-existing-partition.pkl>` tells train.py not to
refit the BGMs; it only re-runs the routing step on the (possibly extended)
input CSV.

## Typical incremental recipe

Say you have a new data batch `new_batch.csv` and want to fold it in without
re-fitting BGMs or re-training unchanged experts.

```bash
# 1. Build the extended training CSV (or append rows; dedup recommended).
python3 - <<'PY'
import pandas as pd
old = pd.read_csv("Optimization/<curr_workspace>/train_merged.csv")
new = pd.read_csv("new_batch.csv")
CORE_IN = ["n","eta","sigma_y","width","height"]
k_old = set(map(tuple, old[CORE_IN].round(4).to_numpy()))
new = new.loc[[tuple(r.round(4)) not in k_old for r in new[CORE_IN].to_numpy()]]
pd.concat([old, new], ignore_index=True).to_csv(
    "Optimization/<new_workspace>/train_merged.csv", index=False)
PY

# 2. Copy the previous run's artifacts into a fresh output dir.
cp -r vi_mogp/hierarchical/results/<prev_run> \
      vi_mogp/hierarchical/results/<new_run>

# 3. Identify which experts gained rows and delete their checkpoint
#    so train.py will re-train them.
python3 - <<'PY'
import pickle, pandas as pd
from pathlib import Path

PART = Path("vi_mogp/hierarchical/results/<new_run>/partition.pkl")
with open(PART, "rb") as f:
    part = pickle.load(f)

# route the NEW rows through the saved BGMs
import numpy as np
from vi_mogp.hierarchical.model import HVIMoGP_rBCM  # uses same scalers

new = pd.read_csv("new_batch.csv")
geo = part["geo_gmm"].predict(part["geo_scaler"].transform(new[["width","height"]]))
# ... (see predict.py for the exact phi-routing call) ...
# For each new row, compute the cluster index.
# Collect the set of affected cluster indices.
AFFECTED = set([...])   # integer cluster IDs

experts_dir = Path("vi_mogp/hierarchical/results/<new_run>/experts")
for cid in AFFECTED:
    p = experts_dir / f"expert_{cid:04d}.pt"
    if p.exists(): p.unlink()
print(f"  cleared {len(AFFECTED)} expert checkpoints; train.py will re-train them")
PY

# 4. Retrain — partition frozen, only affected experts re-learn.
python3 -m vi_mogp.hierarchical.train \
    --train-csv Optimization/<new_workspace>/train_merged.csv \
    --val-csv   Optimization/<new_workspace>/val_merged.csv \
    --test-csv  Optimization/<new_workspace>/test_merged.csv \
    --load-partition vi_mogp/hierarchical/results/<new_run>/partition.pkl \
    --out            vi_mogp/hierarchical/results/<new_run>

# 5. Rerun hotfix + poly residual (cheap compared to full train).
python3 -m vi_mogp.hierarchical.diagnose_model ...
python3 -m vi_mogp.hierarchical.hotfix_retrain ...
python3 -m vi_mogp.hierarchical.add_poly_residual ...

# 6. Benchmark.
python3 -m vi_mogp.hierarchical.invert_rbcm ...
```

## Caveats

1. **Partition is frozen by design.** If the new data shifts the (W, H) or φ(y)
   distribution substantially, the old BGM may underfit it. In that case rerun
   `train.py` *without* `--load-partition` (i.e. the fresh path) to re-cluster
   from scratch. Rule of thumb: if new rows account for < 3 % of the merged
   training set (as with `sigma_low_2500` / 297 919), frozen partition is fine.
2. **Hotfix / poly outputs live in separate directories.** Don't delete
   `rbcm_v2_fresh/` just because `rbcm_v2_fresh_hotfix_poly/` exists — the
   raw `experts/expert_*.pt` are only in the former and are the resume source.
3. **Deleting `expert_NNNN.pt` is the ONLY way to force re-training** of a
   specific expert while keeping the rest frozen. Don't edit inside `model.pt`.
4. **Bake the run tag into filenames**, not just the directory. All diagnostic
   CSVs and benchmark CSVs should include the model tag (e.g.
   `per_expert_mae_rbcm_v2_diag.csv`, `rbcm_v2_kphi30.csv`).
