"""y-shape architecture densification.

Anti-drift is automatic: candidates always route to whichever sub their
simulated y matches by shape. No gate-1 rejection — every successful sim
contributes to its natural sub.

Workflow per gid:
  1. Read state.pkl (length edges, per-bin PCA + KMeans centroids)
  2. Read sub_assignments.csv (existing training data + sub_id labels)
  3. Per sparse sub (N<target): pick anchor points + Gaussian perturb in z
  4. Plus a broad LHS pool in gid's z-box
  5. Sample (W, H) per candidate from gid training distribution
  6. Run sim batch via HeadlessSimulatorMLS
  7. For each result: route y -> length_bin -> shape_sub via PCA+KMeans
  8. Append accepted (theta, y) rows; per-sub counters update

Usage (smoke):
  python -m surrogate.densify_yshape \
      --gid 5 --n-sims 100 --target-N 300
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# HeadlessSimulatorMLS is only used by the CLI densification entry point
# (see __main__ below).  Do NOT import it at module level — that would
# pull in DataPipeline.headless_mls → taichi + MPM kernels for every
# importer of this module (e.g. Optimization.libs.selector at inverse
# time, where no simulation runs).  Use lazy import inside the CLI block.


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def norm_curve_one(y8: np.ndarray) -> np.ndarray:
    """7-d shape feature for a single y(t) curve."""
    y1 = float(y8[0])
    y_8 = max(float(y8[7]), 1e-6)
    return (y8[1:8] - y1) / y_8


def route_y_to_sub(y_obs: np.ndarray, state: dict) -> tuple[int, int]:
    """Return (length_bin_id, global_sub_id) or (-1, -1) if outside coverage."""
    edges = state["length_edges"]
    y8 = float(y_obs[7])
    bin_id = -1
    for b in state["bins"]:
        if b["y8_lo"] <= y8 < b["y8_hi"]:
            bin_id = int(b["bin_id"])
            break
    if bin_id < 0:
        # outside training y8 range; clamp to nearest bin
        if y8 < edges[0]:
            bin_id = int(state["bins"][0]["bin_id"])
        else:
            bin_id = int(state["bins"][-1]["bin_id"])
    bin_obj = next(b for b in state["bins"] if b["bin_id"] == bin_id)
    nc = norm_curve_one(y_obs)
    # standardize using bin's scaler, then PCA-project
    nc_s = (nc - bin_obj["scaler"]["mean"]) / bin_obj["scaler"]["std"]
    pc = bin_obj["pca"]["components"] @ (nc_s - bin_obj["pca"]["mean"])
    centers = bin_obj["kmeans_centroids"]
    d = np.linalg.norm(centers - pc[None, :], axis=1)
    shape_id = int(np.argmin(d))
    # find global sub_id with this (bin_id, shape_id); subs may have been merged
    for s in bin_obj["subs"]:
        if s["shape_id"] == shape_id:
            return bin_id, int(s["sub_id"])
    # if shape_id was merged, route to nearest *existing* sub
    centers_existing = np.stack([s["centroid_pc"] for s in bin_obj["subs"]])
    d2 = np.linalg.norm(centers_existing - pc[None, :], axis=1)
    near = int(np.argmin(d2))
    return bin_id, int(bin_obj["subs"][near]["sub_id"])


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

def per_sub_anchors(df_assign: pd.DataFrame, sub_id: int,
                     n_anchors: int, rng: np.random.Generator) -> pd.DataFrame:
    sub_df = df_assign[df_assign.sub_id == sub_id]
    if len(sub_df) == 0:
        return sub_df
    n = min(n_anchors, len(sub_df))
    return sub_df.sample(n=n, random_state=int(rng.integers(0, 2 ** 31 - 1)))


def propose_targeted(df_assign: pd.DataFrame, sparse_subs: list[dict],
                     deficits: dict[int, int], n_propose_total: int,
                     perturb_scale: float = 0.15,
                     rng: np.random.Generator = None) -> pd.DataFrame:
    """For each sparse sub, propose ~N_per_sub perturbations of its existing
    members. The total is `n_propose_total`, allocated proportional to deficit.
    """
    rng = rng if rng is not None else np.random.default_rng(7)
    if not sparse_subs:
        return pd.DataFrame()
    total_def = sum(deficits.values())
    cands = []
    for s in sparse_subs:
        sub_id = s["sub_id"]
        share = max(2, int(round(n_propose_total * deficits[sub_id] / max(total_def, 1))))
        anchors = per_sub_anchors(df_assign, sub_id, share, rng)
        if len(anchors) == 0:
            continue
        anc_n = anchors["n"].to_numpy()
        anc_eta = np.log(np.clip(anchors["eta"].to_numpy(), 1e-12, None))
        anc_sy = np.log(np.clip(anchors["sigma_y"].to_numpy(), 1e-12, None))
        z = np.column_stack([anc_n, anc_eta, anc_sy])
        # use sub's z_scale (basin-wide std as floor)
        z_std = np.maximum(z.std(axis=0, ddof=1) if len(z) > 1
                            else np.array([0.05, 0.5, 0.5]),
                            np.array([0.035, 0.12, 0.12]))
        sigma = perturb_scale * z_std
        pert = rng.normal(0.0, 1.0, size=(share, 3)) * sigma[None, :]
        new_z = z + pert if len(z) == share else z[rng.integers(0, len(z), size=share)] + pert
        # Sample (W, H) from anchor's existing pairs
        wh_idx = rng.integers(0, len(anchors), size=share)
        Ws = anchors["width"].to_numpy()[wh_idx]
        Hs = anchors["height"].to_numpy()[wh_idx]
        for k in range(share):
            cands.append({
                "n": float(np.clip(new_z[k, 0], 0.30, 1.00)),
                "eta": float(np.clip(np.exp(new_z[k, 1]), 1e-3, 300.0)),
                "sigma_y": float(np.clip(np.exp(new_z[k, 2]), 1e-3, 400.0)),
                "width": float(Ws[k]),
                "height": float(Hs[k]),
                "target_sub": int(sub_id),
            })
    return pd.DataFrame(cands)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=PIPE / "Models" / "v10_yshape_v1")
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--n-sims", type=int, default=100,
                    help="Total sims to run in this batch")
    ap.add_argument("--target-N", type=int, default=300,
                    help="Bring every sub to at least this N")
    ap.add_argument("--perturb-scale", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Output CSV with original + new rows per sub")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="Append flush to out-csv every N sims so a crash mid-run "
                         "loses at most this many points.")
    a = ap.parse_args()

    sd = a.bank / f"state_gid_{a.gid}"
    with open(sd / "state.pkl", "rb") as f:
        state = pickle.load(f)
    df_assign = pd.read_csv(sd / "sub_assignments.csv")
    # Accumulate any previous infill CSVs so counts reflect what we already added
    keep_cols = list(df_assign.columns)
    for inf_csv in sorted(sd.glob("infill_*.csv")):
        d = pd.read_csv(inf_csv)
        d = d[[c for c in keep_cols if c in d.columns]]
        df_assign = pd.concat([df_assign, d], ignore_index=True)
        print(f"  + previous infill {inf_csv.name}: {len(d)} rows")
    print(f"\n=== gid {a.gid} ===")
    print(f"existing total: {len(df_assign)} points across {state['n_subs_total']} subs")

    counts = df_assign.groupby("sub_id").size().to_dict()
    sparse_subs = []
    deficits = {}
    for b in state["bins"]:
        for s in b["subs"]:
            sid = s["sub_id"]
            n = counts.get(sid, 0)
            if n < a.target_N:
                sparse_subs.append({"sub_id": sid, "bin_id": b["bin_id"],
                                     "shape_id": s["shape_id"], "N": n,
                                     "mu_z": s["mu_z"]})
                deficits[sid] = a.target_N - n
    print(f"sparse subs (N<{a.target_N}): {len(sparse_subs)}, total deficit={sum(deficits.values())}")
    if not sparse_subs:
        print("  (no sparse subs; exit)")
        return

    rng = np.random.default_rng(a.seed)
    cands = propose_targeted(df_assign, sparse_subs, deficits,
                              n_propose_total=a.n_sims,
                              perturb_scale=a.perturb_scale, rng=rng)
    print(f"proposed {len(cands)} candidates "
          f"(target subs: {cands.target_sub.value_counts().head(8).to_dict()})")

    print("\ninitialising MLS-MPM headless simulator (one-time)...")
    from DataPipeline.headless_mls import HeadlessSimulatorMLS  # lazy: pulls taichi
    sim = HeadlessSimulatorMLS(arch="cuda")

    if a.out_csv is None:
        a.out_csv = sd / f"infill_gid{a.gid}_{int(time.time())}.csv"
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    ckpt_interval = max(1, int(a.checkpoint_every))
    print(f"checkpointing every {ckpt_interval} sims to {a.out_csv}")

    print("\nrunning sims...")
    new_rows = []
    routed = {}
    target_hit = 0
    t0 = time.time()
    csv_header_written = False
    for i, row in cands.reset_index(drop=True).iterrows():
        ts = time.time()
        try:
            y8 = sim.run(row["n"], row["eta"], row["sigma_y"], row["width"], row["height"])
        except Exception as e:
            print(f"  [{i+1}/{len(cands)}] FAIL: {e}")
            continue
        bin_id, sub_id = route_y_to_sub(y8, state)
        if sub_id == int(row["target_sub"]):
            target_hit += 1
        routed[sub_id] = routed.get(sub_id, 0) + 1
        out = {
            "n": float(row["n"]), "eta": float(row["eta"]),
            "sigma_y": float(row["sigma_y"]),
            "width": float(row["width"]), "height": float(row["height"]),
            **{f"x_{j+1:02d}": float(y8[j]) for j in range(8)},
            "gid": int(a.gid),
            "length_bin": int(bin_id),
            "sub_id": int(sub_id),
            "target_sub": int(row["target_sub"]),
            "sim_seconds": time.time() - ts,
        }
        new_rows.append(out)
        # Append-checkpoint: every `ckpt_interval` sims flush to disk
        if (i + 1) % ckpt_interval == 0:
            df_chunk = pd.DataFrame(new_rows[-ckpt_interval:])
            df_chunk.to_csv(a.out_csv, mode="a" if csv_header_written else "w",
                              header=not csv_header_written, index=False)
            csv_header_written = True
            print(f"  [{i+1}/{len(cands)}] target_hit={target_hit}/{i+1}  "
                  f"({100*target_hit/(i+1):.0f}%)  elapsed={time.time()-t0:.0f}s  "
                  f"[checkpoint]")
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(cands)}] target_hit={target_hit}/{i+1}  "
                  f"({100*target_hit/(i+1):.0f}%)  elapsed={time.time()-t0:.0f}s")
    # final flush of any remainder past the last checkpoint
    flushed = (len(new_rows) // ckpt_interval) * ckpt_interval
    if flushed < len(new_rows):
        df_tail = pd.DataFrame(new_rows[flushed:])
        df_tail.to_csv(a.out_csv, mode="a" if csv_header_written else "w",
                          header=not csv_header_written, index=False)
        csv_header_written = True

    elapsed = time.time() - t0
    print(f"\nsim batch done in {elapsed:.0f}s ({elapsed/max(len(new_rows),1):.1f}s/sim)")
    print(f"target hits (sub matches proposal): {target_hit}/{len(new_rows)}  "
          f"({100*target_hit/max(len(new_rows),1):.0f}%)")
    print(f"\nrouted distribution (sub_id: count):")
    df_new = pd.DataFrame(new_rows)
    counter = df_new.sub_id.value_counts().sort_index()
    for sid, c in counter.items():
        was = counts.get(sid, 0)
        flag = " (was sparse)" if was < a.target_N else ""
        print(f"  sub {sid:>3d}: was N={was:>4d}, +{c:>3d} new, now N={was+c:>4d}{flag}")

    print(f"\nwrote {len(new_rows)} new rows to {a.out_csv} (checkpointed every {ckpt_interval})")


if __name__ == "__main__":
    main()
