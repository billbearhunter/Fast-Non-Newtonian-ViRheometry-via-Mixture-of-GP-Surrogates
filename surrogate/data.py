"""Data loading and preprocessing for VI-MoGP.

We pull all 220k rows from the existing labeled splits (train+val+test),
treating them as one dataset. The existing `cluster_id` column is ignored
— that's from the old MoE's GMM, we are re-clustering from scratch.

Held-out split: a fresh 10% of the 220k, stratified by (W, H) grid, for
final evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from surrogate.config import (
    INPUT_COLS, OUTPUT_COLS, LOG_INPUTS,
    DEVICE, DTYPE, SEED, LOG_EPS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Raw → tensor pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _log_transform_inputs(X: np.ndarray) -> np.ndarray:
    """Apply log to columns in LOG_INPUTS. Input is (N, 5) in INPUT_COLS order."""
    X = X.astype(np.float64).copy()
    for i, col in enumerate(INPUT_COLS):
        if col in LOG_INPUTS:
            X[:, i] = np.log(np.maximum(X[:, i], LOG_EPS))
    return X


@dataclass
class InputScaler:
    """Log-then-standardise, parametrised by mean/std on log-transformed space."""
    mean: np.ndarray   # (5,)
    std:  np.ndarray   # (5,)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return (_log_transform_inputs(X_raw) - self.mean) / self.std

    def inverse_transform(self, X_s: np.ndarray) -> np.ndarray:
        X_log = X_s * self.std + self.mean
        X_raw = X_log.copy()
        for i, col in enumerate(INPUT_COLS):
            if col in LOG_INPUTS:
                X_raw[:, i] = np.exp(X_log[:, i])
        return X_raw

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "log_cols": LOG_INPUTS, "input_cols": INPUT_COLS}

    @classmethod
    def fit(cls, X_raw: np.ndarray) -> "InputScaler":
        X_log = _log_transform_inputs(X_raw)
        return cls(mean=X_log.mean(0), std=X_log.std(0) + 1e-12)

    @classmethod
    def from_dict(cls, d: dict) -> "InputScaler":
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))


@dataclass
class OutputScaler:
    """Centre y (subtract per-dim mean). No std rescaling to preserve monotonicity."""
    mean: np.ndarray   # (8,)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return Y - self.mean

    def inverse_transform(self, Y_s: np.ndarray) -> np.ndarray:
        return Y_s + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "output_cols": OUTPUT_COLS}

    @classmethod
    def fit(cls, Y: np.ndarray) -> "OutputScaler":
        return cls(mean=Y.mean(0))

    @classmethod
    def from_dict(cls, d: dict) -> "OutputScaler":
        return cls(mean=np.asarray(d["mean"]))


# ──────────────────────────────────────────────────────────────────────────────
# Load pipeline
# ──────────────────────────────────────────────────────────────────────────────

def load_all_labeled(ws_dir: Path) -> pd.DataFrame:
    """Concat train+val+test labeled csvs from the existing MoE workspace.

    The `cluster_id` and `cluster_conf` columns (if present) are DROPPED —
    this dataset is about to be re-clustered from scratch.
    """
    frames = []
    for split in ("train", "val", "test"):
        p = ws_dir / f"{split}_labeled.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        df = pd.read_csv(p)
        df["_split_orig"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    drop_cols = [c for c in ("cluster_id", "cluster_conf") if c in full.columns]
    full = full.drop(columns=drop_cols)
    return full


def stratified_holdout(df: pd.DataFrame, frac: float = 0.10,
                       seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by (W, H) geometry bin.

    Ensures held-out has same (W, H) coverage as training.
    """
    rng = np.random.RandomState(seed)
    # bin by 1 cm grid
    df = df.reset_index(drop=True).copy()
    df["_cell"] = (df["width"].round(0).astype(int).astype(str) + "_" +
                   df["height"].round(0).astype(int).astype(str))
    hold_idx = []
    for cell, grp in df.groupby("_cell"):
        n_hold = max(1, int(round(len(grp) * frac)))
        hold_idx.extend(rng.choice(grp.index.values, size=n_hold, replace=False))
    hold = df.loc[hold_idx].drop(columns="_cell").reset_index(drop=True)
    keep = df.drop(index=hold_idx).drop(columns="_cell").reset_index(drop=True)
    return keep, hold


def make_tensors(df: pd.DataFrame,
                 xs: InputScaler | None = None,
                 ys: OutputScaler | None = None,
                 ) -> tuple[torch.Tensor, torch.Tensor, InputScaler, OutputScaler]:
    """df → (X, Y) tensors on DEVICE. Fit scalers on df if not provided."""
    X_raw = df[INPUT_COLS].values.astype(np.float64)
    Y     = df[OUTPUT_COLS].values.astype(np.float64)
    if xs is None:
        xs = InputScaler.fit(X_raw)
    if ys is None:
        ys = OutputScaler.fit(Y)
    X_s = xs.transform(X_raw)
    Y_s = ys.transform(Y)
    X_t = torch.tensor(X_s, dtype=DTYPE, device=DEVICE)
    Y_t = torch.tensor(Y_s, dtype=DTYPE, device=DEVICE)
    return X_t, Y_t, xs, ys


# ──────────────────────────────────────────────────────────────────────────────
# Summary (sanity check)
# ──────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> None:
    print(f"  rows: {len(df):,}")
    print(f"  (W, H) unique cells: {df.groupby(['width','height']).ngroups}")
    for c in INPUT_COLS:
        v = df[c].values
        print(f"  {c:>8}: min={v.min():.4g}  max={v.max():.4g}  mean={v.mean():.4g}")
    for c in OUTPUT_COLS:
        v = df[c].values
        print(f"  {c:>8}: min={v.min():.4g}  max={v.max():.4g}  mean={v.mean():.4g}")


if __name__ == "__main__":
    # smoke test
    print("Loading labeled data...")
    full = load_all_labeled()
    print(f"Full dataset:")
    summarize(full)
    print("\nStratified hold-out (10 %)...")
    keep, hold = stratified_holdout(full, frac=0.10)
    print(f"Train pool: {len(keep):,}  /  Hold-out: {len(hold):,}")
    X_t, Y_t, xs, ys = make_tensors(keep)
    print(f"\nX tensor: {tuple(X_t.shape)}  dtype={X_t.dtype}  device={X_t.device}")
    print(f"Y tensor: {tuple(Y_t.shape)}  dtype={Y_t.dtype}  device={Y_t.device}")
    print(f"xs.mean: {xs.mean}")
    print(f"xs.std:  {xs.std}")
    print(f"ys.mean: {ys.mean}")
