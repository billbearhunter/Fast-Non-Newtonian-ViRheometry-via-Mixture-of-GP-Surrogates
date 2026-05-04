"""Benchmark GP accuracy vs training set size (max_n).

Pick a few subs with N >= 1000 from v3p2 bank, hold out 200 val points,
then train GP at multiple subsample sizes (100, 400, 500, 600, 800, 1000)
and measure forward prediction error (RMSE, MAE, maxerror) on val.

Usage:
  python scripts/bench_max_n_accuracy.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

PIPE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs.engine import load_runtime
from surrogate.experts import ExactExpert
from surrogate import config as HC

device, dtype = HC.DEVICE, HC.DTYPE
geo, xs, ys = load_runtime()

BANK = PIPE / 'Models' / 'v10_yshape_v3p2_round2partial'

# Pick 4 subs with N>=1000 from different gids/bins for robust avg
TARGETS = [
    (0, 10),   # gid 0 bin 1 sub_id 7 N=2023 (large)
    (12, 7),   # gid 12 bin 1 sub_id 5/7 N=1622/1594
    (21, 8),   # gid 21 bin 1 sub_id 6/8 N=1694/1279
    (15, 11),  # gid 15
]

SAMPLE_SIZES = [100, 400, 500, 600, 800, 1000]
N_ITERS = 120
LR = 0.05
SEED = 42

def load_full_sub_data(gid, sub_id):
    """Load full sub training data (X_phys, Y_phys) before max-n cap."""
    sd = BANK / f'state_gid_{gid}'
    sa = pd.read_csv(sd / 'sub_assignments.csv')
    sub_rows = sa[sa.sub_id == sub_id]
    if len(sub_rows) < 1100:
        return None  # skip small subs
    X_cols = ['n', 'eta', 'sigma_y', 'width', 'height']
    Y_cols = [f'x_0{i}' for i in range(1, 9)]
    X = sub_rows[X_cols].to_numpy(dtype=np.float64)
    Y = sub_rows[Y_cols].to_numpy(dtype=np.float64)
    return X, Y, len(sub_rows)

def train_and_eval(X_train, Y_train, X_val, Y_val, n_train_subset, seed):
    """Train GP with subset of size n_train_subset; eval on val."""
    rng = np.random.default_rng(seed)
    if n_train_subset < len(X_train):
        idx = rng.choice(len(X_train), size=n_train_subset, replace=False)
        Xs = X_train[idx]; Ys = Y_train[idx]
    else:
        Xs = X_train; Ys = Y_train
    Xs_t = torch.tensor(xs.transform(Xs), dtype=dtype, device=device)
    Ys_t = torch.tensor(ys.transform(Ys), dtype=dtype, device=device)
    exp = ExactExpert(Xs_t, Ys_t, kernel_name='matern25_ard').to(device)
    exp.set_train_data(Xs_t, Ys_t)
    t0 = time.time()
    final_loss = exp.fit(n_iters=N_ITERS, lr=LR, verbose=False)
    fit_time = time.time() - t0
    exp.eval()
    Xv_t = torch.tensor(xs.transform(X_val), dtype=dtype, device=device)
    with torch.no_grad():
        mu_s, _ = exp.predict(Xv_t)
        Y_pred = mu_s.cpu().numpy() + ys.mean
    err = Y_pred - Y_val  # (N_val, 8)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    maxerr = float(np.max(np.abs(err)))
    return {'rmse': rmse, 'mae': mae, 'maxerr': maxerr,
            'fit_time': fit_time, 'final_loss': float(final_loss),
            'n_train': n_train_subset}

def main():
    rng = np.random.default_rng(SEED)
    print('=== max_n vs accuracy benchmark ===')
    print(f'Method: pick 200 random val points, train on subsets')
    print(f'Sample sizes: {SAMPLE_SIZES}')
    print(f'GP: matern25_ard, n_iters={N_ITERS}, lr={LR}')
    print()

    results = []
    for gid, sub_id in TARGETS:
        loaded = load_full_sub_data(gid, sub_id)
        if loaded is None:
            print(f'[skip] gid {gid} sub {sub_id}: N<1100')
            continue
        X, Y, N = loaded
        # Hold out 200 val points
        idx_all = rng.permutation(N)
        idx_val = idx_all[:200]
        idx_train = idx_all[200:]
        X_val, Y_val = X[idx_val], Y[idx_val]
        X_train, Y_train = X[idx_train], Y[idx_train]
        print(f'\\n>>> gid {gid} sub {sub_id}: N={N}, train pool {len(X_train)}, val 200')
        for n_sub in SAMPLE_SIZES:
            if n_sub > len(X_train):
                continue
            r = train_and_eval(X_train, Y_train, X_val, Y_val, n_sub, seed=SEED+gid*100+sub_id)
            r['gid'] = gid; r['sub_id'] = sub_id
            results.append(r)
            print(f'  N={n_sub:>4}  RMSE={r["rmse"]:.4f}  MAE={r["mae"]:.4f}  maxerr={r["maxerr"]:.4f}  fit={r["fit_time"]:.1f}s  loss={r["final_loss"]:+.3f}')

    df = pd.DataFrame(results)
    if len(df) == 0:
        print('No results — no qualifying subs found.')
        return
    out_csv = PIPE / 'OptimizationResults' / 'bench_max_n_accuracy.csv'
    df.to_csv(out_csv, index=False)
    print(f'\\nSaved: {out_csv}')

    # Aggregate by N (avg across subs)
    print()
    print('=== Average across subs ===')
    print(f'{"N_train":>8}{"RMSE":>9}{"MAE":>9}{"maxerr":>9}{"fit_time":>11}')
    print('-'*46)
    for n_sub in SAMPLE_SIZES:
        d = df[df.n_train == n_sub]
        if len(d) == 0: continue
        print(f'{n_sub:>8}{d.rmse.mean():>9.4f}{d.mae.mean():>9.4f}{d.maxerr.mean():>9.4f}{d.fit_time.mean():>11.2f}')

if __name__ == '__main__':
    main()
