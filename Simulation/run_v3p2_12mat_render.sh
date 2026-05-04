#!/bin/bash
# Sequential sim+render for 12 materials, using θ̂ from v3p2 double mode
set -e
cd "$(dirname "$0")"
mkdir -p results/v3p2_12mat

python <<'PYEOF'
import json, subprocess, sys, time
from pathlib import Path

with open('v3p2_12mat_render_plan.json') as f:
    plan = json.load(f)

# Filter out .zip
plan = [p for p in plan if '.zip' not in p['ref']]
print(f'Running {len(plan)} sim+render tasks')
t0 = time.time()
for i, p in enumerate(plan, 1):
    out_dir = f"results/v3p2_12mat/{Path(p['ref']).name}"
    if Path(f'{out_dir}/snapdiff_08.png').exists():
        print(f'[{i}/{len(plan)}] {Path(p["ref"]).name}: SKIP (already rendered)')
        continue
    t = time.time()
    cmd = ['python', 'main.py',
           '--n', str(round(p['n'], 4)),
           '--eta', str(round(p['eta'], 3)),
           '--sigma_y', str(round(p['sy'], 3)),
           '--ref', f'../{p["ref"]}',
           '--out_dir', out_dir]
    print(f'[{i}/{len(plan)}] {Path(p["ref"]).name}: n={p["n"]:.2f} η={p["eta"]:.1f} σY={p["sy"]:.1f}')
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    if res.returncode != 0:
        print(f'  FAILED ({dt:.0f}s)')
        print(res.stderr[-500:])
    else:
        print(f'  done ({dt:.0f}s)')
print(f'\nTotal: {time.time()-t0:.0f}s')
PYEOF
