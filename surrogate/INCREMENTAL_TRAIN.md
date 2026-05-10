# Incremental Training Notes

The old `vi_mogp.hierarchical.*` training path has been removed from the live
pipeline. Current surrogate work lives under `surrogate/` and the active
production bank is:

```text
Models/yshape_mogp_production/state_gid_{0..24}/
```

For the current y-shape bank, use these scripts instead of the old
`vi_mogp` commands:

```bash
python scripts/partition_flat_kmeans.py
python surrogate/train_yshape_subs.py
python scripts/calibrate_subs_holdout.py --bank Models/yshape_mogp_production --gids 0
```

Operational inverse entry points:

```bash
python -m Optimization.estimate_first_setup -f <ref_dir>
python -m Optimization.estimate_joint_setup -f <ref_dir1> -s <ref_dir2>
```

`Optimization.libs.engine` installs in-memory aliases for legacy pickles that
may still refer to `vi_mogp.*`, so the repository no longer needs a top-level
`vi_mogp/` package.
