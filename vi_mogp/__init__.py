"""Legacy compatibility shim — redirects ``vi_mogp.X`` to ``surrogate.X``.

After the 2026-05 surrogate/ ↔ vi_mogp/ consolidation, the GP runtime lives
under ``surrogate/``. Some pickled artefacts (geo_router, GMM checkpoints,
training-time state dumps) still reference the old ``vi_mogp.*`` module
paths. This shim makes those pickles deserialize without modification —
new code should import from ``surrogate.*`` directly.
"""
import importlib
import sys

_ALIASES = (
    "config", "data", "experts", "gp_base", "grid_geo", "model", "predict",
)

for _name in _ALIASES:
    sys.modules[f"vi_mogp.{_name}"] = importlib.import_module(f"surrogate.{_name}")
