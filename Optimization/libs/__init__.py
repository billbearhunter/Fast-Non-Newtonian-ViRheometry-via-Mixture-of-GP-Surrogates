"""Internal helpers for Optimization/setup1.py and setup2.py.

Mirrors the prev-work `libs/` layout:
    engine.py     — load_runtime (rBCM predictor + scalers + geo router)
    selector.py   — Plan B σY two-stage prior + CMA-ES fitness
    mechanism.py  — analytical Hessian + orthogonality-based setup proposer
"""
