"""Reusable HeadlessSimulator wrapper around AGTaichiMPM2 (MLS-MPM).

API parity with DataPipeline.collect_data.HeadlessSimulator so it's a
drop-in replacement — same `run(n, eta, sigma_y, W, H) -> (8,) np.ndarray`
interface. Key difference from the naive approach:

  * Particle fields are ALLOCATED ONCE at MAX_W × MAX_H × Z_MAX size.
  * Each `run()` updates HB params, static-box walls, and the active
    particle count — no re-instantiation, no Taichi field churn.

This preserves MLS-MPM's per-step speedup (~1.5× at dt=1.5e-4 vs uGIMP
AGTaichi at dt=7.5e-5) across thousands of sims in one process.
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import taichi as ti

_ROOT = Path(__file__).resolve().parents[1]
_SIM_DIR = _ROOT / "Simulation" / "simulation"

# Import AGTaichiMPM2 by absolute path (avoid shadowing the installed
# `taichi` module by Simulation/simulation/taichi.py in the same dir).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "AGTaichiMPM2_mod", str(_SIM_DIR / "AGTaichiMPM2.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AGTaichiMPM2 = _mod.AGTaichiMPM2


def _ensure_taichi(arch: str = "cuda"):
    """Call ti.init() once per process. Safe if already initialised."""
    if getattr(_ensure_taichi, "_done", False):
        return
    a = ti.cuda if arch == "cuda" else ti.cpu
    try:
        ti.init(default_fp=ti.f32, arch=a)
    except Exception as e:
        print(f"[headless_mls] Taichi init on {arch} failed ({e}); falling back CPU")
        ti.init(default_fp=ti.f32, arch=ti.cpu)
    _ensure_taichi._done = True


class HeadlessSimulatorMLS:
    """Drop-in replacement for DataPipeline.collect_data.HeadlessSimulator.

    Parameters
    ----------
    max_W, max_H : float
        Maximum container width/height (cm). Particle arrays are pre-allocated
        for the full (max_W+pad, max_H+pad, Z+pad) cuboid.
    Z : float
        Cuboid depth (cm), same semantics as AGTaichiMPM2 --Z.
    dt : float
        Timestep. MLS-MPM is stable at larger dt than uGIMP (e.g. 1.5e-4
        vs 7.5e-5). Default 1.5e-4 keeps ≤ 0.003 relative error vs uGIMP
        baseline on the three validation materials.
    cell_width, cell_samples_per_dim : float, int
        Grid spacing / particles per cell (match Simulation/config/setting.xml
        defaults: dx=0.126, samples=2).
    rho : float
    arch : str
        "cuda" or "cpu". Passed once to ti.init().
    """

    def __init__(self,
                 max_W: float = 7.0,
                 max_H: float = 7.0,
                 cuboid_Z: float = 4.15,   # <── matches setting.xml cuboid max z
                 back_wall_z_min: float = 4.0,
                 back_wall_z_max: float = 4.3,
                 dt: float = 1.5e-4,
                 cell_width: float = 0.126,
                 cell_samples_per_dim: int = 2,
                 rho: float = 1.2,          # <── matches setting.xml cuboid density
                 bulk_modulus: float = 1.0e5,
                 shear_modulus: float = 1.0e4,
                 grid_min=(-1.0, -1.0, -10.0),
                 grid_max=(30.0, 8.0, 14.0),
                 fps: int = 24,
                 max_frames: int = 8,
                 arch: str = "cuda"):
        Z = cuboid_Z  # alias for backwards-compat w/ AGTaichiMPM2 --Z
        _ensure_taichi(arch)

        # ---- build the internal MLS-MPM instance at MAX size ------------
        self._max_W, self._max_H = float(max_W), float(max_H)
        self._cuboid_Z = float(cuboid_Z)          # particle fill depth (4.15)
        self._back_wall_z_min = float(back_wall_z_min)   # box-3 start (4.0)
        self._back_wall_z_max = float(back_wall_z_max)   # box-3 end   (4.3)
        self._Z = self._cuboid_Z  # kept for backwards compat inside _set_walls
        self._dt = float(dt)
        self._cell_width = float(cell_width)
        self._cell_samples = int(cell_samples_per_dim)
        self._rho = float(rho)
        self._fps = int(fps)
        self._max_frames = int(max_frames)

        # Initial sentinel material (will be overwritten per run)
        args = SimpleNamespace(
            eta=10.0, n=0.5, sigmaY=1.0,
            H=max_H, W=max_W, Z=Z,
            rho=rho,
            max_frames=max_frames, fps=fps, dt=dt,
            bulk_modulus=bulk_modulus, shear_modulus=shear_modulus,
            cell_width=cell_width, cell_samples_per_dim=cell_samples_per_dim,
            grid_min=list(grid_min), grid_max=list(grid_max),
            taichi_arch=arch, save_dat=False,
            out="",
        )
        cuboid_min = (-0.15, -0.15, -0.15)
        cuboid_max = (max_W, max_H, Z)
        self.mpm = AGTaichiMPM2(args, cuboid_min, cuboid_max)
        self._N_alloc = int(self.mpm.ti_particle_count[None])
        print(f"[HeadlessSimulatorMLS] allocated {self._N_alloc} particles "
              f"for max cuboid ({max_W:.1f}×{max_H:.1f}×{Z:.1f}) cm")

        # ---- ensure py_num_saved_frames field exists for API parity -----
        if not hasattr(self.mpm, "py_num_saved_frames"):
            self.mpm.py_num_saved_frames = 0

        # ---- JIT warm-up ------------------------------------------------
        self.mpm.initialize()
        t0 = time.perf_counter()
        self.mpm.step()
        ti.sync()
        self.mpm.initialize()
        print(f"[HeadlessSimulatorMLS] JIT warm-up: {time.perf_counter()-t0:.2f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Runtime reconfiguration — mirrors AGTaichiMPM.changeSetUpData
    # ═══════════════════════════════════════════════════════════════════
    def _set_material(self, n: float, eta: float, sigma_y: float):
        self.mpm.ti_hb_n[None]      = float(n)
        self.mpm.ti_hb_eta[None]    = float(eta)
        self.mpm.ti_hb_sigmaY[None] = float(sigma_y)

    def _set_walls(self, W: float, H: float):
        """Update all 4 wall boxes to match setting.xml exactly."""
        cz = self._cuboid_Z
        # Box 0: floor (unchanged per sim, but re-set for safety)
        self.mpm.ti_static_box_min[0] = ti.Vector([-100.0, -1.0, -100.0])
        self.mpm.ti_static_box_max[0] = ti.Vector([100.0, 0.0, 100.0])
        # Box 1: left wall — note max-z = cuboid_Z (= 4.15), not 4.3
        self.mpm.ti_static_box_min[1] = ti.Vector([-1.0, 0.0, 0.0])
        self.mpm.ti_static_box_max[1] = ti.Vector([0.0, 20.0, cz])
        # Box 2: front depth wall (z<0), max.x depends on W
        self.mpm.ti_static_box_min[2] = ti.Vector([-1.0, 0.0, -0.3])
        self.mpm.ti_static_box_max[2] = ti.Vector([float(W), 20.0, 0.0])
        # Box 3: back depth wall — XML uses [4.0, 4.3], overlaps cuboid rear 0.15
        self.mpm.ti_static_box_min[3] = ti.Vector([-1.0, 0.0, self._back_wall_z_min])
        self.mpm.ti_static_box_max[3] = ti.Vector([float(W), 20.0, self._back_wall_z_max])

    def _set_cuboid(self, W: float, H: float) -> int:
        """Recompute particle ndcount + active N for the new cuboid."""
        cuboid_min = np.array([-0.15, -0.15, -0.15], dtype=np.float32)
        cuboid_max = np.array([float(W), float(H), self._cuboid_Z], dtype=np.float32)
        width = cuboid_max - cuboid_min
        nd = np.ceil(width * self._cell_samples / self._cell_width).astype(np.int32)
        N_active = int(np.prod(nd))
        if N_active > self._N_alloc:
            raise RuntimeError(
                f"N_active={N_active} exceeds pre-allocated {self._N_alloc}; "
                f"construct HeadlessSimulatorMLS with larger max_W/max_H."
            )
        # Update Taichi fields
        self.mpm.ti_particle_init_min.from_numpy(cuboid_min.reshape(1, 3))
        self.mpm.ti_particle_ndcount.from_numpy(nd)
        self.mpm.ti_particle_count[None] = N_active
        return N_active

    # ═══════════════════════════════════════════════════════════════════
    # Public API: matches DataPipeline.collect_data.HeadlessSimulator.run()
    # ═══════════════════════════════════════════════════════════════════
    def run(self, n: float, eta: float, sigma_y: float,
            width: float, height: float) -> np.ndarray:
        """Run one dam-break sim; return 8 flow-distance differences [cm]."""
        self._set_material(n, eta, sigma_y)
        self._set_walls(width, height)
        self._set_cuboid(width, height)
        self.mpm.initialize()
        self.mpm.py_num_saved_frames = 0

        # frame-capture loop — mirrors original HeadlessSimulator.run()
        x0: Optional[float] = None
        diffs: list = []
        while True:
            for _ in range(100):
                self.mpm.step()
                t = self.mpm.ti_iteration[None] * self.mpm.py_dt
                if t * self.mpm.py_fps >= self.mpm.py_num_saved_frames:
                    frame = self.mpm.py_num_saved_frames
                    max_x = self._max_x()
                    if frame == 0:
                        x0 = max_x
                    elif x0 is not None and 1 <= frame <= 8:
                        diffs.append(max_x - x0)
                    self.mpm.py_num_saved_frames += 1

            if self.mpm.py_num_saved_frames > self.mpm.py_max_frames:
                break

        if len(diffs) < 8:
            diffs += [0.0] * (8 - len(diffs))
        return np.array(diffs[:8], dtype=np.float32)

    def _max_x(self) -> float:
        N = int(self.mpm.ti_particle_count[None])
        p = self.mpm.ti_particle_x.to_numpy()[:N]
        return float(p[:, 0].max())


# -----------------------------------------------------------------------
# Simple smoke / self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=0.6)
    ap.add_argument("--eta", type=float, default=50.0)
    ap.add_argument("--sy", type=float, default=50.0)
    ap.add_argument("--W", type=float, default=4.0)
    ap.add_argument("--H", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=1.5e-4)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    sim = HeadlessSimulatorMLS(dt=args.dt)
    for k in range(args.repeat):
        t0 = time.perf_counter()
        y = sim.run(args.n, args.eta, args.sy, args.W, args.H)
        print(f"[run {k}] y = {np.array2string(y, precision=4)}  "
              f"({time.perf_counter()-t0:.2f}s)")
