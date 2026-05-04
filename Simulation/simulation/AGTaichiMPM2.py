"""
AGTaichiMPM2.py
---------------
MLS-MPM (Hu et al., SIGGRAPH 2018) variant of the project's MPM simulator.

Relationship to the other MPM files:
  - libs/3D/MPM3d/AGTaichiMPM.py  (uGIMP, original, coupled with CMA-ES)
  - 7_gamma_dot_viz/mpm_gdot_runner.py  (uGIMP, standalone γ̇ dumper)
  - THIS FILE                           (MLS-MPM, standalone γ̇ dumper)

The three share the **same Herschel-Bulkley constitutive model**
(`scalar_hb_solve_3d` Newton iteration on the same equation), the same
static-box wall BCs, and the same cuboid/grid layout used by
`run_taichi_3d.sh`.

Only the particle↔grid transfer and the velocity-gradient reconstruction
are different:
  uGIMP : tensor-product linear integral weights + weighted grad sum
  MLS   : quadratic B-spline weights + affine velocity field (APIC) +
          velocity gradient reconstructed algebraically from the grid
          velocities via `C_new = 4/dx^2 * Σ w · v_grid ⊗ dpos`.

The CLI is identical to `mpm_gdot_runner.py` so that the two can be
benchmarked on the same setup by `7_gamma_dot_viz/benchmark_ugimp_vs_mls.py`.
Outputs per run:
  out/<dir>/config_XX_gamma_dot.bin   float32, N particles, γ̇ in 1/s
  out/<dir>/config_XX_pos.bin         float32, 3·N (x,y,z) for each particle
  out/<dir>/meta.txt                  parameters + dt/fps/impl
  out/<dir>/timing.json               {mpm_time_sec, time_per_step_ms, ...}

Usage example:
  python libs/3D/MPM3d/AGTaichiMPM2.py \\
      --eta 4.207 --n 0.983 --sigmaY 79.36 --H 4.5 --W 4.0 \\
      --out 7_gamma_dot_viz/out/Tonkatsu_2_mls
"""

import argparse
import ctypes
import json
import os
import time
import numpy as np
import taichi as ti

real = ti.f32
realnp = np.float32
G_CGS = 981.0  # cm/s^2


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, required=True,
                   help="HB consistency (CGS: dyn·s^n/cm²)")
    p.add_argument("--n", type=float, required=True)
    p.add_argument("--sigmaY", type=float, required=True,
                   help="HB yield stress (CGS: dyn/cm²)")
    p.add_argument("--H", type=float, required=True, help="reservoir H [cm]")
    p.add_argument("--W", type=float, required=True, help="reservoir W [cm]")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dt", type=float, default=0.000075,
                   help="time step; MLS-MPM can be pushed larger than uGIMP")
    p.add_argument("--bulk-modulus", type=float, default=1.0e5)
    p.add_argument("--shear-modulus", type=float, default=1.0e4)
    p.add_argument("--cell-width", type=float, default=0.126)
    p.add_argument("--cell-samples-per-dim", type=int, default=2)
    p.add_argument("--grid-min", type=float, nargs=3, default=[-1.0, -1.0, -10.0])
    p.add_argument("--grid-max", type=float, nargs=3, default=[30.0, 8.0, 14.0])
    p.add_argument("--Z", type=float, default=4.3,
                   help="cuboid depth (matches run_taichi_3d.sh)")
    p.add_argument("--taichi-arch", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--save-dat", action="store_true",
                   help="also save config_XX.dat in pipeline format for "
                        "ParticleSkinner + GLRender3d post-processing")
    return p.parse_args()


# ===========================================================================
# MLS-MPM kernel class
# ===========================================================================

@ti.data_oriented
class AGTaichiMPM2:
    def __init__(self, args, cuboid_min, cuboid_max):
        # -------- HB constitutive constants --------
        self.ti_hb_n      = ti.field(dtype=real, shape=())
        self.ti_hb_eta    = ti.field(dtype=real, shape=())
        self.ti_hb_sigmaY = ti.field(dtype=real, shape=())
        self.ti_hb_n[None]     = args.n
        self.ti_hb_eta[None]   = args.eta
        self.ti_hb_sigmaY[None] = args.sigmaY

        # -------- walls --------
        self.ti_num_boxes = ti.field(dtype=int, shape=())
        self.ti_num_boxes[None] = 4
        self.ti_static_box_min = ti.Vector.field(3, dtype=real, shape=4)
        self.ti_static_box_max = ti.Vector.field(3, dtype=real, shape=4)
        H, W, Z = args.H, args.W, args.Z
        boxes = [
            ((-100.0, -1.0, -100.0), (100.0, 0.0, 100.0)),
            ((-1.0, 0.0, 0.0),        (0.0, 20.0, Z)),
            ((-1.0, 0.0, -0.3),       (W,   20.0, 0.0)),
            ((-1.0, 0.0, Z),          (W,   20.0, Z + 0.3)),
        ]
        for i, (lo, hi) in enumerate(boxes):
            self.ti_static_box_min[i] = ti.Vector(lo)
            self.ti_static_box_max[i] = ti.Vector(hi)

        # -------- material / numerical constants --------
        self.py_kappa = args.bulk_modulus
        self.py_mu    = args.shear_modulus
        self.py_dt    = args.dt
        self.py_dx    = args.cell_width
        self.py_invdx = 1.0 / self.py_dx
        self.py_fps   = args.fps
        self.py_max_frames = args.max_frames
        self.ti_g = ti.Vector([0.0, -G_CGS, 0.0])
        self.ti_iteration = ti.field(dtype=int, shape=())
        self.ti_iteration[None] = 0

        # -------- grid --------
        gmin = np.array(args.grid_min, dtype=realnp)
        gmax = np.array(args.grid_max, dtype=realnp)
        center = (gmax + gmin) * 0.5
        width  = gmax - gmin
        self.py_cell_count = np.ceil(width / self.py_dx).astype(int)
        width = self.py_cell_count.astype(realnp) * self.py_dx
        self.ti_grid_min = ti.Vector(center - 0.5 * width)
        self.ti_grid_m = ti.field(dtype=real, shape=self.py_cell_count)
        self.ti_grid_x = ti.Vector.field(3, dtype=real, shape=self.py_cell_count)
        self.ti_grid_v = ti.Vector.field(3, dtype=real, shape=self.py_cell_count)

        # -------- cuboid / particles --------
        cmin = np.array(cuboid_min, dtype=realnp)
        cmax = np.array(cuboid_max, dtype=realnp)
        self.ti_particle_init_min = ti.Vector.field(3, dtype=real, shape=1)
        self.ti_particle_init_min.from_numpy(cmin.reshape(1, 3))
        self.py_particle_init_cell_samples_per_dim = args.cell_samples_per_dim
        self.ti_particle_init_vel = ti.Vector([0.0, 0.0, 0.0])
        self.py_particle_hl = 0.5 * self.py_dx / args.cell_samples_per_dim
        self.py_particle_volume = (self.py_dx / args.cell_samples_per_dim) ** 3
        self.py_particle_mass = args.rho * self.py_particle_volume

        cuboid_width = cmax - cmin
        nd = np.ceil(cuboid_width * args.cell_samples_per_dim
                     / self.py_dx).astype(np.int32)
        self.ti_particle_ndcount = ti.field(dtype=int, shape=3)
        self.ti_particle_ndcount.from_numpy(nd)
        N = int(np.prod(nd))
        print(f"[AGTaichiMPM2] particle count = {N}")

        self.ti_particle_count = ti.field(dtype=int, shape=())
        self.ti_particle_count[None] = N
        self.ti_particle_is_inner_of_box = ti.field(int, shape=N)
        self.ti_particle_x = ti.Vector.field(3, dtype=real, shape=N)
        self.ti_particle_v = ti.Vector.field(3, dtype=real, shape=N)
        self.ti_particle_be = ti.Matrix.field(3, 3, dtype=real, shape=N)
        self.ti_particle_C  = ti.Matrix.field(3, 3, dtype=real, shape=N)
        self.ti_particle_gamma_dot = ti.field(dtype=real, shape=N)

        # grid cell positions (for wall BC check)
        @ti.kernel
        def _fill_grid_pos():
            for I in ti.grouped(self.ti_grid_m):
                self.ti_grid_x[I] = self.ti_grid_min + I * self.py_dx
        _fill_grid_pos()

    # -------- helpers --------
    @staticmethod
    @ti.func
    def dev_3d(A):
        return A - A.trace() * ti.Matrix.identity(real, 3) / 3.0

    @staticmethod
    @ti.func
    def bar_3d(A):
        return A * ti.pow(A.determinant(), -1.0 / 3.0)

    # -------- HB Newton solver (identical to uGIMP version) --------
    @staticmethod
    @ti.func
    def hb_eval_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta,
                   trace_be_bar, dt):
        return (x - sigma_len_pre
                + ti.sqrt(2.0) * dt * mu_div_J * trace_be_bar
                  * ti.pow(((x / ti.sqrt(2.0)) - hb_sigma_y) / hb_eta, 1.0 / hb_n)
                  / 3.0)

    @staticmethod
    @ti.func
    def hb_eval_deriv_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta,
                         trace_be_bar, dt):
        return (1.0
                + dt * mu_div_J * trace_be_bar
                  * ti.pow(((x / ti.sqrt(2.0)) - hb_sigma_y) / hb_eta,
                           1.0 / hb_n - 1.0)
                  / (3.0 * hb_n * hb_eta))

    @ti.func
    def scalar_hb_solve_3d(self, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n,
                           hb_eta, trace_be_bar, dt):
        x = sigma_len_pre
        for _ in range(14):
            fx  = self.hb_eval_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y,
                                  hb_n, hb_eta, trace_be_bar, dt)
            dfx = self.hb_eval_deriv_3d(x, sigma_len_pre, mu_div_J,
                                        hb_sigma_y, hb_n, hb_eta,
                                        trace_be_bar, dt)
            dx = -fx / dfx
            for _j in range(20):
                x_new = x + dx
                if (x_new / ti.sqrt(2.0) - hb_sigma_y) >= 0:
                    x = x_new
                    break
                dx = dx / 2.0
            if ti.abs(dx) < 1.0e-6:
                break
        return x

    # -------- initialisation --------
    @ti.kernel
    def initialize(self):
        self.ti_iteration[None] = 0
        for I in ti.grouped(self.ti_grid_m):
            self.ti_grid_m[I] = 0.0
            self.ti_grid_v[I] = ti.Vector.zero(real, 3)

        for i in range(self.ti_particle_count[None]):
            pi = i % self.ti_particle_ndcount[0]
            pj = (i // self.ti_particle_ndcount[0]) % self.ti_particle_ndcount[1]
            pk = i // (self.ti_particle_ndcount[0] * self.ti_particle_ndcount[1])
            r = ti.Vector([0.5, 0.5, 0.5])
            _I = ti.Vector([pi, pj, pk]).cast(real) + r
            self.ti_particle_x[i] = (self.ti_particle_init_min[0]
                + (self.py_dx / self.py_particle_init_cell_samples_per_dim) * _I)
            self.ti_particle_v[i] = self.ti_particle_init_vel
            self.ti_particle_be[i] = ti.Matrix.identity(real, 3)
            self.ti_particle_C[i]  = ti.Matrix.zero(real, 3, 3)
            self.ti_particle_is_inner_of_box[i] = 0
            self.ti_particle_gamma_dot[i] = 0.0

    # ==========================================================
    # MLS-MPM step
    #
    # Conventions (quadratic B-spline, Hu et al. 2018):
    #   For a particle at x, we find the lower-left node of the 3×3×3
    #   stencil by  base = floor(x/dx - 0.5).
    #   fx = x/dx - base    ∈ [0.5, 1.5]^3
    #   Per-axis weights:
    #     w[0] = 0.5 (1.5 - fx)^2
    #     w[1] = 0.75 - (fx - 1)^2
    #     w[2] = 0.5 (fx - 0.5)^2
    #   D^{-1} = 4/dx^2   (inverse inertia tensor of the B-spline)
    #
    # The constitutive update is identical to the uGIMP version:
    #   L_p = C_new  (velocity gradient from affine field)
    #   f   = I + dt·L_p
    #   then HB plastic correction via scalar_hb_solve_3d.
    # ==========================================================

    @ti.kernel
    def step(self):
        self.ti_iteration[None] += 1

        # ---- reset grid ----
        for I in ti.grouped(self.ti_grid_m):
            self.ti_grid_m[I] = 0.0
            self.ti_grid_v[I] = ti.Vector.zero(real, 3)

        # ---- P2G ----
        for p in range(self.ti_particle_count[None]):
            xp = self.ti_particle_x[p]
            # grid-relative coordinate (our grid does NOT start at 0)
            xp_rel = (xp - self.ti_grid_min) * self.py_invdx
            base = (xp_rel - 0.5).cast(int)
            fx   = xp_rel - base.cast(real)
            w = [
                0.5 * (1.5 - fx) * (1.5 - fx),
                0.75 - (fx - 1.0) * (fx - 1.0),
                0.5 * (fx - 0.5) * (fx - 0.5),
            ]

            # Kirchhoff stress τ (same form as AGTaichiMPM.py)
            J = ti.sqrt(self.ti_particle_be[p].determinant())
            be_bar = self.ti_particle_be[p] * ti.pow(J, -2.0 / 3.0)
            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(real, 3) / 3.0
            tau = (self.py_kappa * 0.5 * (J + 1.0) * (J - 1.0)
                   * ti.Matrix.identity(real, 3)
                   + self.py_mu * dev_be_bar)

            # MLS affine matrix: stress contribution + APIC affine
            # affine @ dpos transfers both force impulse and affine velocity
            stress_aff = (-self.py_dt * 4.0 * self.py_invdx * self.py_invdx
                          * self.py_particle_volume) * tau
            affine = stress_aff + self.py_particle_mass * self.ti_particle_C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(real) - fx) * self.py_dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.ti_grid_v[base + offset] += weight * (
                    self.py_particle_mass * self.ti_particle_v[p]
                    + affine @ dpos)
                self.ti_grid_m[base + offset] += weight * self.py_particle_mass

        # ---- grid update ----
        for I in ti.grouped(self.ti_grid_m):
            if self.ti_grid_m[I] > 0:
                self.ti_grid_v[I] = self.ti_grid_v[I] / self.ti_grid_m[I]
                # gravity
                self.ti_grid_v[I] += self.py_dt * self.ti_g
                # sticking boxes (walls + floor)
                for s in range(self.ti_num_boxes[None]):
                    if (self.ti_static_box_min[s][0] <= self.ti_grid_x[I][0] <= self.ti_static_box_max[s][0]
                        and self.ti_static_box_min[s][1] <= self.ti_grid_x[I][1] <= self.ti_static_box_max[s][1]
                        and self.ti_static_box_min[s][2] <= self.ti_grid_x[I][2] <= self.ti_static_box_max[s][2]):
                        self.ti_grid_v[I] = ti.Vector.zero(real, 3)

        # ---- G2P + constitutive update ----
        for p in range(self.ti_particle_count[None]):
            xp = self.ti_particle_x[p]
            # grid-relative coordinate (our grid does NOT start at 0)
            xp_rel = (xp - self.ti_grid_min) * self.py_invdx
            base = (xp_rel - 0.5).cast(int)
            fx   = xp_rel - base.cast(real)
            w = [
                0.5 * (1.5 - fx) * (1.5 - fx),
                0.75 - (fx - 1.0) * (fx - 1.0),
                0.5 * (fx - 0.5) * (fx - 0.5),
            ]

            new_v = ti.Vector.zero(real, 3)
            new_C = ti.Matrix.zero(real, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(real) - fx) * self.py_dx
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = self.ti_grid_v[base + offset]
                new_v += weight * g_v
                # C_new = (1/D) · Σ w · v_g ⊗ dpos  with  1/D = 4/dx^2
                new_C += (4.0 * self.py_invdx * self.py_invdx
                          * weight) * g_v.outer_product(dpos)

            self.ti_particle_v[p] = new_v
            self.ti_particle_C[p] = new_C

            # velocity gradient = C_new (for quadratic B-spline MLS-MPM)
            vel_grad = new_C
            f = ti.Matrix.identity(real, 3) + self.py_dt * vel_grad
            f_bar = self.bar_3d(f)
            be_bar = self.bar_3d(self.ti_particle_be[p])
            be_bar_pre = f_bar @ be_bar @ f_bar.transpose()
            be = f @ self.ti_particle_be[p] @ f.transpose()
            det_be = be.determinant()
            J = ti.sqrt(det_be)

            sigma_s_pre = self.py_mu * self.dev_3d(be_bar_pre) / J
            sigma_s_pre_len = sigma_s_pre.norm()
            scalar_sigma_pre = sigma_s_pre_len / ti.sqrt(2.0)

            # default γ̇ = 0 (elastic regime)
            self.ti_particle_gamma_dot[p] = 0.0

            if scalar_sigma_pre - self.ti_hb_sigmaY[None] > 0.0:
                sigma_s_pre_hat = sigma_s_pre / sigma_s_pre_len
                sigma_s_len = self.scalar_hb_solve_3d(
                    sigma_s_pre_len, self.py_mu / J,
                    self.ti_hb_sigmaY[None], self.ti_hb_n[None],
                    self.ti_hb_eta[None], be_bar.trace(), self.py_dt)
                sigma_scalar = sigma_s_len / ti.sqrt(2.0)
                excess = sigma_scalar - self.ti_hb_sigmaY[None]
                if excess > 0.0:
                    self.ti_particle_gamma_dot[p] = ti.pow(
                        excess / self.ti_hb_eta[None],
                        1.0 / self.ti_hb_n[None])

                be_bar = ((be_bar.trace() / 3.0) * ti.Matrix.identity(real, 3)
                          + sigma_s_len * J * sigma_s_pre_hat / self.py_mu)
                det_be_bar = be_bar.determinant()
                be = be_bar * ti.pow(det_be, 1.0 / 3.0) / ti.pow(det_be_bar, 1.0 / 3.0)

            self.ti_particle_be[p] = be

            # wall sticking (apply *after* integration of v)
            for s in range(self.ti_num_boxes[None]):
                if (self.ti_static_box_min[s][0] <= xp[0] <= self.ti_static_box_max[s][0]
                    and self.ti_static_box_min[s][1] <= xp[1] <= self.ti_static_box_max[s][1]
                    and self.ti_static_box_min[s][2] <= xp[2] <= self.ti_static_box_max[s][2]):
                    self.ti_particle_v[p] = ti.Vector.zero(real, 3)
                    self.ti_particle_C[p] = ti.Matrix.zero(real, 3, 3)
                    self.ti_particle_is_inner_of_box[p] = 1
                    break
                else:
                    self.ti_particle_is_inner_of_box[p] = 0

            self.ti_particle_x[p] += self.py_dt * self.ti_particle_v[p]


# ===========================================================================
# runner helpers
# ===========================================================================

def save_frame(mpm, out_dir, frame_idx, save_dat=False):
    N = mpm.ti_particle_count[None]
    inside = mpm.ti_particle_is_inner_of_box.to_numpy()[:N].astype(np.int32)
    mask = inside == 0
    n_out = int(mask.sum())
    gdot = mpm.ti_particle_gamma_dot.to_numpy()[:N].astype(np.float32)[mask]
    pos  = mpm.ti_particle_x.to_numpy()[:N].astype(np.float32)[mask]

    gdot.tofile(os.path.join(out_dir, f"config_{frame_idx:02d}_gamma_dot.bin"))
    pos.tofile(os.path.join(out_dir, f"config_{frame_idx:02d}_pos.bin"))

    # pipeline-compatible .dat (matches fileOperation.saveState() in AGTaichiMPM.py)
    if save_dat:
        vel = mpm.ti_particle_v.to_numpy()[:N].astype(np.float32)[mask]
        dat_path = os.path.join(out_dir, f"config_{frame_idx:02d}.dat")
        with open(dat_path, "wb") as f:
            f.write(ctypes.c_int32(n_out))
            pos.flatten().tofile(f)
            (np.ones(n_out, np.float32) * mpm.py_particle_hl).tofile(f)
            vel.flatten().tofile(f)
            np.ones(n_out, np.int32).tofile(f)

    print(f"[AGTaichiMPM2] saved frame {frame_idx}: {gdot.size} particles, "
          f"γ̇ min={gdot.min():.3g} median={np.median(gdot):.3g} "
          f"max={gdot.max():.3g}")


def main():
    args = parse_cli()
    os.makedirs(args.out, exist_ok=True)

    cuboid_min = (-0.15, -0.15, -0.15)
    cuboid_max = (args.W, args.H, args.Z)

    arch = ti.cuda if args.taichi_arch == "cuda" else ti.cpu
    try:
        ti.init(default_fp=real, arch=arch)
    except Exception as e:
        print(f"[AGTaichiMPM2] Taichi init on {args.taichi_arch} failed ({e}); "
              "falling back to CPU")
        ti.init(default_fp=real, arch=ti.cpu)

    mpm = AGTaichiMPM2(args, cuboid_min, cuboid_max)
    mpm.initialize()

    # ---- JIT warm-up (excluded from timing) ----
    t_jit0 = time.perf_counter()
    mpm.step()
    ti.sync()
    jit_time = time.perf_counter() - t_jit0
    print(f"[AGTaichiMPM2] JIT compile + first step: {jit_time:.3f} s "
          "(excluded from timing)")
    mpm.initialize()
    ti.sync()

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"impl=MLS-MPM\neta={args.eta}\nn={args.n}\nsigmaY={args.sigmaY}\n"
                f"H={args.H}\nW={args.W}\nZ={args.Z}\nrho={args.rho}\n"
                f"dt={args.dt}\nfps={args.fps}\nmax_frames={args.max_frames}\n")

    save_frame(mpm, args.out, 0, save_dat=args.save_dat)

    num_saved = 1
    total_steps = 0
    t0 = time.perf_counter()
    while num_saved <= args.max_frames:
        for _ in range(100):
            mpm.step()
            total_steps += 1
            t = mpm.ti_iteration[None] * mpm.py_dt
            if t * mpm.py_fps >= num_saved:
                save_frame(mpm, args.out, num_saved, save_dat=args.save_dat)
                num_saved += 1
                if num_saved > args.max_frames:
                    break
    ti.sync()
    mpm_time = time.perf_counter() - t0

    with open(os.path.join(args.out, "timing.json"), "w") as f:
        json.dump({
            "impl": "MLS-MPM",
            "jit_time_sec": jit_time,
            "mpm_time_sec": mpm_time,
            "total_steps": total_steps,
            "time_per_step_ms": mpm_time * 1000.0 / max(1, total_steps),
            "frames": args.max_frames,
            "time_per_frame_sec": mpm_time / args.max_frames,
            "dt": args.dt,
            "arch": args.taichi_arch,
        }, f, indent=2)

    print(f"[AGTaichiMPM2] done. {args.max_frames + 1} frames in {args.out}/")
    print(f"[AGTaichiMPM2] TIMING: {mpm_time:.2f}s total MPM, "
          f"{total_steps} steps, {mpm_time*1000/total_steps:.2f} ms/step, "
          f"{mpm_time/args.max_frames:.2f} s/frame  (JIT {jit_time:.2f}s not counted)")


if __name__ == "__main__":
    main()
