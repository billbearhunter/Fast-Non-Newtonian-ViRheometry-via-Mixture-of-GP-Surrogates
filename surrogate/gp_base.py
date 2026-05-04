"""GP building blocks for HVIMoGP-rBCM experts.

Split out of the legacy `surrogate.models` (which also held SVGP code that
v2 no longer uses). v2 experts use `SingleOutputExactGP` with a named
kernel from `KERNEL_REGISTRY`; that's all we need here.
"""
from typing import Callable, Dict

import gpytorch


# ── Kernel registry ──────────────────────────────────────────────────────────
def _k_baseline():
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5)
        + gpytorch.kernels.LinearKernel()
    )


KERNEL_REGISTRY: Dict[str, Callable[[], gpytorch.kernels.Kernel]] = {
    "baseline": _k_baseline,
    "matern15": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=1.5)
        + gpytorch.kernels.LinearKernel()),
    "matern05": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=0.5)
        + gpytorch.kernels.LinearKernel()),
    "rbf": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        + gpytorch.kernels.LinearKernel()),
    "rbf_noLin": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()),
    "matern25_noLin": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5)),
    "matern25_ard": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=5)
        + gpytorch.kernels.LinearKernel()),
    "rbf_ard": lambda: gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=5)
        + gpytorch.kernels.LinearKernel()),
}


def make_kernel(name: str = "baseline") -> gpytorch.kernels.Kernel:
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name}. "
                         f"Options: {list(KERNEL_REGISTRY)}")
    return KERNEL_REGISTRY[name]()


# ── Exact GP ─────────────────────────────────────────────────────────────────
class SingleOutputExactGP(gpytorch.models.ExactGP):
    """Exact GP per-expert per-output. Kernel is named; default 'baseline'
    is Matern-2.5 + Linear (what v2 was trained with)."""
    def __init__(self, train_x, train_y, likelihood,
                 kernel_name: str = "baseline"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = make_kernel(kernel_name)
        self.kernel_name  = kernel_name

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
