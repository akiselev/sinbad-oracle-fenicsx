"""Capability registry: what this adapter can independently solve (SV0-C5 D1-D5).

Pure standard library so the adapter can answer every refusal question
(unknown capability, unsupported observable, unaddressable refinement)
without importing dolfinx, and so offline CI can assert the registry's
shape. The dolfinx-backed module named by each entry is imported lazily by
the adapter only after `capability.probe_dolfinx()` has confirmed dolfinx is
importable.

Each entry mirrors exactly one Sinbad case file (`sinbad/cases/<case>.toml`)
and its `.res` model. The capability id is the string Sinbad's
`OracleCapability` enum renders on the wire (`snake_case`); today Sinbad
declares only `poisson`, so the other four are this adapter's proposal for
the enum's extension (recorded in STATUS.md as a cross-repo need) -- the
adapter answers them honestly either way.

Normalization contract, version 1 (`OracleToolIdentity.normalization_version`):
every observable is a single finite IEEE-754 double in the SI unit system the
case authors, computed by exact quadrature of the discrete solution on the
mesh the adapter itself generated. The per-observable definitions live in
`NORMALIZATION.md` and are repeated on each `CapabilitySpec.normalization`
so a mismatch is caught by `tests/test_registry.py`. Adding an observable to
a capability is additive and keeps version 1; changing the definition of an
existing observable id requires bumping `adapter.NORMALIZATION_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class CapabilitySpec:
    #: wire id (`OracleCapability`, snake_case)
    capability: str
    #: Sinbad case file stem this capability mirrors
    sinbad_case: str
    #: the `.res` model name
    model: str
    #: spatial dimension of the mirrored case geometry
    dimension: int
    #: module under `sinbad_oracle_fenicsx` providing `solve(refinement) -> SolveOutcome`
    module: str
    #: observable id -> normalization-1 definition
    normalization: Mapping[str, str]

    @property
    def observables(self) -> frozenset[str]:
        return frozenset(self.normalization)


def _spec(capability, sinbad_case, model, dimension, module, normalization) -> CapabilitySpec:
    return CapabilitySpec(
        capability=capability,
        sinbad_case=sinbad_case,
        model=model,
        dimension=dimension,
        module=module,
        normalization=MappingProxyType(dict(normalization)),
    )


_L2_ERROR = "L2 norm over the domain of (discrete field - exact manufactured field)"
_H1_ERROR = "L2 norm over the domain of grad(discrete field - exact manufactured field)"
_NODAL = (
    "root-mean-square over every mesh vertex and field component of "
    "(vertex value - exact manufactured value), Sinbad's `nodal_l2_error` normalization"
)

CAPABILITIES: Mapping[str, CapabilitySpec] = MappingProxyType(
    {
        "poisson": _spec(
            "poisson",
            "01-poisson",
            "Poisson",
            2,
            "poisson",
            {
                "energy": "integral over the domain of 0.5 * k * dot(grad u, grad u), k = 1",
                "l2_error": _L2_ERROR,
                "h1_seminorm_error": _H1_ERROR,
            },
        ),
        "nonlinear_heat": _spec(
            "nonlinear_heat",
            "03-nonlinear-heat",
            "NonlinearHeat",
            2,
            "nonlinear_heat",
            {
                "total_energy": (
                    "integral over the domain of rho * cp * T at the final time t = 0.4 of the "
                    "BDF2 march (step 0.05, BDF1 start-up step), rho = cp = 1"
                ),
                "l2_error": _L2_ERROR + " at the final time t = 0.4",
                "h1_seminorm_error": _H1_ERROR + " at the final time t = 0.4",
                "nodal_l2_error": _NODAL + " at the final time t = 0.4",
                "steady_l2_error": (
                    "L2 norm of (Newton steady-state solution - exact field), the time-independent "
                    "companion sample for mesh-convergence estimates"
                ),
                "forcing_defect": (
                    "L2 norm over the domain of (case-authored source Q - the source UFL derives "
                    "symbolically as -div(k(T_exact) grad T_exact)); an independent check of the "
                    "case's hand-derived forcing"
                ),
            },
        ),
        "linear_elasticity": _spec(
            "linear_elasticity",
            "17-linear-elasticity",
            "LinearElasticity",
            3,
            "linear_elasticity",
            {
                "displacement_magnitude_sq": "integral over the domain of dot(u, u)",
                "strain_energy": (
                    "integral over the domain of 0.5 * inner(strain, stress), "
                    "stress = lambda tr(strain) I + 2 mu strain, lambda = 1.25, mu = 1"
                ),
                "l2_error": _L2_ERROR,
                "h1_seminorm_error": _H1_ERROR,
                "nodal_l2_error": _NODAL,
                "forcing_defect": (
                    "L2 norm over the domain of (case-authored body force - the body force UFL "
                    "derives symbolically as -div(stress(sym_grad(u_exact))))"
                ),
            },
        ),
        "stokes": _spec(
            "stokes",
            "25-stokes",
            "StokesFlow",
            2,
            "stokes",
            {
                "dissipation": (
                    "integral over the domain of inner(2 mu sym_grad(u), sym_grad(u)), mu = 1.7"
                ),
                "mass_defect": "integral over the domain of div(u)",
                "divergence_l2_norm": "L2 norm over the domain of div(u)",
                "velocity_l2_norm": "L2 norm over the domain of u",
                "pressure_l2_norm": (
                    "L2 norm over the domain of p under the zero-mean gauge (p shifted so its "
                    "domain integral vanishes)"
                ),
                "solution_rms": (
                    "root-mean-square over the concatenated P2 velocity (both components, every "
                    "P2 node) and P1 pressure (every P1 node) degrees of freedom, the pressure "
                    "dofs shifted to zero arithmetic mean (Sinbad's `pressure_mean` gauge)"
                ),
            },
        ),
        "mixed_darcy": _spec(
            "mixed_darcy",
            "13-mixed-darcy",
            "MixedDarcy",
            3,
            "mixed_darcy",
            {
                "flux_l2_norm": "L2 norm over the domain of the RT0 flux u",
                "pressure_l2_norm": (
                    "L2 norm over the domain of the P0 pressure under the zero-mean gauge"
                ),
                "mass_residual_l2": (
                    "L2 norm over the domain of (div(u) - source_term), the strong mass-balance "
                    "residual of the discrete solution"
                ),
                "total_flow": "integral over the whole boundary of dot(u, n)",
                "source_compatibility_defect": (
                    "integral over the domain of source_term minus the integral over the boundary "
                    "of the prescribed normal flux; must vanish for the declared impermeable "
                    "problem to have a solution"
                ),
            },
        ),
    }
)


def lookup(capability: str) -> CapabilitySpec | None:
    return CAPABILITIES.get(capability)
