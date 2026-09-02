# Normalization contract

<!-- Generated from src/sinbad_oracle_fenicsx/registry.py by `python -m sinbad_oracle_fenicsx.normalization_doc`; do not edit by hand. -->

`OracleToolIdentity.normalization_version = 1`.

Every observable is one finite IEEE-754 double (wire form: the `FiniteF64` bit pattern,
see `protocol.py`) in the SI unit system the mirrored Sinbad case authors, computed by
exact quadrature of the discrete solution on the mesh the adapter itself generated
(Finitum's `simplex_box` layout: unit box, `[nx, ny]` bricks each split along the
lower-left/upper-right diagonal in 2-D, six Kuhn tetrahedra per brick in 3-D).
Observable ids are the keys of `OracleResult.observables`; where an id matches a
Sinbad `[[observables]]` name or `.res` observable it computes the same quantity.

Adding an observable id to a capability is additive and keeps this version. Changing
the definition behind an existing id bumps `adapter.NORMALIZATION_VERSION` and this
document together.

## `poisson` -- `01-poisson` (`Poisson`, 2-D)

| observable id | definition |
|---|---|
| `energy` | integral over the domain of 0.5 * k * dot(grad u, grad u), k = 1 |
| `l2_error` | L2 norm over the domain of (discrete field - exact manufactured field) |
| `h1_seminorm_error` | L2 norm over the domain of grad(discrete field - exact manufactured field) |

## `nonlinear_heat` -- `03-nonlinear-heat` (`NonlinearHeat`, 2-D)

| observable id | definition |
|---|---|
| `total_energy` | integral over the domain of rho * cp * T at the final time t = 0.4 of the BDF2 march (step 0.05, BDF1 start-up step), rho = cp = 1 |
| `l2_error` | L2 norm over the domain of (discrete field - exact manufactured field) at the final time t = 0.4 |
| `h1_seminorm_error` | L2 norm over the domain of grad(discrete field - exact manufactured field) at the final time t = 0.4 |
| `nodal_l2_error` | root-mean-square over every mesh vertex and field component of (vertex value - exact manufactured value), Sinbad's `nodal_l2_error` normalization at the final time t = 0.4 |
| `steady_l2_error` | L2 norm of (Newton steady-state solution - exact field), the time-independent companion sample for mesh-convergence estimates |
| `forcing_defect` | L2 norm over the domain of (case-authored source Q - the source UFL derives symbolically as -div(k(T_exact) grad T_exact)); an independent check of the case's hand-derived forcing |

## `linear_elasticity` -- `17-linear-elasticity` (`LinearElasticity`, 3-D)

| observable id | definition |
|---|---|
| `displacement_magnitude_sq` | integral over the domain of dot(u, u) |
| `strain_energy` | integral over the domain of 0.5 * inner(strain, stress), stress = lambda tr(strain) I + 2 mu strain, lambda = 1.25, mu = 1 |
| `l2_error` | L2 norm over the domain of (discrete field - exact manufactured field) |
| `h1_seminorm_error` | L2 norm over the domain of grad(discrete field - exact manufactured field) |
| `nodal_l2_error` | root-mean-square over every mesh vertex and field component of (vertex value - exact manufactured value), Sinbad's `nodal_l2_error` normalization |
| `forcing_defect` | L2 norm over the domain of (case-authored body force - the body force UFL derives symbolically as -div(stress(sym_grad(u_exact)))) |

## `stokes` -- `25-stokes` (`StokesFlow`, 2-D)

| observable id | definition |
|---|---|
| `dissipation` | integral over the domain of inner(2 mu sym_grad(u), sym_grad(u)), mu = 1.7 |
| `mass_defect` | integral over the domain of div(u) |
| `divergence_l2_norm` | L2 norm over the domain of div(u) |
| `velocity_l2_norm` | L2 norm over the domain of u |
| `pressure_l2_norm` | L2 norm over the domain of p under the zero-mean gauge (p shifted so its domain integral vanishes) |
| `solution_rms` | root-mean-square over the concatenated P2 velocity (both components, every P2 node) and P1 pressure (every P1 node) degrees of freedom, the pressure dofs shifted to zero arithmetic mean (Sinbad's `pressure_mean` gauge) |

## `mixed_darcy` -- `13-mixed-darcy` (`MixedDarcy`, 3-D)

| observable id | definition |
|---|---|
| `flux_l2_norm` | L2 norm over the domain of the RT0 flux u |
| `pressure_l2_norm` | L2 norm over the domain of the P0 pressure under the zero-mean gauge |
| `mass_residual_l2` | L2 norm over the domain of (div(u) - source_term), the strong mass-balance residual of the discrete solution |
| `total_flow` | integral over the whole boundary of dot(u, n) |
| `source_compatibility_defect` | integral over the domain of source_term minus the integral over the boundary of the prescribed normal flux; must vanish for the declared impermeable problem to have a solution |
