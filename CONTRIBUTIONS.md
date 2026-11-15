# Development contributions in the StephM10 fork

This fork is maintained by **Leren Wang (StephM10)** and builds on **Sumitabha Brahmachari's chromdyn**. The original framework already provided OpenMM simulation, topology generation, force-field construction, active integrators, trajectory reporters, and Hi-C utilities. The contributions below extend that foundation. Original authorship and the MIT license are retained.

## Comparison baselines

The historical starting point is [`f903e7f`](https://github.com/StephM10/chromdyn/commit/f903e7f1ef6c565b0ff45f533d4819566ca859ca), the main-line version immediately before Leren's first development commit, [`3396a20`](https://github.com/StephM10/chromdyn/commit/3396a20), on 19 November 2025. This is a usable pre-contribution codebase, dated 19 September 2025.

A second baseline is upstream's [`91cb990`](https://github.com/StephM10/chromdyn/commit/91cb990), the v0.2 preparation on 11 December 2025. Earlier extensions were ported to this layout in [`4194b30`](https://github.com/StephM10/chromdyn/commit/4194b30). Comparing against both baselines distinguishes original contributions from changes introduced by upstream during the v0.2 reorganization.

The evidence is the Git history and source APIs, rather than a comparison with the repository's first, license-only commit. The older `Chromdyn_update` work and its v0.2 port describe one development history and are not counted twice.

## Contribution areas

| Area | Earlier capability | Contributions maintained in this fork | Representative evidence |
| --- | --- | --- | --- |
| Periodic simulation and trajectory metadata | A simulation/reporting framework and basic trajectory output existed | Periodic setup and metadata preservation; trajectory topology/type information; wrapped-coordinate views and checks; PBC usage examples | [`3396a20`](https://github.com/StephM10/chromdyn/commit/3396a20), [`f472c00`](https://github.com/StephM10/chromdyn/commit/f472c00), `chromatin_dynamics.py`, `reporters.py`, `traj_utils.py` |
| Periodic contact analysis | Ordinary within-configuration contact maps existed | Nearest-periodic-image contact calculations and examples, plus corrections to the PBC contact workflow | [`1cf3838`](https://github.com/StephM10/chromdyn/commit/1cf3838), [`5b383b7`](https://github.com/StephM10/chromdyn/commit/5b383b7), `hic_utils.py` |
| Trajectory access and structural analysis | The v0.2 baseline included basic writhe/Rg analysis and a static trajectory loader | A trajectory abstraction, type-resolved analysis, Rg components, mass weighting, tangent correlations, persistence-length estimation, and distance-distribution tools | [`9a01a7f`](https://github.com/StephM10/chromdyn/commit/9a01a7f), [`610b84c`](https://github.com/StephM10/chromdyn/commit/610b84c), [`ad12cec`](https://github.com/StephM10/chromdyn/commit/ad12cec), [`9170015`](https://github.com/StephM10/chromdyn/commit/9170015) |
| Dynamics analysis and memory control | Basic geometry/trajectory utilities existed | MSD and anomalous-dynamics analysis, velocity-correlation tools, CPU/CuPy paths, batched processing, memmap trajectory materialization, and averaged/type-resolved MSD | [`3db0455`](https://github.com/StephM10/chromdyn/commit/3db0455), [`c07e253`](https://github.com/StephM10/chromdyn/commit/c07e253), [`a539faf`](https://github.com/StephM10/chromdyn/commit/a539faf), `tests/test_msd_memory_batching.py` |
| Visualization | The upstream framework provided the starting simulation and analysis infrastructure | Trajectory/visualization module separation, particle-type and PBC views, GIF output, heterogeneous bead sizes, and cross-chain connection fixes | [`47a9402`](https://github.com/StephM10/chromdyn/commit/47a9402), [`7a5b612`](https://github.com/StephM10/chromdyn/commit/7a5b612), [`1a27260`](https://github.com/StephM10/chromdyn/commit/1a27260), `visualization.py` |
| Heterogeneous particles | Scalar mass and size parameters existed | Per-particle masses/radii through simulation, repulsion, mass-weighted analysis, and visualization | [`a539faf`](https://github.com/StephM10/chromdyn/commit/a539faf), `tests/test_hetero_mass_radius.py`, `tests/test_hetero_analysis_viz.py` |
| Hi-C processing | Upstream Hi-C calculation and matrix operations existed | Processing refinements, consistent model/contact parameters, and streaming CNDB reads on the CPU path | [`6d67b17`](https://github.com/StephM10/chromdyn/commit/6d67b17), [`4bc6b5a`](https://github.com/StephM10/chromdyn/commit/4bc6b5a), [`c2fd04f`](https://github.com/StephM10/chromdyn/commit/c2fd04f) |
| Optimization and reporting | Optimization methods and structure/energy/stability reporters existed | Separate full-gradient `gd` and sampled `sgd` updates; force-field configuration reporting; configurable stability reinitialization | [`ecdb062`](https://github.com/StephM10/chromdyn/commit/ecdb062), [`aaf926d`](https://github.com/StephM10/chromdyn/commit/aaf926d), [`1a27260`](https://github.com/StephM10/chromdyn/commit/1a27260) |

The same work may appear in an early feature commit, a v0.2 migration, and a later correction. Those are evidence of maintenance and refinement, rather than separate inventions.

## Changes consolidated in November 2026

- The overdamped active Brownian path uses an independent per-DOF OU force `p`, so the OpenMM velocity state is not used as an extra active drift. The preceding implementation is retained as `ActiveBrownianIntegratorLegacy` for comparison with old simulations.
- Optional coefficient caching moves repeated OU/diffusion coefficient calculations outside the integration loop. The cached noise coefficient uses `expm1` to reduce cancellation for small timestep/correlation-time ratios. Supported activity, timestep, friction, and thermal-energy setters refresh the coefficients. Invalid physical parameters are rejected before supported setters modify state.
- The public activity API checks particle-array shapes and finite values. Correlation times must be positive. Short-run timing uses a high-resolution clock and avoids division by zero.
- A general oscillatory harmonic-trap force is available independently of the local rheology experiments. Its potential and force are checked against the analytic expression; invalid inputs are rejected before force registration.
- Numerical regression tests, CPU-compatible heterogeneous-particle checks, and CI coverage for the maintenance branch accompany the implementation.
- The periodic tutorial is stored without execution outputs, and the effective-interaction example now runs a small synthetic CPU demonstration with a symmetric interaction matrix.

Rheology experiment scripts, experiment notebooks, local validation workspaces, and research outputs remain excluded. Publishing the reusable API does not make those experiments part of this maintained distribution.

## Upstream collaboration

Upstream history includes merge [`b96abc9`](https://github.com/s-brahmachari/chromdyn/commit/b96abc9), recording PR #2 from `StephM10/update_v_0.2` on 15 January 2026. Some contributions therefore already form part of upstream's history; later fork development is recorded separately on this branch.

The November 2026 consolidation is maintained in `StephM10/chromdyn:update_v_0.2`. No new upstream PR is part of this update.

## Validation and claim boundaries

Regression checks cover analytic trap energy/forces, parameter-update behavior, OU one-step statistics, thermal diffusion, cached/uncached integration, independence from state velocity, short-run timing, heterogeneous-particle behavior, and selected trajectory/MSD operations. CUDA comparisons run when CUDA is available. The small effective-interaction notebook is a workflow demonstration, not a convergence or equilibration study.

These checks support the tested numerical behaviors. They do not establish a universal speedup, model accuracy, or scientific novelty. Scientific applications still need model-specific convergence, controls, and parameter justification. Several historical commits explicitly record AI-assisted implementation; Git attribution identifies the maintainer responsible for those changes.
