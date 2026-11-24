# Correctness, numerical stability, and scalable analysis update (2026-11-24)

This update strengthens the simulation and trajectory-analysis workflows maintained
in the StephM10 fork of chromdyn. It corrects cases that could silently change
physical forces, frame selection, periodic distances, contact probabilities, or
optimization steps, and makes several expensive analyses faster or less demanding
of GPU memory. The changes are integrated into `update_v_0.2` and `main`.

The original framework is by **Sumitabha Brahmachari**; this fork's development
and maintenance are by **Leren Wang (StephM10)**. This report describes the current
fork update, compared with `cd209f7`, rather than an upstream release or a claim
that all existing chromdyn features were introduced here. See
[contribution history](CONTRIBUTIONS.md) for earlier work and attribution.

## What changes for users, and why it matters

The main significance is **numerical reliability**: successful execution and a
plausible plot can conceal incorrect sampling or analysis. The update adds
analytic and CPU/GPU regression cases for the affected paths, alongside the
implementation fixes.

| Workflow | Previous failure or limitation | Result of this update |
| --- | --- | --- |
| Trajectory time order and frame selection | String sorting could put frame 10 before frame 2; sparse keys could omit frames; distance analysis could apply the frame stride twice. | Numeric ordering and one position-based selection keep coordinates, boxes, and requested samples aligned. This protects time-dependent observables and subsampled distributions. |
| Periodic geometry and polymer connectivity | Wrapping each coordinate did not give minimum-image distances; contour pairs could cross chain boundaries. | Distances use the recorded orthorhombic box and chain-local contour pairs. In a box of length 10, positions 0.1 and 9.9 give distance 0.2 rather than 9.8. |
| Contact maps | Parallel PBC analysis could discard trailing frames and underweight the result; CPU/GPU self-contact conventions differed. | Every selected frame contributes with its actual weight, and self contacts are zero. The 21-frame/two-worker example previously used only 20 frames, reducing identical-frame results by 4.76%. |
| Forces and dynamics | Origin/axis gradients could be invalid despite finite potential energy; deleting forces could change the wrong interaction; changing an active-Langevin step size left half-step coefficients stale. | Analytic force checks, live force-index mapping, and synchronized half-steps protect the intended equations on the tested paths. |
| Long-trajectory analysis | GPU single-precision FFT terms could produce negative small-lag MSD; CPU/GPU geometry precision could change bin assignments. | Float64 GPU MSD intermediates and consistent spatial-correlation precision preserve the tested analytic values and bin membership. |
| Large analyses | Scalar writhe loops were expensive, and GPU contact calculations could transfer the full trajectory despite a batch setting. | Vectorized writhe and actual HDF5-to-GPU frame batches reduce measured runtime and limit trajectory-transfer memory. The dense contact matrix still requires O(N²) memory. |
| Repeated simulations and distribution | Output depended on the log switch; files and log routing could outlive their intended owner; package discovery included local experiment/build directories. | Independent output controls, owned-file finalization, isolated logs, and wheel-content checks make repeated runs and installation artifacts more reproducible. |

**Validation:** the final version-controlled suite has **201 passing tests** with
local GPU comparisons enabled. Both pushed branches passed GitHub CI. The final
N=250 writhe benchmark was about **196x faster** than the scalar implementation,
with absolute difference **3.04e-13**; this is a measured small example, not a
universal speedup. Small MSD and spatial-correlation workloads were faster on CPU
than GPU. Section 5 records the final measurements and the test-count scope.

## Compatibility and upgrading existing workflows

- **Nadam changes numerically:** it now follows Dozat's Algorithm 8 with constant
  momentum. Continuing old moment arrays applies the corrected update rule;
  optimization histories spanning the update should record that change.
- **PDB export uses consistent angstrom coordinates and boxes.** CNDB coordinates
  retain their recorded numerical length scale. Reduced simulations still need
  their own physical calibration; no new calibration is inferred.
- **Stability reporting changes:** active integrators use their live temperature
  scale; overdamped runs do not reset velocities; non-finite monitored energy
  raises an explicit error. Unknown custom integrators need a reference energy.
- **Writhe defaults to the vectorized implementation.** `method="legacy"` remains
  available for comparison; both reject NaN/Inf coordinates. Finite planar
  degeneracies retain their tested zero behavior.
- **Output finalization is explicit:** `save_reports()` closes and detaches owned
  reports; `close()` also closes owned log handlers. Use it when a run is finished.
- **Retained conventions:** `0.008314`, the friction equations, the RMSProp
  `beta1` mapping, cosine continuation, and the integration ordering are unchanged.

Users of the affected operations should regenerate those derived results from
their original trajectories when comparing software versions. These conditional
defects do not establish that every prior simulation or analysis was affected.
No research production trajectory was rerun for this update. See
[numerical conventions](NUMERICAL_NOTES.md) for the exact API and unit contracts.

The sections below retain the detailed implementation and verification record.

---

## 1. Phase 1 Correctness & API Safety Fixes

1. **Topology generation (`chromdyn/topology.py`)**
   - Unified residue/atom indexing in `TopologyGenerator.gen_top` via `divmod`, resolving `UnboundLocalError` when `chain_len < atoms_per_residue` and avoiding duplicate residue creation for remainder atoms.
   - Removed the `len(types) < 4` restriction so arbitrary-length single type names are preserved; rejected non-positive `atoms_per_residue`.

2. **Force field management (`chromdyn/forcefield.py`)**
   - Added `_flat_bottom_expression` (`select(q, ..., 0)` with `sqrt(q + delta(q))`) for `add_flat_bottom_harmonic`, `add_custom_flat_bottom_harmonic`, and `add_cylindrical_confinement`, eliminating `0/0` NaN gradients at the origin (`r = 0`, including zero confinement radius) without perturbing the potential elsewhere.
   - Replaced `atom.id` with 0-based `atom.index` across bond, angle, FENE, and nonbonded-exclusion builders.
   - Rebuilt `force_name_map` from live `System` forces (`_refresh_force_names`) on `removeForce`, resolved force indices by pointer identity (`_getForceIndex`), and rejected duplicate force names in `register_force`.
   - Added nonnegative finiteness validation for `r0`, `r_cyl`, and `z_cyl`.

3. **Active integrators (`chromdyn/integrators.py`)**
   - Updated `ActiveLangevinIntegrator` to derive `halfdt = 0.5 * dt` live each step so `setStepSize()` updates take effect accurately; added scalar/per-DOF validation and cancellation-free `(1 + c^2) * tanh(...)` OU variance expressions.
   - Documented friction conventions (`g` vs `gamma`) clearly and updated `IntegratorManager` log messages to reference `.set_activity(F_seq, tau_seq)`.

4. **Reporters & simulation lifecycle (`chromdyn/reporters.py`, `chromdyn/chromatin_dynamics.py`, `chromdyn/utilities.py`)**
   - Updated `StabilityReporter` to handle overdamped integrators without injecting spurious velocities, read live `kbT`, support `T = 0` finiteness checks with optional `stability_reference_energy`, and raise `FloatingPointError` on non-finite energies.
   - Isolated per-simulation `LogManager` handlers, deferred `output_dir` creation until an output is enabled, supported saving reports when `write_logs=False`, ensured `run(..., report=False)` always restores reporters via `try...finally`, and closed all owned reporter/log handles in `save_reports()` / `close()`.

5. **Optimization (`chromdyn/optimization.py`)**
   - Corrected `method="nadam"` to match Dozat Algorithm 8 (constant momentum `beta1`, `t+1` bias correction on `m_t`), avoided mutating caller-owned `phi_sim` arrays, and eliminated the import-time `HiCManager()` side effect.
   - Added explicit method validation (`VALID_METHODS`) in `__init__`, `set_opt_method`, and `update_step`.

6. **Hi-C utilities (`chromdyn/hic_utils.py`)**
   - Fixed `_remove_neighbor` so multiple offsets (e.g., `k=[0, 1]`) are all cleared rather than overwritten.
   - Fixed `_divide_into_subtraj` via `np.array_split` so trailing frames are included and weighted accurately.
   - Unified CUDA capability detection/CPU fallback (`_resolve_platform`), enforced zero diagonals on CPU PBC Hi-C, fixed `get_numpy_from_hic` random symmetric matrix generation, and initialized `self.hic_map` for `get_hic_map()` / `get_shape()`.

7. **Trajectory analysis & PDB export (`chromdyn/traj_utils.py`)**
   - Sorted CNDB numeric frame keys numerically (`_sorted_frame_keys`) across `TrajectoryLoader` and `Trajectory` (supporting zero-padded and sparse frame keys).
   - Fixed double frame-slicing in `Trajectory.get_distance_distribution_by_separation_and_type`, restricted contour pairs to within each topology chain, and implemented orthorhombic MIC (`_orthorhombic_lengths`) when `wrapped=True`.
   - Converted both atom positions and `CRYST1` box dimensions to angstroms in `save_pdb`.

---

## 2. Phase 2 Performance, Numerical Stability & Logging Upgrades

1. **Vectorized & unified spatial velocity correlation (`chromdyn/traj_utils.py`)**
   - Replaced nested Python `i_local` × `type_pairs` loops (and inner GPU host-device transfers/syncs) with a shared `_calculate_spatial_vel_corr_impl` using a precomputed `(n_types, n_types)` pair lookup table, block-tiled upper-triangle evaluation, `searchsorted`, and flattened 2D `bincount`.

2. **MSD numerical stability, FFT acceleration & backend unification (`chromdyn/traj_utils.py`)**
   - Unified `_msd_fft_cpu_batch` / `_msd_fft_gpu_batch` and `_autocorrFFT_*` / `_calculate_vacf_*` into shared CPU/GPU paths.
   - Subtracted frame-0 coordinates in `float64` prior to FFT/prefix-sum evaluation and accumulated `S1` in `float64`, preventing catastrophic cancellation under large coordinate offsets (`+1e5`).
   - Padded transforms using `scipy.fft.next_fast_len`, combined all 3 spatial power spectra into a single `irfft` (with `workers=-1` on CPU), moved CuPy memory-pool cleanup out of inner loops, and fixed `collect_distances_from_center_unified` / `collect_distances_from_com_unified` fallback handling when CuPy is absent.

3. **True batched Hi-C GPU streaming (`chromdyn/hic_utils.py`)**
   - Fixed `_calc_HiC_from_traj_array_gpu` so only `batch_size` frames are transferred to the GPU per iteration and accumulated in `cp.float64`.
   - Added streaming HDF5-to-GPU batch readers (`_gen_hic_from_cndb_gpu_streaming`, `_gen_pbc_hic_from_cndb_gpu_streaming`) for `gen_hic_from_cndb` and `gen_pbc_hic_from_cndb`.

4. **Vectorized writhe with legacy backup (`chromdyn/traj_utils.py`)**
   - Added `method: str = "vectorized"` (default) to `compute_writhe_single_curve`, `compute_writhe_between_curves`, and `compute_writhe_trajectory`, backed by block-tiled `_segment_solid_angle_batch` with zero-normal degenerate protection and corrected `closed=False` segment bounds. Preserved `method="legacy"` for exact backward comparison.

5. **Quiet imports & unified logging (`chromdyn/traj_utils.py`, `chromdyn/visualization.py`, `chromdyn/forcefield.py`, `chromdyn/hic_utils.py`, `.gitignore`)**
   - Removed the import-time `UserWarning` when CuPy is not installed; routed analysis and visualization status messages through module `logging` loggers; set `Analyzer.compute_RG(..., print_choice=False)` by default.
   - Deduplicated `.gitignore` and added `.pytest_tmp*/` and `notebooks/output/`.

---

## 3. Verification & Benchmarks

- **Initial local full test suite**, recorded before final review (`openmm` Python 3.11.15, `OpenMM 8.5.1`, `SciPy 1.16.3`, `CuPy 14.1.1`, `CHROMDYN_TEST_GPU=1`): **217 passed, 0 failed, 0 skipped**. The final reproducible, version-controlled test scope is recorded separately below.
- **Measured checks**:
  - `Writhe (N=250)`: `legacy = 2.2911 s` vs `vectorized = 0.0104 s` (**219.7x speedup**, `abs_diff = 1.15e-13`).
  - `MSD (M=20003 prime frames, B=50, +1e5 offset)`: `0.1176 s` on CPU, `max_rel_diff = 7.81e-10` vs unshifted trajectory, `min_msd = 0.00e+00`.
  - `SpatialVelCorr (T=3, N=400)`: CPU `0.0199 s`, max CPU vs CUDA GPU difference `7.09e-08`.
- **Documentation**: Recorded conventions in [NUMERICAL_NOTES.md](NUMERICAL_NOTES.md) and [README.md](README.md).

---

## 4. Deferred Items (For Future Consideration)

- `.cndb` storage evolution (Route A zero-format optimization vs optional chunked v2 format).
- Deduplicating `EnergyReporter` / `print_force_info` queries per unique force group.
- Shared `resolve_array_backend` helper across all analysis modules.

---

## 5. Final Review Corrections and Verification

Final review compared the update with `cd209f7` and checked both documented
behavior and implementation standards. It reproduced three additional numerical
edge cases and corrected them before pushing:

- **GPU MSD precision:** for `x(t)=t` over 20,003 frames, the previous float32
  spectra gave lag-3 MSD `-21.9632` instead of `9`. Shifted coordinates and
  spectra now remain float64 on GPU; the measured lag-3 result is approximately
  `9.00000006`. Both bead-resolved and average-MSD public APIs are covered.
- **Spatial correlation bins:** silently downcasting float64 coordinates to
  float32 on GPU moved a separation of `0.1` across the first float32 bin edge
  and changed which bins were empty. Both backends now evaluate geometry and
  velocity products in float64, preserving the existing bin edges and insertion
  rule. Exact-edge and large-offset CPU/CUDA comparisons are covered.
- **Non-finite writhe input:** NaN normals were treated as zero-normal
  degeneracies and could yield a finite partial writhe. All three public writhe
  entry points now reject NaN/Inf with `ValueError`, for both implementations;
  finite planar/duplicate-point controls retain their behavior.

The canonical final command selects **version-controlled tests only** (local
experimental scripts/tests remain excluded) with `CHROMDYN_TEST_GPU=1`:
**201 passed, 0 failed, 0 skipped**. This includes 18 added regression cases for
the three final-review findings. The unchanged pre-review tracked suite had
183 tests; its scope differs from the initial local 217-test run above.

Final static checks for undefined names/redefinitions and `git diff --check`
passed. Local wheel construction passed. Refreshed seeded benchmarks (seed
20261124, this machine only, measured separately from the test run) gave:

- Writhe, N=250: median legacy `2.647 s`, vectorized `0.01349 s`, approximately
  `196x` faster; absolute difference `3.04e-13`.
- MSD, 20,003 frames x 50 beads, batch size 10: CPU `0.149 s`, GPU `0.352 s`;
  max relative translation difference `8.62e-9`, CPU/GPU max absolute difference
  `1.88e-7`, minimum MSD zero.
- Spatial velocity correlation, 3 frames x 400 beads, chunk size 128: median
  CPU `0.01559 s`, GPU `0.05213 s`; maximum absolute difference `8.33e-17`.

These small workloads favor CPU over GPU; batching and float64 accuracy do not
imply a GPU speedup at every input size. The `0.008314` constant, friction
equations, RMSProp mapping, cosine continuation, and integration ordering remain
unchanged.

Wheel-content inspection also reproduced setuptools discovering local namespace
directories (`local_rheology_checks`, `build/lib`, and `scripts`) as runtime
packages. Package discovery is now explicitly limited to `chromdyn` and its
subpackages. `scripts/check_wheel_contents.py` rejects unexpected wheel roots;
it was checked against the contaminated wheel and the corrected build and is
run by CI after package construction. Local experimental files remain excluded
from Git.
