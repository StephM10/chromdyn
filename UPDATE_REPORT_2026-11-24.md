# Update Report (2026-11-24)

This report consolidates all correctness fixes, numerical-stability and performance optimizations, logging cleanups, and regression tests integrated into `update_v_0.2` and merged into `main`.

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

- **Full test suite** (`openmm` Python 3.11.15, `OpenMM 8.5.1`, `SciPy 1.16.3`, `CuPy 14.1.1`, `CHROMDYN_TEST_GPU=1`): **217 passed, 0 failed, 0 skipped**.
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
