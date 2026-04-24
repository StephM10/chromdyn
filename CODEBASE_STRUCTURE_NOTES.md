# chromdyn Codebase Structure Notes

This document records the current understanding of the `chromdyn` package structure and core workflows. It is intended as a quick context file for future development sessions.

## Project Purpose

`chromdyn` is a Python package for chromosome and polymer dynamics simulation and analysis. It uses OpenMM as the simulation backend and provides tools for topology generation, force-field assembly, integrator management, trajectory output, Hi-C-related analysis, trajectory post-processing, and visualization.

Typical workflow:

1. Use `TopologyGenerator` to create an OpenMM `Topology`.
2. Use `ChromatinDynamics` to create an OpenMM `System` and the package managers.
3. Add interactions and constraints through `sim.force_field_manager`.
4. Call `sim.simulation_setup()` to create the OpenMM `Simulation`, integrator, initial configuration, and reporters.
5. Call `sim.run()` to run the simulation.
6. Use `.cndb` trajectories, energy reports, Hi-C utilities, and analysis tools for post-processing.

## Top-Level Files and Directories

- `chromdyn/`: Python package source code.
- `tests/`: pytest tests and longer manual or semi-automated test routines.
- `notebooks/`: tutorial and simulation example notebooks.
- `scripts/`: pre-release helper scripts.
- `.github/workflows/`: CI and PyPI publishing workflows.
- `pyproject.toml`: package metadata, dependencies, build configuration, and formatter configuration.
- `README.md`: short user-facing overview, installation notes, and quick example.

## Public API

`chromdyn/__init__.py` exports the most common objects at package level:

- `ChromatinDynamics`
- `TopologyGenerator`
- `ForceFieldManager`
- `PlatformManager`
- `IntegratorManager`
- `TrajectoryLoader`
- `Trajectory`
- `Analyzer`
- `HiCManager`
- `SaveStructure`
- `StabilityReporter`
- `EnergyReporter`
- `config_generator`
- `LogManager`
- `save_pdb`

Common user import:

```python
from chromdyn import ChromatinDynamics, TopologyGenerator, Analyzer, HiCManager
```

## Core Modules

### `chromdyn/chromatin_dynamics.py`

The main class is `ChromatinDynamics`.

Responsibilities:

- Store the simulation name, package version, output directory, and logger.
- Create an OpenMM `System` from the provided topology.
- Add one particle per bead with the requested mass.
- Initialize `PlatformManager` and `ForceFieldManager`.
- Configure PBC, integrator, OpenMM `Simulation`, initial coordinates, and reporters in `simulation_setup()`.
- Provide `run()`, `pause_reporters()`, `resume_reporters()`, and `save_reports()`.
- Provide active-integrator helpers through `set_activity()` and `get_active_params()`.
- Log per-force information through `print_force_info()`, including force group, particle/bond/exclusion counts, and per-particle potential energy.

Important conventions:

- Forces are normally added before calling `simulation_setup()`.
- `self.simulation` is `None` until `simulation_setup()` is called.
- PBC is controlled by `simulation_setup(PBC=True, box_vectors=...)`; this also switches nonbonded forces to periodic cutoff behavior.

### `chromdyn/topology.py`

The main class is `TopologyGenerator`.

Responsibilities:

- Generate an OpenMM `Topology`.
- Support multiple chains.
- Support a single type, an explicit type list, `"unique"` types, or type loading from a file.
- Support `atoms_per_residue`.
- Support `isRing`, which adds a bond between the first and last atom in selected chains.
- Provide `print_top()` and `save_top()`.

Important outputs:

- `generator.topology`
- `generator.atom_types`

### `chromdyn/forcefield.py`

The main class is `ForceFieldManager`.

Responsibilities:

- Hold the OpenMM `System` and topology.
- Register forces consistently while maintaining `forceDict` and `force_name_map`.
- Automatically create exclusions from topology bonds for `CustomNonbondedForce` objects.
- Manage nonbonded method and cutoff; under PBC it checks whether the cutoff exceeds half the smallest box length.

Existing force methods include:

- `add_harmonic_bonds()`
- `add_harmonic_angles()`
- `add_fene_bonds()`
- `add_harmonic_trap()`
- `add_flat_bottom_harmonic()`
- `add_custom_flat_bottom_harmonic()`
- `add_cylindrical_confinement()`
- `add_self_avoidance()`
- `add_lennard_jones_force()`
- `add_wca_force()`
- `add_LJ_repulsion()`
- `add_type_to_type_interaction()`
- `addLEFBonds()`
- `removeCOM()`
- `constrain_monomer_pos()`
- `apply_force_z_axis()`
- `removeForce()`

New interactions should usually be added here and registered through `register_force()`.

### `chromdyn/integrators.py`

The main class is `IntegratorManager`.

Supported integrator names:

- `"langevin"`
- `"brownian"`
- `"active-langevin"`
- `"active-brownian"`

The module also defines two custom OpenMM `CustomIntegrator` classes:

- `ActiveBrownianIntegrator`
- `ActiveLangevinIntegrator`

Active parameters are stored as per-DOF variables. `ChromatinDynamics.set_activity()` sets each bead's `F_act` and `t_corr`.

### `chromdyn/platforms.py`

The main class is `PlatformManager`.

Responsibilities:

- List available OpenMM platforms.
- Select platforms such as `CUDA`, `OpenCL`, or `CPU`.
- Fall back to `CPU` when the requested platform is unavailable.

### `chromdyn/reporters.py`

Main reporters:

- `SaveStructure`: saves coordinates to `.cndb` HDF5 files.
- `StabilityReporter`: checks kinetic and potential energy per particle, and can reinitialize velocities when instability is detected.
- `EnergyReporter`: writes step, temperature, Rg, kinetic energy per particle, potential energy per particle, and optional force-group energies.

`.cndb` conventions:

- Each frame is saved with a numeric string key, such as `"0"` or `"1"`.
- The `types` dataset stores bead types.
- The `topology` group stores atoms, bonds, chain ids, and residue names.
- For PBC simulations, each frame dataset stores the periodic box in its `box` attribute.

### `chromdyn/traj_utils.py`

This is the most feature-rich analysis module.

Main classes:

- `Analyzer`: static and class-method analysis utilities.
- `TrajectoryLoader`: simple `.cndb` coordinate loader returning NumPy arrays.
- `Trajectory`: fuller `.cndb` trajectory object that reads topology, types, box vectors, and selected coordinates.

`Analyzer` includes:

- writhe calculations.
- radius of gyration.
- tangent correlation.
- VACF.
- spatial velocity correlation.
- MSD and anomalous dynamics.
- PBC coordinate wrapping.
- distance collection by bead type.

`Trajectory` includes:

- `.cndb` loading.
- coordinate access through `xyz()`.
- topology reconstruction.
- `chain_info()`.
- type-specific Rg and persistence-length analysis.
- wrapped-coordinate handling.
- wrapped-trajectory checks.
- distance distributions by genomic separation and bead type.

### `chromdyn/hic_utils.py`

The main class is `HiCManager`.

Responsibilities:

- Load Hi-C matrices.
- Filter matrices.
- Perform symmetric normalization.
- Normalize by diagonal averages.
- Generate contact probabilities.
- Generate simulated Hi-C from `.cndb` trajectories.
- Support CPU multiprocessing.
- Use CUDA/GPU paths when CuPy is installed and available.
- Support PBC Hi-C generation through the minimum image convention.

Notes:

- `gen_hic_from_cndb()` does not use PBC.
- `gen_pbc_hic_from_cndb()` reads box vectors through `Trajectory`.

### `chromdyn/optimization.py`

The main class is `EnergyLandscapeOptimizer`.

Responsibilities:

- Load experimental Hi-C data.
- Initialize and save optimizer state.
- Compute gradients from experimental and simulated Hi-C differences.
- Update parameter matrices using several optimization algorithms.

Supported optimization methods:

- `adam`
- `nadam`
- `rmsprop`
- `adagrad`
- `sgd`
- `gd`

Supported learning-rate schedulers:

- `none`
- `step`
- `cosine`
- `exponential`

### `chromdyn/visualization.py`

Visualization utilities. `matplotlib` is an optional dependency.

Main capabilities:

- Static 3D trajectory plots.
- Trajectory animations.
- PBC box drawing.
- PBC image display.
- Coloring by chain or bead type.

### `chromdyn/utilities.py`

Contains:

- `LogManager`: creates loggers that can write to console and files.
- `config_generator`: generates initial coordinate configurations.

`config_generator` supports:

- `"randomwalk"`
- `"saw3d"`
- `"random"`

### `chromdyn/check_install.py`

Installation check entry point:

```bash
python -m chromdyn.check_install
```

It checks:

- whether `chromdyn` can be imported.
- whether OpenMM is available and which platforms are visible.
- whether a minimal CPU simulation can run.

## Tests and CI

`tests/run_test.py` contains:

- `test_minimal_simulation_runs()`, which is the pytest test that runs automatically.
- a `run_tests` class with longer simulation checks and examples for harmonic bonds, confinement, self-avoidance, force removal, and bad-solvent behavior.

CI is defined in `.github/workflows/ci.yml`:

- Use Python 3.11.
- Install the package.
- Install pytest.
- Run `pytest -v`.
- Run `python -m build` to check package builds.

Publishing is defined in `.github/workflows/publish.yml`; it builds and publishes to PyPI when a GitHub release is published.

## Development Notes

- Any natural language written into this repository should be in English.
- New forces should usually live in `ForceFieldManager` and be registered through `register_force()`.
- New simulation workflow parameters usually belong in `ChromatinDynamics.simulation_setup()`.
- New trajectory analysis should be placed according to responsibility: `Analyzer`, `Trajectory`, or `HiCManager`.
- `.cndb` is the central trajectory format; reporter changes should be checked against `Trajectory` and `HiCManager`.
- Several validations currently use `assert`; for more robust public APIs, these can gradually be replaced with explicit exceptions.
- Some analysis and visualization paths still use `print()`; these can gradually be unified under the package logger.
- Git commands may trigger a `safe.directory` ownership check under the current sandbox user. This affects commands such as `git status`, but not normal file reads.

