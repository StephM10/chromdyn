# chromdyn
*A Python library for chromosome dynamics simulation and analysis with options to integrate Hi-C data.*

The original simulation framework was developed by [Sumitabha Brahmachari](https://github.com/s-brahmachari/chromdyn).
Leren Wang's development extensions are maintained on [`StephM10/chromdyn:update_v_0.2`](https://github.com/StephM10/chromdyn/tree/update_v_0.2) and extend the analysis, periodic workflows, reporting, and active-dynamics tools.

## Maintained extensions

- Periodic trajectory metadata, coordinate wrapping, nearest-image contact analysis, and PBC tutorials.
- Trajectory loading and analysis, including MSD, velocity correlations, type-resolved and mass-weighted radius of gyration, and persistence-length fitting.
- CPU/GPU analysis paths and batched trajectory/MSD processing where supported.
- Visualization of periodic structures and particle types, with support for heterogeneous bead sizes.
- Per-particle masses and radii, force-field parameter reporting, and separate full-gradient and stochastic-gradient optimization paths.
- An overdamped active Brownian integrator with independent active-force state, optional cached coefficients, and regression checks.
- A reusable oscillatory harmonic-trap force API. Local rheology experiment scripts and research outputs are excluded from this update.

See [Contribution history](CONTRIBUTIONS.md) for the original-main baseline, the v0.2 migration, and commit-level attribution. Features listed here are code capabilities; performance and scientific conclusions require measurements for the chosen model and parameters.

## Overview

`chromdyn` provides tools for building, simulating, and analyzing 3D chromosome structures.  
It uses **OpenMM** for fast CPU/GPU simulations and includes:

- Polymer topology generation  
- Customizable force fields  
- Hi-C-based restraints  
- Simulation utilities  
- Analysis tools

## Installation

### From source

```
git clone https://github.com/s-brahmachari/chromdyn.git
cd chromdyn
pip install -e .
```

For visualization and regression tests, install the test extra:

```bash
pip install -e ".[test]"
```

To use Leren's maintained development branch, clone it explicitly before installation:

```bash
git clone --branch update_v_0.2 https://github.com/StephM10/chromdyn.git
```

The versions on PyPI, upstream main, and the maintained development branch may differ.

## Verify Installation

Run the built-in test script:

```
python -m chromdyn.check_install
```

From a clean source checkout with the test extra installed:

```bash
python -m pytest -v
```

The small numerical tests use OpenMM's CPU Reference platform. CUDA-specific comparisons skip when CUDA is unavailable. CI checks both `main` and `update_v_0.2`.

See [Numerical conventions and corrected behavior](NUMERICAL_NOTES.md) for friction
parameters, frame selection, periodic distances, PDB units, Dozat Nadam, and report
finalization. Additional local GPU regressions can be enabled with
`CHROMDYN_TEST_GPU=1` when OpenMM GPU platforms and CuPy are available.

## Quick Example

```python
from chromdyn import ChromatinDynamics, TopologyGenerator

gen = TopologyGenerator()
gen.gen_top([100])

sim = ChromatinDynamics(
    gen.topology,
    name="demo",
    platform_name="CPU",
)

sim.force_field_manager.add_harmonic_bonds()
sim.simulation_setup(
    init_struct="randomwalk",
    integrator="langevin",
    temperature=300.0,
    timestep=0.01,
)

sim.run(1000)
sim.close()  # Finalize owned report and log files.
```

## Tutorials

See the `notebooks/` directory for examples on:

- Building polymer models  
- Running simulations  
- Using Hi-C restraints  
- Analyzing trajectories  

- [Small effective-interaction example](notebooks/effective_simulation_example.ipynb): synthetic interactions, a short CPU simulation, and a contact map.
- [Periodic simulation tutorial](notebooks/PBC_sim_test.ipynb): periodic simulation and analysis examples, stored without execution outputs.

Coordinates saved by a periodic simulation may remain unwrapped. Coordinate wrapping and nearest-periodic-image contacts are explicit analysis choices; an ordinary contact-map call does not automatically select the PBC observable.

## Active dynamics and oscillatory traps

Select `integrator="active-brownian"` for overdamped active dynamics. The active force is stored separately from the OpenMM velocity state. Set per-particle activity through `sim.set_activity(F_seq, tau_seq)`; correlation times must be finite and positive. Cached coefficients are enabled by default and refreshed when supported parameter setters are used. Set `precompute_coefficients=False` in `simulation_setup()` for the uncached path.

Add an oscillatory trap before setting up the simulation:

```python
sim.force_field_manager.apply_oscillatory_tweezer(
    mono_id=0, k_ot=2.0, A=0.5, omega=1.0, pos=(0.0, 0.0, 0.0),
)
```

Here `omega` is angular frequency. With an active integrator, pass `advance_time=True` to `simulation_setup()` to advance the trap's shared Context parameter `t`. Otherwise update that parameter explicitly. Length, time, and energy use OpenMM internal units: nm, ps, and kJ/mol.


## License

MIT License.
