# Numerical conventions and corrected behavior

## Friction and thermal energy

OpenMM's builtin Brownian integrator uses the inverse-time friction parameter
gamma. Its displacement has deterministic mobility `1/(m*gamma)` and thermal
variance per component `2*RT*dt/(m*gamma)`.

`ActiveBrownianIntegrator` uses physical drag `g` directly:

```text
dx = (f+p)*dt/g + sqrt(2*RT*dt/g)*xi
```

Consequently `g=m*gamma` is needed to compare the two models at a given mass.
The active-Brownian displacement does not depend on particle mass. The public
`friction` argument supplies `g` for `active-brownian`, and supplies `gamma` for
builtin Brownian, builtin Langevin, and `active-langevin`. No per-particle drag
extension is introduced by these fixes. See the [OpenMM equations](https://docs.openmm.org/latest/userguide/theory/04_integrators.html#brownianintegrator).

The existing active-integrator numerical constant `RT=0.008314*T` is retained.
Active Langevin recomputes its half-step from the current step size each step.
Its OU variance uses an algebraically equivalent expression that avoids
cancellation for small time-step ratios; its integration ordering is unchanged.

## Frames, contour pairs, and periodic distances

CNDB numeric frame keys are sorted numerically while preserving their original
strings, including zero padding. `Trajectory.xyz(frames=[start,end,stride])`
selects ordinal positions in that sequence; `end` is exclusive. Coordinates and
box vectors use the same positions. Memory, batched, and memmap reads share this
selection rule. The legacy `TrajectoryLoader.load(...,d=...)` keeps its existing
frame-ID divisibility rule (`int(key) % d == 0`).

For type-resolved distances at contour separation `s`, pairs are constructed
within each topology chain. For chains `[A0,A1]` and `[B0,B1]`, `s=1` includes
`A0-A1` and `B0-B1`; it excludes `A1-B0`. Without topology, the array is treated
as one linear chain. Ring chains retain chain-order separation, without a
shortest-ring-contour convention.

`wrapped=True` requests minimum-image distances and requires recorded box data.
For an orthorhombic box with positive lengths L, the displacement is
`dr -= L*round(dr/L)`. This differs from merely placing each coordinate inside
the box. The wrapping/MIC and PDB paths validate the orthorhombic-box convention;
tilted boxes require a separate implementation. CPU and GPU distance workers
use the same pairing and displacement calculation.

Hi-C excludes self contacts on all backends. PBC multiprocessing includes every
selected frame and weights each partition by its actual frame count. CUDA
unavailability routes nonperiodic calculations to the CPU streaming reader.

## Lengths and PDB export

CNDB coordinates are retained as recorded. Contact cutoffs and other analysis
length parameters must use the same numerical length scale as those coordinates.
Reduced-unit simulations do not acquire a physical calibration from this reader.

PDB is a separate export boundary: OpenMM positions and box vectors are both
converted to angstroms before writing. Thus a context coordinate of 1 nm is
written as 10 angstroms, and a context box length of 10 nm as 100 angstroms.
The coordinate-to-box ratio is preserved. See the [wwPDB coordinate specification](https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM).

## Nadam

`method="nadam"` implements Dozat's Algorithm 8 with constant momentum
`mu_t=beta1`, without a warm-up schedule:

```text
m_t = beta1*m_(t-1) + (1-beta1)*gradient_t
v_t = beta2*v_(t-1) + (1-beta2)*gradient_t^2
direction_t = beta1*m_t/(1-beta1^(t+1))
              + (1-beta1)*gradient_t/(1-beta1^t)
vhat_t = v_t/(1-beta2^t)
w_t = w_(t-1) - eta_t*direction_t/(sqrt(vhat_t)+epsilon)
```

Reference: Timothy Dozat, *Incorporating Nesterov Momentum into Adam* (2015),
[Algorithm 8](https://cs229.stanford.edu/proj2015/054_report.pdf). The source
comment records the same reference. This corrects the previous update rule;
continuing from old moment arrays applies the corrected rule on subsequent steps.
When loading moment arrays, supply the next iteration through the existing `it`
constructor argument.

RMSProp retains `rho=beta1`. Cosine learning rates retain their periodic
continuation after the first minimum at `scheduler_T_max`. Neither convention
is changed by this bug-fix set. Optimizer calls leave caller-owned contact
matrices unchanged, including read-only arrays.

## Stability and output ownership

Stability checks read the current active-integrator `kbT`, or the builtin
integrator's current temperature. Dimensionless thresholds normally use energy
per particle divided by `1.5*RT`. At zero temperature, only finiteness is checked
unless a positive `stability_reference_energy` is supplied to `simulation_setup`
(or `reference_energy` to `StabilityReporter`), in kJ/mol per particle.

Overdamped integrators use potential-energy diagnostics and do not reset
velocities or apply kinetic-temperature thresholds. For the current active
Brownian model, context velocities do not enter its position update. An unknown
custom integrator needs an explicit reference energy. Non-finite monitored
energies are written to the stability report and raise `FloatingPointError`.

`write_logs=False` disables the log file; it allows trajectory, energy, stability,
and force-field outputs. A run with every output disabled creates no output
directory. Each simulation owns independent logging handlers.

`run(...,report=False)` temporarily detaches all simulation reporters and restores
the original list on success or failure, preserving prior pause states.
`run()` returns `None`. `save_reports()` finalizes and detaches owned report files;
external reporters remain attached and open. `close()` also closes owned log
handlers. Both finalization methods are safe to call repeatedly.


## Analysis acceleration, writhe, and GPU batching

MSD (`Analyzer.compute_msd` and `Analyzer.compute_msd_average`) shifts each bead
trajectory by its initial frame in `float64` before FFT and prefix-sum
evaluation, accumulates `S1` in `float64`, enforces `msd[0] = 0`, and pads
transforms with `scipy.fft.next_fast_len` while combining all three spatial power
spectra before a single inverse rFFT. Spatial velocity correlation
(`Analyzer.calculate_spatial_vel_corr`) shares a single vectorized CPU/GPU
implementation using `searchsorted` against the exact `float32` bin edges and
flattened `bincount` accumulation, preserving identical `NaN` behavior for empty
bins.

`Analyzer.compute_writhe_single_curve`, `Analyzer.compute_writhe_between_curves`,
and `Analyzer.compute_writhe_trajectory` default to `method="vectorized"` with
degenerate zero-normal guards and open-curve (`closed=False`) segment bounds;
passing `method="legacy"` selects the original pairwise scalar implementation.

Hi-C GPU calculation (`_calc_HiC_from_traj_array_gpu` and the streaming CUDA
CNDB entry points) transfers at most `batch_size` frames to the device at a time
and accumulates contact probabilities in `float64` before dividing by frame count.
