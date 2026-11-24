#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

import numpy as np
from openmm import (
    LangevinIntegrator, BrownianIntegrator, CustomIntegrator, Integrator, Vec3, unit,
)
from .utilities import LogManager


def _validate_scalar(name, value, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}")


def _validate_active_parameter(name, values):
    if name not in {"t_corr", "F_act"}:
        return
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or (name == "t_corr" and np.any(values <= 0)):
        raise ValueError(f"{name} must be finite" + (" and positive" if name == "t_corr" else ""))


# -------------------------------------------------------------------
# Integrator Manager: Creates integrators for Brownian or Langevin dynamics
# -------------------------------------------------------------------
class IntegratorManager:
    # Note: for langevin cases, the friction is actually the friction_phys / mass
    VALID_INTEGRATORS = [
        "langevin",
        "brownian",
        "active-langevin",
        "active-brownian",
    ]  # List of supported integrators
    DEFAULT_PARAMS = {
        "temperature": 300.0,
        "friction": 0.1,
        "timestep": 0.01,
    }

    def __init__(self, logger=None, integrator="langevin", **kwargs):
        """
        Initialize the IntegratorManager with flexible parameters.

        Args:
            integrator (str): Type of integrator ('brownian', 'langevin', 'active').
            logger (Logger, optional): Logger instance. If None, initializes a default logger.
            **kwargs: Optional parameters (temperature, friction, timestep, tcorr, Fact).

        Friction convention: builtin Brownian/Langevin and active Langevin use
        gamma (inverse time), corresponding to physical drag m*gamma. Active
        Brownian uses physical drag g directly: dx=(f+p)*dt/g+sqrt(2*kbT*dt/g)*xi.
        Mass does not enter the active-Brownian position update.
        """
        self.logger = logger or LogManager().get_logger(__name__)
        self.logger.info("Creating integrator ...")
        self.is_active = False
        if isinstance(integrator, str):
            assert (
                integrator in self.VALID_INTEGRATORS
            ), "integrator name does not exist"
            self.integrator_name = integrator

            self.temperature = kwargs.get(
                "temperature", self.DEFAULT_PARAMS["temperature"]
            )
            self.friction = kwargs.get("friction", self.DEFAULT_PARAMS["friction"])
            self.timestep = kwargs.get("timestep", self.DEFAULT_PARAMS["timestep"])
            self.advance_time = kwargs.get("advance_time", False)
            self.precompute_coefficients = kwargs.get("precompute_coefficients", True)
            self.store_thermal_kick = kwargs.get("store_thermal_kick", False)
            self.store_unconstrained_positions = kwargs.get(
                "store_unconstrained_positions", False
            )

            self.integrator = self.create_integrator(self.integrator_name)

        elif isinstance(integrator, Integrator):
            self.integrator_name = "custom"
            self.integrator = integrator
            self.logger.info("Created custom integrator")
        # self.logger.info('-'*60)

    def create_integrator(self, integrator: str):
        """
        Create and return an OpenMM integrator based on initialized settings.

        Returns:
            OpenMM Integrator object.
        """
        try:
            if integrator == "brownian":
                # self.logger.info(f"Creating {integrator} ...")
                self.logger.info(
                    f"BrownianIntegrator: temperature={self.temperature} | friction={self.friction} | timestep={self.timestep}"
                )
                return BrownianIntegrator(
                    self.temperature, self.friction, self.timestep
                )

            elif integrator == "langevin":
                # self.logger.info(f"Creating {integrator} ...")
                self.logger.info(
                    f"LangevinIntegrator: temperature={self.temperature} | friction={self.friction} | timestep={self.timestep}"
                )
                return LangevinIntegrator(
                    self.temperature, self.friction, self.timestep
                )

            elif integrator == "active-langevin":
                self.logger.info(
                    f"Active LangevinIntegrator: temperature={self.temperature} | friction={self.friction} | timestep={self.timestep}"
                )
                self.logger.info("Initialized active parameters: F=0.0 and t_corr=1.0.")
                self.logger.info(
                    "These parameters are per dof variables and can be set any time using .set_activity(F_seq, tau_seq)"
                )
                # self.logger.info("integrator.setPerDofVariableByName('t_corr', [Vec3(t,t,t) for t in tau_list]).")
                self.is_active = True
                return ActiveLangevinIntegrator(
                    self.temperature, self.friction, self.timestep, advance_time=self.advance_time
                )

            elif integrator == "active-brownian":
                self.logger.info(
                    f"Active BrownianIntegrator: temperature={self.temperature} | friction={self.friction} | timestep={self.timestep}"
                )
                self.logger.info("Initialized active parameters: F=0.0 and t_corr=1.0.")
                self.logger.info(
                    "These parameters are per dof variables and can be set any time using .set_activity(F_seq, tau_seq)"
                )
                # self.logger.info("integrator.setPerDofVariableByName('t_corr', [Vec3(t,t,t) for t in tau_list]).")
                self.is_active = True
                return ActiveBrownianIntegrator(
                    self.temperature,
                    self.friction,
                    self.timestep,
                    advance_time=self.advance_time,
                    store_unconstrained_positions=self.store_unconstrained_positions,
                    precompute_coefficients=self.precompute_coefficients,
                    store_thermal_kick=self.store_thermal_kick,
                )

        except Exception as e:
            self.logger.exception(f"[ERROR] Failed to create integrator: {e}")
            raise


# -------------------------------------------------------------------
# ActiveBrownianIntegrator: Custom Integrator Class
# -------------------------------------------------------------------


class ActiveBrownianIntegrator(CustomIntegrator):
    R"""
    Overdamped Brownian dynamics with a separate per-DOF active OU force ``p``.

    ``collision_rate`` is physical drag g (mass/time in OpenMM numerical units),
    rather than the inverse-time gamma of OpenMM BrownianIntegrator. The latter
    uses g=m*gamma. Particle mass does not affect this integrator's displacement.

    ``precompute_coefficients=True`` caches the OU and Brownian coefficients.
    Explicit activity, step-size, ``g``, or ``kbT`` setters refresh the cache;
    ``refresh_coefficients()`` is also available after low-level state changes.
    Set ``store_unconstrained_positions=True`` to save pre-constraint positions
    in ``x1``. This optional snapshot is off by default to avoid an extra GPU
    update kernel; position constraints are applied in either mode.
    With store_thermal_kick, thermal_kick holds the exact stochastic displacement
    used in the most recent position update, before constraints.
    """

    def __init__(
        self,
        temperature=120.0,
        collision_rate=1.0,
        timestep=0.001,
        corr_time=10.0,
        constraint_tolerance=1e-8,
        advance_time=False,
        store_unconstrained_positions=False,
        precompute_coefficients=True,
        store_thermal_kick=False,
    ):
        super().__init__(timestep)
        self._validate_scalar("timestep", self.getStepSize().value_in_unit(unit.picosecond))
        self._validate_scalar("friction", collision_rate)
        self._validate_scalar("correlation time", corr_time)
        self._validate_scalar("temperature", temperature, allow_zero=True)
        self.precompute_coefficients = precompute_coefficients
        self._coefficient_cache_ready = False

        # advance_time=True requires a Context parameter 't', typically supplied
        # by a time-dependent force such as OscillatoryTweezer.
        self.advance_time_flag = advance_time
        self.store_unconstrained_positions = store_unconstrained_positions
        self.addGlobalVariable("kbT", 0.008314 * temperature)
        self.addGlobalVariable("g", collision_rate)
        self.addPerDofVariable("t_corr", corr_time)
        self.addPerDofVariable("p", 0)
        self.addPerDofVariable("F_act", 0)
        self.setConstraintTolerance(constraint_tolerance)
        self.collisionRate = collision_rate
        self.corrTime = corr_time
        self.temperature = temperature

        if self.precompute_coefficients:
            self.addGlobalVariable("bd_drift", 0)
            self.addGlobalVariable("bd_noise", 0)
            self.addPerDofVariable("ou_decay", 0)
            self.addPerDofVariable("ou_noise", 0)
            self.refresh_coefficients()
        self._coefficient_cache_ready = True

        self.addPerDofVariable("x1", 0)  # optional pre-constraint snapshot
        self.addUpdateContextState()
        if self.precompute_coefficients:
            self.addComputePerDof("p", "ou_decay * p + ou_noise * gaussian")
            drift = "bd_drift * (f + p)"
            thermal = "bd_noise * gaussian"
        else:
            self.addComputePerDof(
                "p",
                "exp(-dt / t_corr) * p + sqrt(1 - exp(-2 * dt / t_corr)) * F_act * gaussian",
            )
            drift = "dt * (f + p) / g"
            thermal = "sqrt(2 * kbT * dt / g) * gaussian"
        if store_thermal_kick:
            self.addPerDofVariable("thermal_kick", 0)
            self.addComputePerDof("thermal_kick", thermal)
            thermal = "thermal_kick"
        self.addComputePerDof("x", f"x + {drift} + {thermal}")
        if self.store_unconstrained_positions:
            self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        if self.advance_time_flag:
            self.addComputeGlobal("t", "t + dt")

    def refresh_coefficients(self):
        """Refresh cached coefficients without changing positions or OU state.

        Normal setters call this automatically. Call it explicitly after
        bypassing the Python setters to modify the underlying OpenMM object.
        It is a no-op when precomputation is disabled.
        """
        if not self.precompute_coefficients:
            return
        dt = self.getStepSize().value_in_unit(unit.picosecond)
        g = self.getGlobalVariableByName("g")
        kbT = self.getGlobalVariableByName("kbT")
        tau = np.asarray(self.getPerDofVariableByName("t_corr"), dtype=float)
        amplitude = np.asarray(self.getPerDofVariableByName("F_act"), dtype=float)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Brownian timestep must be finite and positive")
        if not np.isfinite(g) or g <= 0:
            raise ValueError("Brownian friction g must be finite and positive")
        if not np.isfinite(kbT) or kbT < 0:
            raise ValueError("Brownian kbT must be finite and nonnegative")
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("Active correlation times must be finite and positive")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("Active force amplitudes must be finite")

        decay = np.exp(-dt / tau)
        # expm1 avoids cancellation for the common small-dt/long-tau regime.
        noise = amplitude * np.sqrt(-np.expm1(-2 * dt / tau))
        super().setPerDofVariableByName("ou_decay", [Vec3(*row) for row in decay])
        super().setPerDofVariableByName("ou_noise", [Vec3(*row) for row in noise])
        super().setGlobalVariableByName("bd_drift", float(dt / g))
        super().setGlobalVariableByName("bd_noise", float(np.sqrt(2 * kbT * dt / g)))

    def set_activity(self, F_seq, tau_seq):
        """Set particle activity and refresh once after both inputs are updated."""
        amplitudes = np.asarray(F_seq, dtype=float)
        times = np.asarray(tau_seq, dtype=float)
        if amplitudes.ndim != 1 or times.shape != amplitudes.shape or not times.size:
            raise ValueError("Activity and correlation times must be nonempty 1D arrays of equal length")
        self._validate_per_dof_parameter("F_act", amplitudes)
        self._validate_per_dof_parameter("t_corr", times)
        super().setPerDofVariableByName(
            "t_corr", [Vec3(t, t, t) for t in times]
        )
        super().setPerDofVariableByName(
            "F_act", [Vec3(f, f, f) for f in amplitudes]
        )
        self.refresh_coefficients()

    @staticmethod
    def _validate_scalar(name, value, allow_zero=False):
        _validate_scalar(f"Brownian {name}", value, allow_zero)

    @staticmethod
    def _validate_per_dof_parameter(name, values):
        _validate_active_parameter(name, values)

    def _validate_global_parameter(self, name, value):
        if name == "g":
            self._validate_scalar("friction", value)
        elif name == "kbT":
            self._validate_scalar("kbT", value, allow_zero=True)

    def _refresh_after_parameter_change(self, name):
        if (
            getattr(self, "_coefficient_cache_ready", False)
            and self.precompute_coefficients
            and name in {"dt", "g", "kbT", "t_corr", "F_act"}
        ):
            self.refresh_coefficients()

    def setStepSize(self, size):
        value = size.value_in_unit(unit.picosecond) if unit.is_quantity(size) else size
        self._validate_scalar("timestep", value)
        super().setStepSize(size)
        self._refresh_after_parameter_change("dt")

    def setPerDofVariable(self, index, values):
        self._validate_per_dof_parameter(self.getPerDofVariableName(index), values)
        super().setPerDofVariable(index, values)
        self._refresh_after_parameter_change(self.getPerDofVariableName(index))

    def setPerDofVariableByName(self, name, values):
        self._validate_per_dof_parameter(name, values)
        super().setPerDofVariableByName(name, values)
        self._refresh_after_parameter_change(name)

    def setGlobalVariable(self, index, value):
        self._validate_global_parameter(self.getGlobalVariableName(index), value)
        super().setGlobalVariable(index, value)
        self._refresh_after_parameter_change(self.getGlobalVariableName(index))

    def setGlobalVariableByName(self, name, value):
        self._validate_global_parameter(name, value)
        super().setGlobalVariableByName(name, value)
        self._refresh_after_parameter_change(name)


class ActiveLangevinIntegrator(CustomIntegrator):
    def __init__(self, temperature, gamma, dt, corr_time=1.0, advance_time=False):
        temperature = temperature.value_in_unit(unit.kelvin) if unit.is_quantity(temperature) else temperature
        gamma = gamma.value_in_unit(unit.picosecond**-1) if unit.is_quantity(gamma) else gamma
        dt = dt.value_in_unit(unit.picosecond) if unit.is_quantity(dt) else dt
        corr_time = corr_time.value_in_unit(unit.picosecond) if unit.is_quantity(corr_time) else corr_time
        _validate_scalar("temperature", temperature, allow_zero=True)
        _validate_scalar("gamma", gamma, allow_zero=True)
        _validate_scalar("timestep", dt)
        _validate_scalar("correlation time", corr_time)
        super().__init__(dt)

        # WARNING: advance_time=True assumes the Context has a global parameter 't'.
        # This is typically added by a CustomExternalForce (e.g., OscillatoryTweezer).
        # If 't' is not defined in any force, OpenMM will raise an Unknown Variable exception.
        self.advance_time_flag = advance_time

        # Globals
        self.addGlobalVariable("kbT", 0.008314 * temperature)  # kJ/mol
        self.addGlobalVariable("gamma", gamma)  # friction (1/ps)
        self.addGlobalVariable("c", 0)  # exp(-gamma*dt), set each step
        self.addGlobalVariable("one_minus_c2", 0)  # 1 - c^2
        self.addGlobalVariable("halfdt", 0.5 * dt)

        # Per-DOF
        self.addPerDofVariable("t_corr", corr_time)  # per-particle correlation time
        self.addPerDofVariable("p", 0)  # active force (NOT a velocity)
        self.addPerDofVariable("x1", 0)  # for constraints

        # Optional: per-DOF active strength σ_p (RMS of p in steady state)
        self.addPerDofVariable("F_act", 0)  # set from Python per particle (Vec3)

        # Precompute c each step
        # Keep the existing global layout, but derive halfdt from the live dt.
        self.addComputeGlobal("halfdt", "0.5*dt")
        self.addComputeGlobal("c", "exp(-gamma*dt)")
        # Equivalent to -expm1(-2*gamma*dt); expm1 is not a Lepton function.
        self.addComputeGlobal("one_minus_c2", "(1+c*c)*tanh(gamma*dt)")

        self.addUpdateContextState()

        # ---- Update active force p via exact OU ----
        # p <- e^{-dt/τ} p + sqrt(1 - e^{-2dt/τ}) * σ_p * ξ
        self.addComputePerDof(
            "p",
            "exp(-halfdt/t_corr)*p + sqrt((1+exp(-2*halfdt/t_corr))*tanh(halfdt/t_corr)) * F_act * gaussian",
        )

        # ---- First half kick with conservative + active ----
        self.addComputePerDof("v", "v + (f + p) * (halfdt / m)")

        # ---- Drift ----
        self.addComputePerDof("x", "x + v * dt")

        # Enforce position constraints
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x - x1)/dt")

        # ---- Langevin thermostat (Ornstein–Uhlenbeck on v) ----
        # v <- c v + sqrt((1 - c^2) * kbT/m) * ξ
        self.addComputePerDof("v", "c * v + sqrt(one_minus_c2 * kbT / m) * gaussian")

        # Enforce velocity constraints (projects only v; p is untouched)
        self.addConstrainVelocities()

        # ---- Second half kick ----
        self.addComputePerDof("v", "v + (f + p) * (halfdt / m)")
        self.addConstrainVelocities()

        self.addComputePerDof(
            "p",
            "exp(-halfdt/t_corr)*p + sqrt((1+exp(-2*halfdt/t_corr))*tanh(halfdt/t_corr)) * F_act * gaussian",
        )

        # Advance global time at the end of the step
        if self.advance_time_flag:
            self.addComputeGlobal("t", "t + dt")

    def setStepSize(self, size):
        value = size.value_in_unit(unit.picosecond) if unit.is_quantity(size) else size
        _validate_scalar("timestep", value)
        super().setStepSize(size)

    def setPerDofVariable(self, index, values):
        _validate_active_parameter(self.getPerDofVariableName(index), values)
        super().setPerDofVariable(index, values)

    def setPerDofVariableByName(self, name, values):
        _validate_active_parameter(name, values)
        super().setPerDofVariableByName(name, values)

    def _validate_global_parameter(self, name, value):
        if name in {"gamma", "kbT"}:
            _validate_scalar(name, value, allow_zero=True)

    def setGlobalVariable(self, index, value):
        self._validate_global_parameter(self.getGlobalVariableName(index), value)
        super().setGlobalVariable(index, value)

    def setGlobalVariableByName(self, name, value):
        self._validate_global_parameter(name, value)
        super().setGlobalVariableByName(name, value)


class ActiveBrownianIntegratorLegacy(CustomIntegrator):
    R"""
    Legacy backup of the original ActiveBrownianIntegrator implementation.

    This version uses the OpenMM state velocity ``v`` as a proxy for active
    force. It is kept here for reproducibility and comparison with older runs;
    IntegratorManager does not instantiate it by default.
    """

    def __init__(
        self,
        temperature=120.0,
        collision_rate=1.0,
        timestep=0.001,
        corr_time=10.0,
        constraint_tolerance=1e-8,
        advance_time=False,
    ):
        super().__init__(timestep)

        self.advance_time_flag = advance_time

        kbT = 0.008314 * temperature
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        self.addPerDofVariable("t_corr", corr_time)
        self.addPerDofVariable("F_act", 0)
        self.setConstraintTolerance(constraint_tolerance)

        self.collisionRate = collision_rate
        self.corrTime = corr_time
        self.temperature = temperature

        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()

        self.addComputePerDof(
            "v",
            "(exp(- dt / t_corr ) * v) + ((sqrt(1 - exp( - 2 * dt / t_corr)) * F_act / g) * gaussian)",
        )
        self.addConstrainVelocities()

        self.addComputePerDof(
            "x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kbT / g) * dt) * gaussian)"
        )

        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()

        if self.advance_time_flag:
            self.addComputeGlobal("t", "t + dt")
