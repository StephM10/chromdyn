#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

from openmm import LangevinIntegrator, BrownianIntegrator, CustomIntegrator, Integrator
from .utilities import LogManager


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
                    "These parameters are per dof variables and can be set any time using .set_active_params(F_seq, tau_seq)"
                )
                # self.logger.info("integrator.setPerDofVariableByName('t_corr', [Vec3(t,t,t) for t in tau_list]).")
                self.is_active = True
                return ActiveLangevinIntegrator(
                    self.temperature, self.friction, self.timestep
                )

            elif integrator == "active-brownian":
                self.logger.info(
                    f"Active BrownianIntegrator: temperature={self.temperature} | friction={self.friction} | timestep={self.timestep}"
                )
                self.logger.info("Initialized active parameters: F=0.0 and t_corr=1.0.")
                self.logger.info(
                    "These parameters are per dof variables and can be set any time using .set_active_params(F_seq, tau_seq)"
                )
                # self.logger.info("integrator.setPerDofVariableByName('t_corr', [Vec3(t,t,t) for t in tau_list]).")
                self.is_active = True
                return ActiveBrownianIntegrator(
                    self.temperature, self.friction, self.timestep
                )

        except Exception as e:
            self.logger.exception(f"[ERROR] Failed to create integrator: {e}")
            raise


# -------------------------------------------------------------------
# ActiveBrownianIntegrator: Custom Integrator Class
# -------------------------------------------------------------------


class ActiveBrownianIntegrator(CustomIntegrator):
    R"""
    The :class:`~.ActiveBrownianIntegrator` class is a custom Brownian integrator. Just like all Brownian integrators there are no velocities, there are only forces and displacements.
    Here we use the velocities as a proxy to keep track of the active force for each monomer.
    Hence velocity v of a monomer represents the active force f = gamma * v.
    """

    def __init__(
        self,
        temperature=120.0,
        collision_rate=1.0,
        timestep=0.001,
        corr_time=10.0,
        constraint_tolerance=1e-8,
    ):

        # Create a new CustomIntegrator
        super().__init__(timestep)

        # add global variables
        kbT = 0.008314 * temperature
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        # self.addGlobalVariable("t_corr", corr_time)
        self.addPerDofVariable("t_corr", corr_time)
        self.addPerDofVariable("F_act", 0)
        self.setConstraintTolerance(constraint_tolerance)

        # add attributes
        self.collisionRate = collision_rate
        self.corrTime = corr_time
        self.temperature = temperature

        self.addPerDofVariable("x1", 0)  # for constraints
        self.addUpdateContextState()

        # update velocities. note velocities represent the active force and only depend on their value in previous time step
        # IMPORTANT: the force-group "0" is associated with keeping track of the active noise
        self.addComputePerDof(
            "v",
            "(exp(- dt / t_corr ) * v) + ((sqrt(1 - exp( - 2 * dt / t_corr)) * F_act / g) * gaussian)",
        )
        self.addConstrainVelocities()

        # compute position update
        # note that the update computed below *OVER* estimates the contribution from the force-group "0" or the active force
        # the intended contribution of the active force is in the (v * dt) term, but the (f * dt / g) term also has
        # the contribution of force-group 0, which needs to be taken out -- as is done in the following line
        self.addComputePerDof(
            "x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kbT / g) * dt) * gaussian)"
        )
        #self.addComputePerDof("x", "x - (dt  * F_act / g)")

        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained


class ActiveLangevinIntegrator(CustomIntegrator):
    def __init__(self, temperature, gamma, dt, corr_time=1.0):
        super().__init__(dt)

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
        self.addComputeGlobal("c", "exp(-gamma*dt)")
        self.addComputeGlobal("one_minus_c2", "1 - c*c")

        self.addUpdateContextState()

        # ---- Update active force p via exact OU ----
        # p <- e^{-dt/τ} p + sqrt(1 - e^{-2dt/τ}) * σ_p * ξ
        self.addComputePerDof(
            "p",
            "exp(-halfdt/t_corr)*p + sqrt(1 - exp(-2*halfdt/t_corr)) * F_act * gaussian",
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
            "exp(-halfdt/t_corr)*p + sqrt(1 - exp(-2*halfdt/t_corr)) * F_act * gaussian",
        )
