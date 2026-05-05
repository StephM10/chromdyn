#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

import time
import numpy as np
from pathlib import Path
from openmm import System, Vec3
from openmm.app import Simulation, Topology
import openmm.unit as unit
from numpy.typing import NDArray
from importlib import metadata
from .platforms import PlatformManager
from .integrators import IntegratorManager
from .forcefield import ForceFieldManager
from .utilities import config_generator, LogManager
from .reporters import SaveStructure, StabilityReporter, EnergyReporter, ForceFieldReporter
from typing import Optional, Tuple, List, Dict, Union, Any


class ChromatinDynamics:
    """
    Chromatin Dynamics simulation using OpenMM with modular managers:
    - System setup, forces, integrators, analysis, and logging.
    """

    def __init__(
        self,
        topology: Topology,
        name: str = "ChromatinDynamics",
        platform_name: str = "CUDA",
        output_dir: str = "output",
        console_stream: bool = True,
        mass: Union[float, List[float], NDArray] = 1.0,
        write_logs: bool = True,
    ) -> None:

        self.name = name
        self.version = _get_version()
        if write_logs:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.logger = LogManager(
                log_file=self.output_dir / f"{name}.log"
            ).get_logger(__name__, console=console_stream)
        else:
            self.output_dir = None
            self.logger = LogManager().get_logger(__name__, console=console_stream)

        self.logger.info("*" * 60)
        self.logger.info(f"{f'chromdyn v{self.version} : Chromatin Dynamics':^60}")
        # self.logger.info(f"{'':^60}")
        self.logger.info("*" * 60)

        self.topology = topology
        self.system = System()
        self.num_particles = topology.getNumAtoms()
        
        # Handle scalar or array-like mass
        if np.isscalar(mass):
            self.masses = np.full(self.num_particles, float(mass))
        else:
            self.masses = np.asarray(mass, dtype=float)
            if len(self.masses) != self.num_particles:
                raise ValueError(f"Length of mass array ({len(self.masses)}) must match num_particles ({self.num_particles})")

        # Optional: check mass consistency for atom types in topology
        atom_types_dict = {}
        for idx, atom in enumerate(self.topology.atoms()):
            atype = atom.element
            if atype not in atom_types_dict:
                atom_types_dict[atype] = []
            atom_types_dict[atype].append(self.masses[idx])

        for atype, m_list in atom_types_dict.items():
            if len(set(m_list)) > 1:
                self.logger.warning(
                    f"Atom type '{atype}' has inconsistent masses: {np.unique(m_list)}"
                )

        # Add particles to the system
        for m in self.masses:
            self.system.addParticle(float(m))

        self.logger.info(
            f"System initialized with {self.num_particles} particles. Output directory: {self.output_dir}"
        )

        self.platform_manager = PlatformManager(logger=self.logger)
        self.platform_manager.set_platform(platform_name)

        self.force_field_manager = ForceFieldManager(
            self.topology, self.system, logger=self.logger, Nonbonded_cutoff=5.0
        )
        self.logger.info(
            "force_field_manager initialized. Use this to add forces before running setup."
        )

        self.simulation = None
        self.reporters: dict[str, object] = {}

    def simulation_setup(
        self,
        init_struct: Union[str, NDArray] = "randomwalk",
        PBC: bool = False,
        box_vectors: Optional[
            Tuple[
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
            ]
        ] = None,
        integrator: str = "langevin",
        temperature: float = 120.0,
        timestep: float = 0.01,
        friction: float = 0.1,
        save_pos: bool = True,
        save_energy: bool = True,
        save_forcefield: bool = True,
        stability_check: bool = True,
        stability_report_interval: int = 500,
        energy_report_interval: int = 1000,
        pos_report_interval: int = 1000,
        force_reinitialize: bool = True,
    ) -> None:
        """Sets up the integrator, platform, context, and attaches reporters."""

        # ---- Periodic boundary conditions (optional) ----
        if PBC:
            assert (
                box_vectors is not None
            ), "When PBC=True, supply box_vectors=((Lx,0,0),(0,Ly,0),(0,0,Lz)) in nanometers."
            a = Vec3(*box_vectors[0])
            b = Vec3(*box_vectors[1])
            c = Vec3(*box_vectors[2])
            # Set the periodic box on the System BEFORE creating the Context
            self.system.setDefaultPeriodicBoxVectors(a, b, c)
            # Ensure all existing/future nonbonded forces use CutoffPeriodic
            self.force_field_manager.set_nonbonded_method("Periodic")
        else:
            self.force_field_manager.set_nonbonded_method("NonPeriodic")

        self.integrator_manager = IntegratorManager(
            integrator=integrator,
            temperature=temperature,
            friction=friction,
            logger=self.logger,
            timestep=timestep,
        )

        self.simulation = Simulation(
            self.topology,
            self.system,
            self.integrator_manager.integrator,
            self.platform_manager.get_platform(),
        )

        self.logger.info("Setting up simulation context...")
        if isinstance(init_struct, str):
            gen_config = config_generator()
            positions, msg = gen_config.get_config(
                mode=init_struct, num_steps=self.num_particles
            )

        else:
            assert init_struct.shape == (
                self.num_particles,
                3,
            ), "init_struct shape != (num_particles,3)"
            positions, msg = init_struct, "User defined initial configuration loaded"

        # Wrap initial coordinates into the primary cell (keeps bonds near minimum image)
        if PBC:
            # Get box lengths from the System (nm)
            a_vec, b_vec, c_vec = self.system.getDefaultPeriodicBoxVectors()
            # Lx, Ly, Lz = a_vec.x, b_vec.y, c_vec.z

            pos = np.asarray(positions, dtype=float)
            # pos[:, 0] = pos[:, 0] % Lx
            # pos[:, 1] = pos[:, 1] % Ly
            # pos[:, 2] = pos[:, 2] % Lz
            positions = pos

        self.simulation.context.setPositions(positions)

        self.logger.info(msg)
        self.logger.info("Simulation context initialized.")
        self.print_force_info()

        if save_pos:
            path = self.output_dir / f"{self.name}_positions.cndb"
            self.reporters["position"] = SaveStructure(
                path,
                reportInterval=pos_report_interval,
                topology=self.topology,
                PBC=PBC,
            )
            self.simulation.reporters.append(self.reporters["position"])
            self.logger.info(f"Position reporter created: {path}")

        if save_energy:
            path = self.output_dir / f"{self.name}_energy_report.txt"
            self.reporters["energy"] = EnergyReporter(
                path,
                self.force_field_manager,
                reportInterval=energy_report_interval,
                reportForceGrp=True,
            )
            self.simulation.reporters.append(self.reporters["energy"])
            self.logger.info(f"Energy reporter created: {path}")

        if stability_check:
            path = self.output_dir / f"{self.name}_stability_report.txt"
            self.reporters["stability"] = StabilityReporter(
                path,
                reportInterval=stability_report_interval,
                logger=self.logger,
                kinetic_threshold=5.0,
                potential_threshold=5.0,
                force_reinitialize=force_reinitialize,
            )
            self.simulation.reporters.append(self.reporters["stability"])
            self.logger.info(f"Stability reporter created: {path}")

        if save_forcefield and self.output_dir is not None:
            path = self.output_dir / f"{self.name}_forcefield.txt"
            self.reporters["forcefield"] = ForceFieldReporter(
                path,
                force_field_manager=self.force_field_manager,
                integrator_manager=self.integrator_manager,
                system_info={
                    "name": self.name,
                    "num_particles": self.num_particles,
                    "mass": self.masses,
                    "PBC": PBC,
                    "box_vectors": box_vectors,
                },
            )
            self.logger.info(f"Force-field reporter created: {path}")

    def set_activity(self, F_seq: List, tau_seq: List):

        assert hasattr(
            self, "simulation"
        ), "Simulation object not set up! Call .simulation_setup() first!"
        assert (
            self.integrator_manager.is_active
        ), "Choose active-langevin integrator while setting up simulation."
        assert (
            len(F_seq) == self.num_particles
        ), "F_seq length differs from the number of system particles"
        assert (
            len(tau_seq) == self.num_particles
        ), "tau_seq length differs from the number of system particles"

        self.integrator_manager.integrator.setPerDofVariableByName(
            "t_corr", [Vec3(t, t, t) for t in tau_seq]
        )
        self.integrator_manager.integrator.setPerDofVariableByName(
            "F_act", [Vec3(xx, xx, xx) for xx in F_seq]
        )

        self.logger.info(
            f"Added active force and correlation times for {len(F_seq)} particles."
        )

        if "forcefield" in self.reporters:
            self.reporters["forcefield"].update_activity(F_seq, tau_seq)

    def get_active_params(self, mono_id: int = None) -> Dict[str, Any]:
        assert (
            self.integrator_manager.is_active
        ), "Activity not present in the simulation."
        ret = {}
        if mono_id is None:
            ret["t_corr"] = self.integrator_manager.integrator.getPerDofVariableByName(
                "t_corr"
            )
            ret["F_act"] = self.integrator_manager.integrator.getPerDofVariableByName(
                "F_act"
            )
        else:
            assert (
                mono_id < self.num_particles
            ), f"inconsistent mono_id, system has {self.num_particles} particles"
            ret["t_corr"] = self.integrator_manager.integrator.getPerDofVariableByName(
                "t_corr"
            )[mono_id]
            ret["F_act"] = self.integrator_manager.integrator.getPerDofVariableByName(
                "F_act"
            )[mono_id]
        return ret

    def run(self, n_steps: int, verbose: bool = True, report: bool = True) -> float:
        """Runs the simulation and reports performance."""
        if not self.simulation:
            raise RuntimeError(
                "Simulation not initialized. Call simulation_setup() first."
            )

        if verbose:
            self.logger.info("-" * 60)
            self.logger.info(f"Running simulation for {n_steps} steps...")

        if not report:
            self.pause_reporters()

        start = time.time()
        try:
            self.simulation.step(n_steps)
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise

        elapsed = time.time() - start
        steps_per_sec = n_steps / elapsed

        if verbose:
            self.logger.info(
                f"Completed {n_steps} steps in {elapsed:.2f}s ({steps_per_sec:.0f} steps/s)"
            )
            self.logger.info("-" * 60)

        if not report:
            self.resume_reporters()

    def pause_reporters(self) -> None:
        for name, reporter in self.reporters.items():
            if hasattr(reporter, "pause"):
                reporter.pause()
                self.logger.info(f"Paused reporter: {name}")

    def resume_reporters(self) -> None:
        for name, reporter in self.reporters.items():
            if hasattr(reporter, "resume"):
                reporter.resume()
                self.logger.info(f"Resumed reporter: {name}")

    def save_reports(self) -> None:
        if "position" in self.reporters and hasattr(
            self.reporters["position"], "close"
        ):
            self.reporters["position"].close()
            self.logger.info("Closed position reporter file.")

    def print_force_info(self) -> None:
        """Logs a formatted summary of forces and per-particle energy."""
        context = self.simulation.context
        system = self.simulation.system
        num_particles = system.getNumParticles()

        self.logger.info("-" * 120)
        self.logger.info(
            f"{'Index':<6} {'Force Class':<30} {'Force Name':<20} {'Group':<8} "
            f"{'Particles':<12} {'Bonds':<12} {'Exclusions':<12} {'P.E./Particle':<20}"
        )
        self.logger.info("-" * 120)

        for i, force in enumerate(system.getForces()):
            group = force.getForceGroup()
            force_class = force.__class__.__name__
            force_name = self.force_field_manager.force_name_map.get(i, "Unnamed")
            state = context.getState(getEnergy=True, groups={group})
            pot_energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            per_particle_energy = pot_energy / num_particles if num_particles else 0.0

            num_particles_force = getattr(force, "getNumParticles", lambda: "N/A")()
            num_bonds_force = getattr(force, "getNumBonds", lambda: "N/A")()
            num_exclusions = getattr(force, "getNumExclusions", lambda: "N/A")()

            self.logger.info(
                f"{i:<6} {force_class:<30} {force_name:<20} {group:<8} "
                f"{num_particles_force:<12} {num_bonds_force:<12} {num_exclusions:<12} {per_particle_energy:<20.3f}"
            )

        total_state = context.getState(getEnergy=True)
        pot_energy = total_state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        kin_energy = total_state.getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        self.logger.info("-" * 120)
        self.logger.info(
            f"Total particles: {num_particles} | K.E./particle: {kin_energy/num_particles:.3f} | P.E./particle: {pot_energy/num_particles:.3f}"
        )
        self.logger.info("-" * 120)

    def __repr__(self) -> str:
        return f"<ChromatinDynamics(name={self.name}, particles={self.num_particles}, output_dir={self.output_dir})>"


def _get_version() -> str:
    try:
        return metadata.version("chromdyn")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"
