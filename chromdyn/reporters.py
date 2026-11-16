#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *
import numpy as np
import h5py
from openmm.app import Simulation
from openmm import System, State
from openmm.app import Topology
import openmm.unit as unit
from openmm import CMMotionRemover
import os
from datetime import datetime
from .utilities import LogManager
from pathlib import Path
from typing import Union, Optional, Tuple
from .traj_utils import Analyzer


class SaveStructure:
    def __init__(
        self,
        report_file: Union[str, Path],
        PBC: bool,
        topology: Topology,
        reportInterval: int = 1000,
    ):
        self.filename: str = str(report_file)
        self.reportInterval: int = reportInterval
        self.topology = topology
        self.PBC = PBC
        mode: str = self.filename.split(".")[-1].lower()
        if mode not in ["cndb"]:
            raise ValueError(
                f"Unsupported file format: {mode}. Supported formats are 'cndb'."
            )
        self.mode: str = mode
        self.savestep: int = 0
        self.is_paused: bool = False

        if self.mode == "cndb":
            if os.path.exists(self.filename):
                backup_name: str = self.filename + ".bkp"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.filename, backup_name)

            self.saveFile: h5py.File = h5py.File(self.filename, "w")

        # Initialize static datasets
        self._save_types()
        self._save_full_topology()

    def _save_types(self):
        """Extracts atom types (from element) and saves as a dataset."""
        type_list = []
        for atom in self.topology.atoms():
            # Use element symbol as type if it's an Element object, else use string directly
            t_str = (
                atom.element if isinstance(atom.element, str) else atom.element.symbol
            )
            type_list.append(t_str)

        # Save as variable-length string dataset
        dt = h5py.special_dtype(vlen=str)
        self.saveFile.create_dataset("types", data=np.array(type_list, dtype=dt))

    def _save_full_topology(self):
        """
        Saves the OpenMM Topology directly to HDF5 datasets without using JSON.
        """
        if "topology" in self.saveFile:
            del self.saveFile["topology"]
        top_grp = self.saveFile.create_group("topology")

        # 1. extract atom data
        # use structured array (Structured Array) to store
        atom_dtype = np.dtype(
            [("name", "S20"), ("element", "S5"), ("res_idx", "i4"), ("chain_idx", "i4")]
        )

        chain_list = list(self.topology.chains())
        res_list = list(self.topology.residues())
        chain_to_idx = {c: i for i, c in enumerate(chain_list)}
        res_to_idx = {r: i for i, r in enumerate(res_list)}

        atoms_data = []
        for atom in self.topology.atoms():
            if atom.element is None:
                elem = "X"
            elif hasattr(atom.element, "symbol"):
                elem = atom.element.symbol  # This is an Element object
            else:
                elem = str(atom.element)
            atoms_data.append(
                (
                    atom.name.encode("utf-8"),
                    elem.encode("utf-8"),
                    res_to_idx[atom.residue],
                    chain_to_idx[atom.residue.chain],
                )
            )
        top_grp.create_dataset("atoms", data=np.array(atoms_data, dtype=atom_dtype))

        # 2. extract bond data
        atom_to_idx = {a: i for i, a in enumerate(self.topology.atoms())}
        bonds_data = [
            [atom_to_idx[b.atom1], atom_to_idx[b.atom2]] for b in self.topology.bonds()
        ]
        if bonds_data:
            top_grp.create_dataset("bonds", data=np.array(bonds_data, dtype="i4"))

        # 3. store metadata
        dt_str = h5py.special_dtype(vlen=str)
        chain_ids = [c.id for c in chain_list]
        res_names = [r.name for r in res_list]
        top_grp.create_dataset("chain_ids", data=np.array(chain_ids, dtype=dt_str))
        top_grp.create_dataset("res_names", data=np.array(res_names, dtype=dt_str))

    def close(self) -> None:
        self.saveFile.close()

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def describeNextReport(
        self, simulation: Simulation
    ) -> tuple[int, bool, bool, bool, bool]:
        """Get information about the next report this object will generate."""
        steps: int = self.reportInterval - simulation.currentStep % self.reportInterval
        return (
            steps,
            True,
            False,
            False,
            False,
        )  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """Generate a report."""
        if not self.is_paused:
            data: np.ndarray = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            if self.mode == "cndb":
                self.saveFile[str(self.savestep)] = np.array(data)

                if self.PBC:
                    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer
                    )  # Parameter enforcePeriodicbox in integrators is set to False by default
                    self.saveFile[str(self.savestep)].attrs["box"] = box

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            self.saveFile.flush()
            self.savestep += 1


class StabilityReporter:
    def __init__(
        self,
        filename: Union[str, Path],
        reportInterval: int = 100,
        logger: Optional[object] = None,
        kinetic_threshold: float = 5.0,
        potential_threshold: float = 1000.0,
        scale: float = 1.0,
        force_reinitialize: bool = True,
    ):
        self.saveFile = open(filename, "w")
        self.interval: int = reportInterval
        self.kinetic_threshold: float = kinetic_threshold
        self.potential_threshold: float = potential_threshold
        self.scale: float = scale
        self.force_reinitialize: bool = force_reinitialize
        self.logger = logger or LogManager().get_logger(__name__)
        self.logger.info(
            f"StabilityReporter initialized with thresholds: K.E. = {self.kinetic_threshold}, P.E. = {self.potential_threshold}, force_reinitialize={self.force_reinitialize}"
        )

    def describeNextReport(
        self, simulation: Simulation
    ) -> Tuple[int, bool, bool, bool, bool]:
        """
        Required by OpenMM Reporter interface. Specifies when the next report should occur.
        """
        steps_to_next_report = self.interval - simulation.currentStep % self.interval
        return (
            steps_to_next_report,
            False,
            False,
            False,
            True,
        )  # positions, velocities, forces, energies

    def report(self, simulation: Simulation, state: State) -> None:
        """
        Main method called at every reportInterval steps. Checks energies and reinitializes velocities if needed.
        """
        num_particles = simulation.system.getNumParticles()
        e_kinetic = (
            state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            / num_particles
        )
        e_potential = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            / num_particles
        )

        if hasattr(simulation.integrator, "getTemperature"):
            temperature = simulation.integrator.getTemperature().value_in_unit(
                unit.kelvin
            )
        else:
            temperature = 240.0

        kBT = 0.008314 * temperature
        e_kinetic_expected = 1.5 * kBT

        if (
            e_kinetic / e_kinetic_expected > self.kinetic_threshold
            or abs(e_potential) / e_kinetic_expected > self.potential_threshold
        ):

            if self.force_reinitialize:
                seed = np.random.randint(100_000)
                simulation.context.setVelocitiesToTemperature(temperature, seed)
                self.saveFile.write(
                    f"<<INSTABILITY | Reinitialized velocities>> Step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}\n"
                )
                self.saveFile.flush()
                if simulation.currentStep % (self.interval * 100) == 0:
                    self.logger.warning(
                        f"<<INSTABILITY | Reinitialized velocities>> at step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}"
                    )
            else:
                self.saveFile.write(
                    f"<<INSTABILITY | Detected but NOT reinitialized velocities>> Step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}\n"
                )
                self.saveFile.flush()
                if simulation.currentStep % (self.interval * 100) == 0:
                    self.logger.warning(
                        f"<<INSTABILITY | Detected but NOT reinitialized velocities>> at step {simulation.currentStep}: K.E. = {e_kinetic:.2f} | P.E. = {e_potential:.2f}"
                    )


class EnergyReporter:
    def __init__(
        self,
        report_file: Union[str, Path],
        force_field_manager,
        reportInterval: int = 1000,
        reportForceGrp: bool = False,
    ):
        self.filename: str = str(report_file)
        self.saveFile = open(report_file, "w")
        self.interval: int = reportInterval
        self.ff_man = force_field_manager
        self.report_force_grp: bool = reportForceGrp
        self.is_initialized: bool = False
        self.is_paused: bool = False

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def _make_header(self, simulation: Simulation) -> None:
        system: System = simulation.system
        self.saveFile.write(
            f"{'Step':<10} {'Temperature':<12} {'RG':<10} {'K.E./particle':<15} {'P.E./particle':<15}"
        )
        if self.report_force_grp:
            for i, force in enumerate(system.getForces()):
                group = force.getForceGroup()
                force_name = self.ff_man.force_name_map.get(i, "Unnamed")
                force_name_w_grp = f"{force_name}({group})"
                self.saveFile.write(f"{force_name_w_grp:<20}")
        self.saveFile.write("\n")
        self.saveFile.flush()

    def _initialize_constants(self, simulation: Simulation) -> None:
        system: System = simulation.system
        dof: int = 0
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0 * unit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if (
                system.getParticleMass(p1) > 0 * unit.dalton
                or system.getParticleMass(p2) > 0 * unit.dalton
            ):
                dof -= 1
        if any(
            isinstance(system.getForce(i), CMMotionRemover)
            for i in range(system.getNumForces())
        ):
            dof -= 3
        self._dof: int = dof
        self.is_initialized = True

    def describeNextReport(
        self, simulation: Simulation
    ) -> tuple[int, bool, bool, bool, bool]:
        steps_to_next_report: int = (
            self.interval - simulation.currentStep % self.interval
        )
        return (steps_to_next_report, True, False, False, True)

    def report(self, simulation: Simulation, state: State) -> None:
        if not self.is_initialized:
            self._initialize_constants(simulation)
            self._make_header(simulation)

        if not self.is_paused:
            system: System = simulation.system
            context = simulation.context
            num_particles: int = system.getNumParticles()
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            Rg: float = Analyzer.compute_RG(positions, print_choice=False)

            integrator = simulation.context.getIntegrator()
            if hasattr(integrator, "computeSystemTemperature"):
                temperature: float = (
                    integrator.computeSystemTemperature().value_in_unit(unit.kelvin)
                )
            else:
                temperature = (
                    2
                    * state.getKineticEnergy()
                    / (self._dof * unit.MOLAR_GAS_CONSTANT_R)
                ).value_in_unit(unit.kelvin)

            ke_per_particle: float = (
                state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
                / num_particles
            )
            pe_per_particle: float = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                / num_particles
            )
            self.saveFile.write(
                f"{simulation.currentStep:<10} {temperature:<12.3f} {Rg:<10.4f} {ke_per_particle:<15.4f} {pe_per_particle:<15.4f}"
            )

            if self.report_force_grp:
                for i, force in enumerate(system.getForces()):
                    group: int = force.getForceGroup()
                    state_grp = context.getState(getEnergy=True, groups={group})
                    pot_energy: float = state_grp.getPotentialEnergy().value_in_unit(
                        unit.kilojoules_per_mole
                    )
                    pe_grp_per_particle: float = pot_energy / num_particles
                    self.saveFile.write(f"{pe_grp_per_particle:<20.4f}")

            self.saveFile.write("\n")
            self.saveFile.flush()


class ForceFieldReporter:
    """
    Write-once reporter that records the complete force-field configuration
    (energy expressions, all parameters, integrator settings, system metadata)
    to a plain-text file at the time of simulation setup.

    Call ``update_activity(F_seq, tau_seq)`` after ``set_activity()`` to append
    the per-particle activity parameters to the same file.
    """

    def __init__(
        self,
        report_file: Union[str, Path],
        force_field_manager,
        integrator_manager,
        system_info: dict,
    ):
        """
        Parameters
        ----------
        report_file       : path to the output ``.txt`` file.
        force_field_manager : the ``ForceFieldManager`` instance attached to the simulation.
        integrator_manager  : the ``IntegratorManager`` instance attached to the simulation.
        system_info         : dict with keys ``name``, ``num_particles``, ``mass``,
                              ``PBC``, and optionally ``box_vectors``.
        """
        self.filename = str(report_file)
        self.ff_man = force_field_manager
        self.im = integrator_manager
        self.system_info = system_info
        self._write()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nb_method_name(method_int: int) -> str:
        names = {
            0: "NoCutoff",
            1: "CutoffNonPeriodic",
            2: "CutoffPeriodic",
            3: "Ewald",
            4: "PME",
            5: "LJPME",
        }
        return names.get(method_int, str(method_int))

    def _force_block(self, force_obj, force_name: str) -> list:
        """Return a list of formatted lines describing one OpenMM force."""
        lines = []
        lines.append(f"  Name              : {force_name}")
        lines.append(f"  Class             : {force_obj.__class__.__name__}")
        lines.append(f"  Group             : {force_obj.getForceGroup()}")

        if hasattr(force_obj, "getEnergyFunction"):
            lines.append(f"  Energy expression : {force_obj.getEnergyFunction()}")

        if hasattr(force_obj, "getNumGlobalParameters") and force_obj.getNumGlobalParameters() > 0:
            lines.append("  Global parameters :")
            for i in range(force_obj.getNumGlobalParameters()):
                pname = force_obj.getGlobalParameterName(i)
                pval = force_obj.getGlobalParameterDefaultValue(i)
                lines.append(f"    {pname} = {pval}")

        if hasattr(force_obj, "getNumPerParticleParameters") and force_obj.getNumPerParticleParameters() > 0:
            pp = [force_obj.getPerParticleParameterName(i) for i in range(force_obj.getNumPerParticleParameters())]
            lines.append(f"  Per-particle params: {', '.join(pp)}")

        if hasattr(force_obj, "getNumPerBondParameters") and force_obj.getNumPerBondParameters() > 0:
            pb = [force_obj.getPerBondParameterName(i) for i in range(force_obj.getNumPerBondParameters())]
            lines.append(f"  Per-bond params   : {', '.join(pb)}")

        if hasattr(force_obj, "getCutoffDistance"):
            try:
                lines.append(f"  Cutoff (nm)       : {float(force_obj.getCutoffDistance()):.4f}")
            except Exception:
                pass

        if hasattr(force_obj, "getNonbondedMethod"):
            lines.append(f"  Nonbonded method  : {self._nb_method_name(force_obj.getNonbondedMethod())}")

        for getter, label in [
            ("getNumParticles", "Num particles     "),
            ("getNumBonds",     "Num bonds         "),
            ("getNumAngles",    "Num angles        "),
            ("getNumExclusions","Num exclusions    "),
        ]:
            if hasattr(force_obj, getter):
                lines.append(f"  {label}: {getattr(force_obj, getter)()}")

        if hasattr(force_obj, "getFrequency"):
            lines.append(f"  Removal frequency : {force_obj.getFrequency()}")

        if hasattr(force_obj, "getNumPerParticleParameters") and hasattr(force_obj, "getNumParticles"):
            num_p = force_obj.getNumParticles()
            if num_p > 0:
                for i in range(force_obj.getNumPerParticleParameters()):
                    pname = force_obj.getPerParticleParameterName(i)
                    vals = []
                    for p in range(num_p):
                        try:
                            # getParticleParameters returns (parameters_tuple)
                            params = force_obj.getParticleParameters(p)
                            vals.append(params[0][i])
                        except Exception:
                            pass
                    if len(vals) == num_p:
                        import numpy as np
                        vals_arr = np.array(vals)
                        if len(np.unique(vals_arr)) > 1:
                            lines.append(f"  Per-particle param: {pname} (Heterogeneous array)")
                            lines.append(f"  full_{pname}_array: {vals_arr.tolist()}")
                        else:
                            lines.append(f"  Per-particle param: {pname} = {vals_arr[0]}")

        return lines

    def _write(self) -> None:
        """Write the full force-field report to disk."""
        si = self.system_info
        im = self.im
        sep = "-" * 62
        header = "=" * 62

        with open(self.filename, "w") as fh:
            fh.write(header + "\n")
            fh.write("  chromdyn : Force-field Configuration\n")
            fh.write(f"  Simulation : {si.get('name', '')}\n")
            fh.write(f"  Created    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(header + "\n\n")

            fh.write("[SYSTEM]\n")
            fh.write(f"  num_particles     : {si['num_particles']}\n")
            
            mass_data = si.get("mass", 1.0)
            if hasattr(mass_data, '__iter__'):
                import numpy as np
                mass_arr = np.array(mass_data)
                if len(np.unique(mass_arr)) > 1:
                    fh.write(f"  particle_mass (Da): Heterogeneous array\n")
                    fh.write(f"  full_mass_array   : {mass_arr.tolist()}\n")
                else:
                    fh.write(f"  particle_mass (Da): {mass_arr[0]}\n")
            else:
                fh.write(f"  particle_mass (Da): {mass_data}\n")
            fh.write(f"  PBC               : {si['PBC']}\n")
            bv = si.get("box_vectors")
            if bv is not None:
                fh.write(
                    f"  box_vectors (nm)  : {tuple(bv[0])} | {tuple(bv[1])} | {tuple(bv[2])}\n"
                )
            fh.write("\n")

            # ---- INTEGRATOR ----
            fh.write("[INTEGRATOR]\n")
            fh.write(f"  type              : {getattr(im, 'integrator_name', 'unknown')}\n")
            fh.write(f"  temperature       : {getattr(im, 'temperature', 'N/A')}\n")
            fh.write(f"  friction          : {getattr(im, 'friction', 'N/A')}\n")
            fh.write(f"  timestep          : {getattr(im, 'timestep', 'N/A')}\n")
            fh.write("\n")

            # ---- FORCES ----
            fh.write("[FORCES]\n")
            for fname, fobj in self.ff_man.forceDict.items():
                fh.write(sep + "\n")
                for line in self._force_block(fobj, fname):
                    fh.write(line + "\n")
            fh.write(sep + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_activity(self, F_seq, tau_seq) -> None:
        """
        Append per-particle activity parameters to the existing report file.
        Call this after ``ChromatinDynamics.set_activity()``.
        """
        with open(self.filename, "a") as fh:
            fh.write("\n[ACTIVITY]\n")
            fh.write(f"  F_seq   : {[float(x) for x in F_seq]}\n")
            fh.write(f"  tau_seq : {[float(x) for x in tau_seq]}\n")
