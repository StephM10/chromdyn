#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

from openmm import (
    System,
    Discrete2DFunction,
    HarmonicBondForce,
    CustomNonbondedForce,
    CustomExternalForce,
    CMMotionRemover,
    CustomBondForce,
    HarmonicAngleForce,
)
from openmm.app import Topology
import numpy as np
import pandas as pd
from .utilities import LogManager
from typing import Optional, Tuple, List, Dict, Union, Any


class ForceFieldManager:
    def __init__(
        self,
        topology: Topology,
        system: System,
        logger: Optional[object] = None,
        Nonbonded_cutoff: float = 3.0,
        Nonbonded_method: str = "NonPeriodic",
        exclude_bonds_from_NonBonded: bool = True,
    ):
        self.logger = logger or LogManager().get_logger(__name__)
        self.topology = topology
        self.num_particles = topology.getNumAtoms()
        self.system = system
        self.nonbonded_cutoff = Nonbonded_cutoff
        self.nonbonded_method = (
            CustomNonbondedForce.CutoffNonPeriodic
            if Nonbonded_method == "NonPeriodic"
            else CustomNonbondedForce.CutoffPeriodic
        )
        self.forceDict: Dict[str, object] = {}
        self.force_name_map: Dict[int, str] = {}
        self.exclude_bonds_from_NonBonded = exclude_bonds_from_NonBonded

    def set_nonbonded_method(self, method: str) -> None:
        """Update nonbonded method for this manager and all existing CustomNonbondedForce objects.
        method ∈ {'NonPeriodic','Periodic'}.
        """
        from openmm import CustomNonbondedForce

        self.nonbonded_method = (
            CustomNonbondedForce.CutoffNonPeriodic
            if method == "NonPeriodic"
            else CustomNonbondedForce.CutoffPeriodic
        )

        if method == "Periodic":
            # get box vector
            box_vectors = self.system.getDefaultPeriodicBoxVectors()
            if (
                box_vectors[0].x == 0
                and box_vectors[1].y == 0
                and box_vectors[2].z == 0
            ):
                self.logger.warning(
                    "PBC requested but DefaultPeriodicBoxVectors are zero! PBC will likely fail."
                )
                min_box_dim = 99999.9  # avoid division by zero, but OpenMM will still throw an error
            else:
                lengths = [box_vectors[0].x, box_vectors[1].y, box_vectors[2].z]
                min_box_dim = min(lengths)

            # check cutoff
            if self.nonbonded_cutoff > min_box_dim / 2.0:
                new_cutoff = (
                    min_box_dim / 2.0 - 0.01
                )  # slightly adjust in case of float error
                self.logger.warning(
                    f"Requested cutoff {self.nonbonded_cutoff} nm is too large for "
                    f"box dimension {min_box_dim} nm. Clamping cutoff to {new_cutoff:.4f} nm."
                )
                self.nonbonded_cutoff = new_cutoff

        for name, force in self.forceDict.items():
            if force.__class__.__name__ == "CustomNonbondedForce":
                force.setNonbondedMethod(self.nonbonded_method)
                force.setCutoffDistance(self.nonbonded_cutoff)
                print(f"Updated nonbonded method for {name} to {method}.")
                self.logger.info(f"Updated nonbonded method for {name} to {method}.")

    def register_force(self, force_obj: object, name: str, **kwargs: Any) -> None:
        if (
            self.exclude_bonds_from_NonBonded
            and force_obj.__class__.__name__ == "CustomNonbondedForce"
        ):
            force_obj.createExclusionsFromBonds(
                [[int(bond[0].id), int(bond[1].id)] for bond in self.topology.bonds()],
                1,
            )
            self.logger.info("Added exclusions from bonded monomers.")
        force_index = self.system.addForce(force_obj)
        self.forceDict[name] = force_obj
        self.force_name_map[force_index] = name
        self.logger.info(f"{name} force successfully added to system.")
        self.logger.info("-" * 50)

    def add_exclusions_from_bonds(self, force: object) -> None:
        for bond in self.topology.bonds():
            force.addExclusion(int(bond[0].id), int(bond[1].id))

    def _isForceDictEqualSystemForces(self) -> bool:
        forcesInDict = [x.this for x in self.forceDict.values()]
        forcesInSystem = [x.this for x in self.system.getForces()]
        return len(forcesInDict) == len(forcesInSystem) and all(
            f in forcesInSystem for f in forcesInDict
        )

    def _getForceIndex(self, forceName: str) -> int:
        return [key for key, item in self.force_name_map.items() if item == forceName][
            0
        ]

    def removeForce(self, context: object, forceName: str) -> None:
        if forceName in self.forceDict:
            self.system.removeForce(self._getForceIndex(forceName))
            del self.forceDict[forceName]
            context.reinitialize(preserveState=True)
            self.logger.info(f"Removed {forceName} from the system!")
            assert (
                self._isForceDictEqualSystemForces()
            ), "Forces in forceDict should be the same as in the system!"
        else:
            self.logger.warning(f"The system does not have force {forceName}!!")
            self.logger.warning(
                f"The forces applied in the system are: {list(self.forceDict.keys())}"
            )
            raise ValueError

    def addLEFBonds(
        self,
        anchors: List[Tuple[int, int]],
        r0: float = 1.0,
        k: float = 10.0,
        group: int = 31,
    ) -> None:
        # Define harmonic bond energy expression using a global parameter for k
        bond_force = CustomBondForce("0.5 * k_LEF * (r - r0)^2")
        bond_force.setForceGroup(group)

        # Add per-bond parameter r0, k_LEF
        bond_force.addPerBondParameter("r0")
        bond_force.addPerBondParameter("k_LEF")

        # Add each bond with its own equilibrium distance
        for i, j in anchors:
            bond_force.addBond(int(i), int(j), [r0, k])

        self.logger.info(
            f"Adding {len(anchors)} unique LEF bonds with r0={r0}, k={k}, group={group}"
        )
        self.register_force(bond_force, "LEFBonds")

    def removeCOM(self, frequency: int = 100, group: int = 31) -> None:
        self.logger.info(
            f"Adding CMMotionRemover with frequency: {frequency} and force group: {group}"
        )
        cmm_remove = CMMotionRemover(frequency)
        cmm_remove.setForceGroup(group)
        self.register_force(cmm_remove, "CMMRemover")

    def add_harmonic_bonds(
        self, r0: float = 1.0, k: float = 10.0, group: int = 0
    ) -> None:
        bond_force = HarmonicBondForce()
        bond_force.setForceGroup(group)
        num_bonds = 0
        for bond in self.topology.bonds():
            bond_force.addBond(int(bond[0].id), int(bond[1].id), r0, k)
            num_bonds += 1
        self.logger.info(
            f"Adding {num_bonds} harmonic bonds with r0={r0}, k={k}, group={group}"
        )
        self.register_force(bond_force, "HarmonicBonds")

    def add_harmonic_angles(
        self, theta0: float = 180.0, k: float = 2.0, group: int = 4
    ) -> None:
        """Adds harmonic angle forces between triplets of bonded particles."""
        theta0_rad = theta0 * (np.pi / 180)
        angle_force = HarmonicAngleForce()
        angle_force.setForceGroup(group)

        bonded_neighbors = {atom.index: [] for atom in self.topology.atoms()}
        for bond in self.topology.bonds():
            a, b = int(bond[0].id), int(bond[1].id)
            bonded_neighbors[a].append(b)
            bonded_neighbors[b].append(a)

        num_angles = 0
        for j, neighbors in bonded_neighbors.items():
            if len(neighbors) < 2:
                continue
            if len(neighbors) > 3:
                self.logger.error(
                    "More than two bonded neighbors. Cannot resolve angle triplet."
                )
                raise ValueError

            for i in neighbors:
                for k_ in neighbors:
                    if i >= k_:
                        continue
                    angle_force.addAngle(i, j, k_, theta0_rad, k)
                    num_angles += 1

        self.logger.info(f"Adding {num_angles} harmonic angles with parameters:")
        self.logger.info(
            f"θ₀: {theta0}° ({theta0_rad:.4f} rad), k: {k}, force group: {group}"
        )
        self.register_force(angle_force, "HarmonicAngles")

    def add_fene_bonds(self, k: float = 30.0, R0: float = 1.5, group: int = 0) -> None:
        """Adds FENE (Finite Extensible Nonlinear Elastic) bonds."""
        self.logger.info(f"Adding FENE bonds with k={k}, R0={R0}, force group={group}")
        fene_force = CustomBondForce(
            "-0.5 * k_fene * R0_fene^2 * log(1 - (r/R0_fene)^2)"
        )
        fene_force.addGlobalParameter("k_fene", k)
        fene_force.addGlobalParameter("R0_fene", R0)

        for bond in self.topology.bonds():
            fene_force.addBond(int(bond[0].id), int(bond[1].id), ())

        fene_force.setForceGroup(group)
        self.register_force(fene_force, "FENEBonds")

    def add_harmonic_trap(
        self,
        kr: float = 0.1,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        group: int = 1,
    ) -> None:
        """Adds a harmonic trap to confine particles within a spherical region."""
        self.logger.info(
            f"Adding Harmonic Trap with kr={kr}, center={center}, force group={group}"
        )
        restraintForce = CustomExternalForce(
            "0.5 * k_harm_trap * r * r; "
            "r = sqrt( (x - x0_trap)^2 + (y - y0_trap)^2 + (z - z0_trap)^2 )"
        )
        restraintForce.addGlobalParameter("k_harm_trap", kr)
        restraintForce.addGlobalParameter("x0_trap", center[0])
        restraintForce.addGlobalParameter("y0_trap", center[1])
        restraintForce.addGlobalParameter("z0_trap", center[2])

        for i in range(self.topology.getNumAtoms()):
            restraintForce.addParticle(i, ())

        restraintForce.setForceGroup(group)
        self.register_force(restraintForce, "HarmonicTrap")

    def add_flat_bottom_harmonic(
        self,
        k: float = 0.1,
        r0: float = 10.0,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        group: int = 1,
    ) -> None:
        """Adds a flat-bottom harmonic potential to confine particles inside a spherical boundary."""
        self.logger.info("-" * 50)
        self.logger.info("Adding Flat-Bottom Harmonic potential with parameters:")
        self.logger.info(f"k = {k}, r0 = {r0}, group = {group}")

        energy_expr = (
            "step(r - rRes) * 0.5 * kR * (r - rRes)^2;"
            "r = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
        )

        restraintForce = CustomExternalForce(energy_expr)
        restraintForce.setForceGroup(group)
        restraintForce.addGlobalParameter("kR", k)
        restraintForce.addGlobalParameter("rRes", r0)
        restraintForce.addGlobalParameter("x0", center[0])
        restraintForce.addGlobalParameter("y0", center[1])
        restraintForce.addGlobalParameter("z0", center[2])

        for i in range(self.topology.getNumAtoms()):
            restraintForce.addParticle(i, ())

        self.register_force(restraintForce, "FlatBottomHarmonic")

    def add_custom_flat_bottom_harmonic(
        self,
        particle_indices: List[int],
        ks: List[float],
        r0s: List[float],
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        group: int = 1,
    ) -> None:
        """
        Adds a flat-bottom harmonic potential with per-particle force constants (k)
        and cutoff radii (r0). Particles beyond r0 from the center feel a harmonic
        potential with spring constant k.

        Parameters
        ----------
        particle_indices : list of int
            Atom indices to which the potential will be applied.
        ks : list of float
            Force constants (k) for each selected atom (in kJ/mol/nm^2).
        r0s : list of float
            Flat-bottom cutoff radii (r0) for each selected atom (in nm).
        center : tuple of float, default=(0,0,0)
            Reference point for the spherical restraint.
        group : int, default=1
            Force group to assign the force to.
        """
        n_atoms = self.topology.getNumAtoms()

        # --- sanity checks ---
        if not (len(particle_indices) == len(ks) == len(r0s)):
            raise ValueError(
                "particle_indices, ks, and r0s must all have the same length"
            )

        if any(idx < 0 or idx >= n_atoms for idx in particle_indices):
            raise ValueError(f"One or more indices are out of range (0..{n_atoms-1})")

        self.logger.info("-" * 50)
        self.logger.info(
            "Adding Custom Flat-Bottom Harmonic potential with per-particle params"
        )
        self.logger.info(f"Group = {group}, Num particles = {len(particle_indices)}")

        # Energy expression: flat-bottom harmonic
        energy_expr = (
            "step(r - rRes) * 0.5 * kR * (r - rRes)^2;"
            "r = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
        )

        restraintForce = CustomExternalForce(energy_expr)
        restraintForce.setForceGroup(group)

        # Global center parameters
        restraintForce.addGlobalParameter("x0", center[0])
        restraintForce.addGlobalParameter("y0", center[1])
        restraintForce.addGlobalParameter("z0", center[2])

        # Per-particle parameters: kR, rRes
        restraintForce.addPerParticleParameter("kR")
        restraintForce.addPerParticleParameter("rRes")

        # Add only the requested particles with their per-particle params
        for idx, k, r0 in zip(particle_indices, ks, r0s):
            restraintForce.addParticle(idx, [k, r0])

        self.register_force(restraintForce, "CustomFlatBottomHarmonic")

    def add_cylindrical_confinement(
        self,
        r_cyl: float = 5.0,
        z_cyl: float = 10.0,
        k_cyl: float = 10.0,
        group: int = 1,
    ) -> None:
        """
        Adds a cylindrical confinement potential to the system. Particles are confined within
        a cylinder defined by radius in the xy-plane and height along z.
        """
        self.logger.info("-" * 50)
        self.logger.info("Adding Cylindrical Confinement force:")
        self.logger.info(
            f"  r_cyl = {r_cyl}, z_cyl = {z_cyl}, k_cyl = {k_cyl}, group = {group}"
        )

        energy_expr = (
            "step(r_xy - r_cyl) * 0.5 * k_cyl * (r_xy - r_cyl)^2 + "
            "step(z^2 - z_cyl^2) * 0.5 * k_cyl * (z - z_cyl)^2; "
            "r_xy = sqrt(x^2 + y^2)"
        )

        confinement_force = CustomExternalForce(energy_expr)
        confinement_force.addGlobalParameter("r_cyl", r_cyl)
        confinement_force.addGlobalParameter("z_cyl", z_cyl)
        confinement_force.addGlobalParameter("k_cyl", k_cyl)
        confinement_force.setForceGroup(group)

        for i in range(self.topology.getNumAtoms()):
            confinement_force.addParticle(i, ())

        self.register_force(confinement_force, "CylindricalConfinement")

    def add_self_avoidance(
        self, Ecut: float = 4.0, k: float = 5.0, r: float = 1.0, radii: Optional[Union[List[float], np.ndarray]] = None, group: int = 2
    ) -> None:
        """
        Adds soft-core self-avoidance force.
        
        Args:
            Ecut (float): Energy cutoff parameter. Default = 4.0.
            k (float): Steepness of the soft-core repulsion. Default = 5.0.
            r (float): Old scalar parameter representing the uniform contact distance 
                       (diameter) between any two identical beads. If provided without `radii`, 
                       all beads share this same interaction distance. Default = 1.0.
            radii (array-like, optional): Array of individual bead radii. If provided, 
                       the interaction distance between bead i and bead j will be computed 
                       dynamically as `r_rep_i + r_rep_j`. This enables heterogeneous 
                       particle sizes. When this is passed, the scalar parameter `r` is ignored.
            group (int): Force group for this force. Default = 2.
        """
        if radii is not None:
            self.logger.info("Using per-particle 'radii' array. The scalar parameter 'r' will be ignored.")
            repul_energy = "0.5 * Ecut * (1.0 + tanh((k_rep * (r_rep1 + r_rep2 - r))))"
            avoidance_force = CustomNonbondedForce(repul_energy)
            avoidance_force.setForceGroup(group)
            avoidance_force.setCutoffDistance(self.nonbonded_cutoff)
            avoidance_force.setNonbondedMethod(self.nonbonded_method)

            avoidance_force.addGlobalParameter("Ecut", Ecut)
            avoidance_force.addGlobalParameter("k_rep", k)
            avoidance_force.addPerParticleParameter("r_rep")

            radii_array = np.asarray(radii, dtype=float)
            num_particles = getattr(self, "num_particles", self.system.getNumParticles())
            if len(radii_array) != num_particles:
                raise ValueError("Length of radii array must match number of particles")
            
            for r_val in radii_array:
                avoidance_force.addParticle([r_val])
                
            self.logger.info("Adding Self-avoidance force with parameters:")
            self.logger.info(
                f"Ecut={Ecut}, k_rep={k}, radii array (unique: {np.unique(radii_array)}), cutoff={self.nonbonded_cutoff}, group={group}"
            )
        else:
            repul_energy = "0.5 * Ecut * (1.0 + tanh((k_rep * (r_rep - r))))"
            avoidance_force = CustomNonbondedForce(repul_energy)
            avoidance_force.setForceGroup(group)
            avoidance_force.setCutoffDistance(self.nonbonded_cutoff)
            avoidance_force.setNonbondedMethod(self.nonbonded_method)

            avoidance_force.addGlobalParameter("Ecut", Ecut)
            avoidance_force.addGlobalParameter("r_rep", r)
            avoidance_force.addGlobalParameter("k_rep", k)

            num_particles = getattr(self, "num_particles", self.system.getNumParticles())
            for _ in range(num_particles):
                avoidance_force.addParticle(())

            self.logger.info("Adding Self-avoidance force with parameters:")
            self.logger.info(
                f"Ecut={Ecut}, k_rep={k}, r={r} (uniform diameter), cutoff={self.nonbonded_cutoff}, group={group}"
            )

        self.register_force(avoidance_force, "SelfAvoidance")

    def add_lennard_jones_force(
        self,
        epsilon: Optional[object] = None,
        sigma: Optional[object] = None,
        group: int = 3,
    ) -> None:
        """
        Adds a Lennard-Jones nonbonded force.
        """
        LJ_energy = "4 * e_LJ * ((sigma_LJ / r) ^ 12 - (sigma_LJ / r) ^ 6);\
                    e_LJ = min(e_LJ1, e_LJ2); sigma_LJ = 0.5 * (sigma_LJ1 + sigma_LJ2)"
        lj_force = CustomNonbondedForce(LJ_energy)
        lj_force.addPerParticleParameter("e_LJ")
        lj_force.addPerParticleParameter("sigma_LJ")

        lj_force.setForceGroup(group)
        lj_force.setCutoffDistance(self.nonbonded_cutoff)
        lj_force.setNonbondedMethod(self.nonbonded_method)

        num_particles = getattr(self, "num_particles", self.system.getNumParticles())
        if epsilon is None:
            epsilon_values = [0.5] * num_particles
        elif isinstance(epsilon, (float, int)):
            epsilon_values = [float(epsilon)] * num_particles
        elif isinstance(epsilon, (List, np.ndarray, Tuple)):
            epsilon_values = epsilon
        assert len(epsilon_values) == num_particles, "Wrong length of LJ epsilon values"

        if sigma is None:
            sigma_values = [1.0] * num_particles
        elif isinstance(sigma, (float, int)):
            sigma_values = [float(sigma)] * num_particles
        elif isinstance(sigma, (List, np.ndarray, Tuple)):
            sigma_values = sigma

        assert len(sigma_values) == num_particles, "Wrong length of LJ sigma values"

        for idx in range(num_particles):
            lj_force.addParticle([epsilon_values[idx], sigma_values[idx]])

        self.logger.info("Adding Lennard-Jones force:")
        self.logger.info(
            f"Particles:{num_particles}, epsilon = {np.unique(epsilon_values)}, sigma={np.unique(sigma_values)}, cutoff={self.nonbonded_cutoff}, group={group}"
        )
        self.register_force(lj_force, "LennardJones")

    def add_wca_force(
        self,
        epsilon: Optional[object] = None,
        sigma: Optional[object] = None,
        group: int = 3,
    ) -> None:
        """
        Adds a Weeks–Chandler–Andersen (WCA) repulsive nonbonded force.
        """
        # Define the WCA energy expression with a cutoff at r = 2^(1/6) * sigma_LJ
        WCA_energy = """
        step(2^(1/6) * sigma_wca - r) * (4 * e_wca * ((sigma_wca / r)^12 - (sigma_wca / r)^6) + e_wca);
        e_wca = min(e_wca1, e_wca2);
        sigma_wca = 0.5 * (sigma_wca1 + sigma_wca2)
        """

        wca_force = CustomNonbondedForce(WCA_energy)
        wca_force.addPerParticleParameter("e_wca")
        wca_force.addPerParticleParameter("sigma_wca")

        wca_force.setForceGroup(group)
        wca_force.setCutoffDistance(self.nonbonded_cutoff)
        wca_force.setNonbondedMethod(self.nonbonded_method)

        num_particles = getattr(self, "num_particles", self.system.getNumParticles())

        if epsilon is None:
            epsilon_values = [0.5] * num_particles
        elif isinstance(epsilon, (float, int)):
            epsilon_values = [float(epsilon)] * num_particles
        elif isinstance(epsilon, (List, np.ndarray, Tuple)):
            epsilon_values = epsilon
        assert (
            len(epsilon_values) == num_particles
        ), "Wrong length of WCA epsilon values"

        if sigma is None:
            sigma_values = [1.0] * num_particles
        elif isinstance(sigma, (float, int)):
            sigma_values = [float(sigma)] * num_particles
        elif isinstance(sigma, (List, np.ndarray, Tuple)):
            sigma_values = sigma
        assert len(sigma_values) == num_particles, "Wrong length of WCA sigma values"

        for idx in range(num_particles):
            wca_force.addParticle([epsilon_values[idx], sigma_values[idx]])

        self.logger.info("Adding WCA force:")
        self.logger.info(
            f"Particles: {num_particles}, epsilon = {np.unique(epsilon_values)}, "
            f"sigma = {np.unique(sigma_values)}, cutoff = {self.nonbonded_cutoff}, group = {group}"
        )
        self.register_force(wca_force, "WCA")

    def add_LJ_repulsion(self, sigma: float = 1.0, radii: Optional[Union[List[float], np.ndarray]] = None, group: int = 6) -> None:
        """
        Adds hard-core self-avoidance with a simple repulsive Lennard-Jones potential.

        Args:
            sigma (float): LJ sigma parameter. Default = 1.0. Used as uniform diameter.
            radii (array-like, optional): Array of individual bead radii. If provided, `sigma` is ignored.
            group (int): Force group for this force. Default = 6
        """
        if radii is not None:
            self.logger.info("Using per-particle 'radii' array. The scalar parameter 'sigma' will be ignored.")
            repul_energy = "(sigma_LJ / r) ^ 12; sigma_LJ = sigma1 + sigma2"
            hard_repel_force = CustomNonbondedForce(repul_energy)
            hard_repel_force.setForceGroup(group)
            hard_repel_force.setCutoffDistance(self.nonbonded_cutoff)
            hard_repel_force.setNonbondedMethod(self.nonbonded_method)

            hard_repel_force.addPerParticleParameter("sigma")

            radii_array = np.asarray(radii, dtype=float)
            num_particles = getattr(self, "num_particles", self.system.getNumParticles())
            if len(radii_array) != num_particles:
                raise ValueError("Length of radii array must match number of particles")

            for s_val in radii_array:
                hard_repel_force.addParticle([s_val])

            self.logger.info("Adding Hard-core repulsion force with parameters:")
            self.logger.info(
                f"radii array (unique: {np.unique(radii_array)}), cutoff={self.nonbonded_cutoff}, group={group}"
            )
        else:
            repul_energy = "(sigma / r) ^ 12"
            hard_repel_force = CustomNonbondedForce(repul_energy)
            hard_repel_force.setForceGroup(group)
            hard_repel_force.setCutoffDistance(self.nonbonded_cutoff)
            hard_repel_force.setNonbondedMethod(self.nonbonded_method)

            hard_repel_force.addGlobalParameter("sigma", sigma)

            num_particles = getattr(self, "num_particles", self.system.getNumParticles())
            for _ in range(num_particles):
                hard_repel_force.addParticle(())

            self.logger.info("Adding Hard-core repulsion force with parameters:")
            self.logger.info(
                f"sigma={sigma} (uniform diameter), cutoff={self.nonbonded_cutoff}, group={group}"
            )

        self.register_force(hard_repel_force, "HardCoreLJ")

    def add_type_to_type_interaction(
        self,
        interaction_matrix: Union[List[List[float]], np.ndarray],
        type_labels: List[str],
        mu: float = 2.0,
        rc: float = 2.0,
        group: int = 5,
        verbose: bool = True,
    ) -> None:
        """
        Adds type-to-type nonbonded interactions with a tanh potential.

        Args:
            interaction_matrix (2D array): Interaction matrix aligned with type_labels.
            type_labels (list of str): Type labels corresponding to interaction matrix.
            mu (float): Steepness parameter of the tanh potential.
            rc (float): Reference distance.
            group (int): Force group.
            verbose (bool): Whether to log unused types.
        """
        self.mu = mu
        self.rc = rc

        type_list = [atom.element for atom in self.topology.atoms()]
        used_types = sorted(set(type_list))
        self.logger.info("-" * 50)
        self.logger.info(
            f"Number of unique types detected in polymer: {len(used_types)}"
        )

        missing_types = [t for t in used_types if t not in type_labels]
        if missing_types:
            raise ValueError(
                f"Types found in topology but missing in interaction matrix: {missing_types}"
            )

        unused_types = [t for t in type_labels if t not in used_types]
        if unused_types and verbose:
            self.logger.warning(
                f"Types defined in interaction matrix but not used in topology: {unused_types}"
            )

        type_to_idx = {label: idx for idx, label in enumerate(type_labels)}
        subset_indices = [type_to_idx[t] for t in used_types]
        reduced_matrix = np.array(interaction_matrix)[
            np.ix_(subset_indices, subset_indices)
        ]

        num_types = len(used_types)
        flat_interaction = reduced_matrix.flatten().tolist()

        energy_expr = "interaction_map(t1, t2) * 0.5 * (1 + tanh(mu * (rc - r)));"

        type_force = CustomNonbondedForce(energy_expr)
        type_force.setForceGroup(group)
        type_force.setCutoffDistance(self.nonbonded_cutoff)
        type_force.setNonbondedMethod(self.nonbonded_method)
        type_force.addGlobalParameter("mu", self.mu)
        type_force.addGlobalParameter("rc", self.rc)

        interaction_func = Discrete2DFunction(num_types, num_types, flat_interaction)
        type_force.addTabulatedFunction("interaction_map", interaction_func)

        type_force.addPerParticleParameter("t")
        type_idx_map = {label: idx for idx, label in enumerate(used_types)}
        for type_label in type_list:
            type_force.addParticle([float(type_idx_map[type_label])])

        self.logger.info(
            f"Adding Type-to-Type interaction (force group {group}, {num_types} types)."
        )
        self.logger.info(
            f"Parameters -> mu: {mu}, rc: {rc}, cutoff: {self.nonbonded_cutoff}"
        )

        self.register_force(type_force, "TypeToType")

    def _get_type_interaction_matrix(
        self, file_path: str
    ) -> Tuple[List[str], np.ndarray]:
        """
        Loads type-to-type interaction matrix from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing interaction matrix.

        Returns:
            type_labels (List[str]): List of type labels.
            interaction_matrix (np.ndarray): Interaction matrix (2D array).
        """
        df = pd.read_csv(file_path)  # Assume first column and row are type labels
        type_labels: List[str] = list(df.columns)  # Extract type names from header
        interaction_matrix: np.ndarray = df.values.astype(
            float
        )  # Convert DataFrame to NumPy array (float)

        if self.logger:
            self.logger.info(f"Loaded interaction matrix from {file_path}")
            self.logger.info(
                f"Interaction matrix shape: {interaction_matrix.shape}, Type labels: {type_labels}"
            )

        return type_labels, interaction_matrix

    def _initialize_mono_pos_constraint(self, group: int) -> None:
        constraint_energy = (
            "0.5 * k_constraint * dist ^ 2 ;"
            "dist = sqrt((x - x_con) ^ 2  + (y - y_con) ^ 2 + (z - z_con) ^ 2)"
        )

        constraint_force = CustomExternalForce(constraint_energy)
        constraint_force.addPerParticleParameter("k_constraint")
        constraint_force.addPerParticleParameter("x_con")
        constraint_force.addPerParticleParameter("y_con")
        constraint_force.addPerParticleParameter("z_con")
        constraint_force.setForceGroup(group)

        self.register_force(constraint_force, "PosConstraint")

    def constrain_monomer_pos(
        self,
        mono_id: int,
        pos: Tuple[float, float, float],
        k: float = 50.0,
        group: int = 8,
    ) -> None:
        if "PosConstraint" not in self.forceDict.keys():
            self._initialize_mono_pos_constraint(group)

        self.forceDict["PosConstraint"].addParticle(
            int(mono_id), [float(k), float(pos[0]), float(pos[1]), float(pos[2])]
        )

    def _initialize_force_z_axis(self, group: int) -> None:
        z_pulling_energy = "- f_z * z + 0.5 * k_xy * ( x * x + y * y )"
        z_pulling_force = CustomExternalForce(z_pulling_energy)
        z_pulling_force.addPerParticleParameter("f_z")
        z_pulling_force.addPerParticleParameter("k_xy")
        z_pulling_force.setForceGroup(group)
        self.register_force(z_pulling_force, "zAxialPull")

    def apply_force_z_axis(
        self, mono_id: int, fz: float, k_xy: float = 100.0, group: int = 9
    ) -> None:
        if "zAxialPull" not in self.forceDict.keys():
            self._initialize_force_z_axis(group)

        self.forceDict["zAxialPull"].addParticle(int(mono_id), [float(fz), float(k_xy)])
