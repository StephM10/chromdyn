import logging

import numpy as np
import pytest
from openmm import (
    Context,
    CustomExternalForce,
    Platform,
    System,
    VerletIntegrator,
    unit,
)
from openmm.app import Topology

from chromdyn.forcefield import ForceFieldManager


def manager(n=1, ids=None):
    topology = Topology()
    residue = topology.addResidue("GEN", topology.addChain())
    atoms = [
        topology.addAtom("B", "A", residue, id=None if ids is None else ids[i])
        for i in range(n)
    ]
    for a, b in zip(atoms, atoms[1:]):
        topology.addBond(a, b)
    system = System()
    for _ in atoms:
        system.addParticle(1)
    return ForceFieldManager(topology, system, logger=logging.getLogger(__name__))


def external(value):
    force = CustomExternalForce(f"{value}*x")
    force.addParticle(0, [])
    return force


def test_repeated_force_removal_preserves_remaining_force_identity_and_state():
    fm = manager()
    for name, value in zip("ABC", [1, 2, 3]):
        fm.register_force(external(value), name)
    integrator = VerletIntegrator(0.001)
    context = Context(fm.system, integrator, Platform.getPlatformByName("Reference"))
    context.setPositions([[2, 0, 0]])
    context.setVelocities([[3, 0, 0]])
    context.setTime(7)
    for name, remaining in [("A", ["B", "C"]), ("B", ["C"]), ("C", [])]:
        fm.removeForce(context, name)
        assert list(fm.force_name_map.values()) == remaining
        assert fm.system.getNumForces() == len(remaining)
        state = context.getState(getPositions=True, getVelocities=True)
        np.testing.assert_allclose(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), [[2, 0, 0]]
        )
        np.testing.assert_allclose(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            [[3, 0, 0]],
        )
        assert state.getTime().value_in_unit(unit.picosecond) == 7
    fm.register_force(external(4), "A")
    assert fm.force_name_map == {0: "A"}


def test_duplicate_force_name_is_rejected_without_adding_a_force():
    fm = manager()
    fm.register_force(external(1), "same")
    with pytest.raises(ValueError, match="same"):
        fm.register_force(external(2), "same")
    assert fm.system.getNumForces() == 1


@pytest.mark.parametrize("ids", [None, ["alpha", "beta", "gamma"], ["10", "20", "90"]])
def test_bonds_angles_and_exclusions_use_particle_indices(ids):
    fm = manager(3, ids)
    fm.add_harmonic_bonds()
    fm.add_harmonic_angles()
    fm.add_fene_bonds()
    fm.add_self_avoidance()
    assert list(fm.forceDict["HarmonicBonds"].getBondParameters(0)[:2]) == [0, 1]
    assert list(fm.forceDict["HarmonicAngles"].getAngleParameters(0)[:3]) == [0, 1, 2]
    assert list(fm.forceDict["FENEBonds"].getBondParameters(1)[:2]) == [1, 2]
    nonbonded = next(
        f for f in fm.system.getForces() if hasattr(f, "getExclusionParticles")
    )
    assert {
        tuple(nonbonded.getExclusionParticles(i))
        for i in range(nonbonded.getNumExclusions())
    } == {(0, 1), (1, 2)}
