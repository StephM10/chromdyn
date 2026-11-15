"""Small analytical checks for the reusable force API; no experiment scripts."""

import numpy as np
import pytest
from openmm import Context, Platform, VerletIntegrator, unit

from chromdyn import ChromatinDynamics, TopologyGenerator


def _simulation():
    generator = TopologyGenerator()
    generator.gen_top([1])
    return ChromatinDynamics(
        generator.topology, platform_name="Reference",
        write_logs=False, console_stream=False,
    )


def test_trap_energy_and_force_match_analytic_potential():
    sim = _simulation()
    sim.force_field_manager.apply_oscillatory_tweezer(
        0, k_ot=4.0, A=0.5, omega=2.0, pos=(1.0, 2.0, 3.0),
    )
    integrator = VerletIntegrator(0.001)
    context = Context(sim.system, integrator, Platform.getPlatformByName("Reference"))
    position = np.array([[1.6, 1.7, 3.2]])
    context.setPositions(position)
    context.setParameter("t", 0.3)
    center = np.array([1 + 0.5*np.cos(0.6), 2, 3])
    displacement = position[0] - center
    state = context.getState(getEnergy=True, getForces=True)
    np.testing.assert_allclose(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), 2*np.dot(displacement, displacement))
    np.testing.assert_allclose(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer), [-4*displacement])


@pytest.mark.parametrize("parameters", [
    {"mono_id": -1}, {"mono_id": 1}, {"mono_id": 0.5},
    {"k_ot": -1}, {"A": float("nan")},
    {"pos": (0, 0)}, {"group": 32},
])
def test_invalid_trap_parameters_do_not_register_a_force(parameters):
    sim = _simulation()
    kwargs = dict(mono_id=0, k_ot=2, A=0.5, omega=1)
    kwargs.update(parameters)
    with pytest.raises(ValueError):
        sim.force_field_manager.apply_oscillatory_tweezer(**kwargs)
    assert sim.system.getNumForces() == 0
    assert "OscillatoryTweezer" not in sim.force_field_manager.forceDict
