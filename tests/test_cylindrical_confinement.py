"""Analytical regression checks for cylindrical confinement and both end caps."""

import numpy as np
import pytest
from openmm import Context, Platform, VerletIntegrator, unit

from chromdyn import ChromatinDynamics, TopologyGenerator


@pytest.mark.parametrize("position", [
    (0.5, 0.25, 0.0),
    (0.5, 0.25, 2.0),
    (0.5, 0.25, -2.0),
    (0.5, 0.25, 2.5),
    (0.5, 0.25, -2.5),
    (2.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0),
    (2.0, 0.0, 2.5),
    (2.0, 0.0, -2.5),
])
def test_cylinder_energy_and_force_match_symmetric_geometry(position):
    generator = TopologyGenerator()
    generator.gen_top([1])
    sim = ChromatinDynamics(
        generator.topology, platform_name="Reference",
        write_logs=False, console_stream=False,
    )
    radius, half_height, stiffness = 1.5, 2.0, 4.0
    sim.force_field_manager.add_cylindrical_confinement(
        r_cyl=radius, z_cyl=half_height, k_cyl=stiffness,
    )
    integrator = VerletIntegrator(0.001)
    context = Context(sim.system, integrator, Platform.getPlatformByName("Reference"))
    context.setPositions([position])

    xyz = np.asarray(position)
    radial_distance = np.linalg.norm(xyz[:2])
    radial_excess = max(radial_distance - radius, 0.0)
    axial_excess = max(abs(xyz[2]) - half_height, 0.0)
    expected_energy = 0.5 * stiffness * (radial_excess**2 + axial_excess**2)
    expected_force = np.zeros(3)
    expected_force[:2] = -stiffness * radial_excess * xyz[:2] / radial_distance
    expected_force[2] = -stiffness * axial_excess * np.sign(xyz[2])

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    np.testing.assert_allclose(energy, expected_energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(force[0], expected_force, rtol=1e-12, atol=1e-12)
