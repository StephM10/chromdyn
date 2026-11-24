import os

import numpy as np
import pytest
from openmm import Context, Platform, VerletIntegrator, unit

from chromdyn import ChromatinDynamics, TopologyGenerator

PLATFORMS = ["Reference", "CPU"]
if os.environ.get("CHROMDYN_TEST_GPU") == "1":
    PLATFORMS += [
        Platform.getPlatform(i).getName()
        for i in range(Platform.getNumPlatforms())
        if Platform.getPlatform(i).getName() in {"CUDA", "OpenCL"}
    ]


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("kind", ["trap", "flat", "custom_flat", "cylinder"])
@pytest.mark.parametrize("radius", [0.0, 1.0])
def test_confinement_center_and_boundary_match_analytic_force(platform, kind, radius):
    gen = TopologyGenerator()
    gen.gen_top([1])
    sim = ChromatinDynamics(
        gen.topology, platform_name="Reference", write_logs=False, console_stream=False
    )
    fm = sim.force_field_manager
    center = np.array([0.2, -0.3, 0.7]) if kind != "cylinder" else np.zeros(3)
    if kind == "trap":
        fm.add_harmonic_trap(kr=2, center=tuple(center))
    elif kind == "flat":
        fm.add_flat_bottom_harmonic(k=2, r0=radius, center=tuple(center))
    elif kind == "custom_flat":
        fm.add_custom_flat_bottom_harmonic([0], [2], [radius], center=tuple(center))
    else:
        fm.add_cylindrical_confinement(r_cyl=radius, z_cyl=1, k_cyl=2)
    integ = VerletIntegrator(0.001)
    properties = (
        {"Precision": "double"}
        if platform in {"CUDA", "OpenCL"}
        else ({"Threads": "1"} if platform == "CPU" else {})
    )
    context = Context(
        sim.system, integ, Platform.getPlatformByName(platform), properties
    )
    for distance in [0.0, 1e-6, radius, radius + 0.25]:
        position = center + [distance, 0, 0]
        if kind == "cylinder":
            position[2] = 1.5
        context.setPositions([position])
        state = context.getState(getEnergy=True, getForces=True)
        excess = distance if kind == "trap" else max(distance - radius, 0)
        expected_force = [-2 * excess, 0, -1 if kind == "cylinder" else 0]
        expected_energy = excess**2 + (0.25 if kind == "cylinder" else 0)
        np.testing.assert_allclose(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            expected_energy,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            ),
            [expected_force],
            atol=1e-9,
        )


@pytest.mark.parametrize("kind", ["flat", "custom_flat", "cylinder"])
def test_negative_confinement_radius_is_rejected(kind):
    gen = TopologyGenerator()
    gen.gen_top([1])
    sim = ChromatinDynamics(gen.topology, write_logs=False, console_stream=False)
    with pytest.raises(ValueError, match="radius|r0|r_cyl"):
        if kind == "flat":
            sim.force_field_manager.add_flat_bottom_harmonic(r0=-1)
        elif kind == "custom_flat":
            sim.force_field_manager.add_custom_flat_bottom_harmonic([0], [1], [-1])
        else:
            sim.force_field_manager.add_cylindrical_confinement(r_cyl=-1)


@pytest.mark.parametrize("z_cyl", [-1.0, np.nan])
def test_invalid_cylinder_half_height_is_rejected(z_cyl):
    gen = TopologyGenerator()
    gen.gen_top([1])
    sim = ChromatinDynamics(gen.topology, write_logs=False, console_stream=False)
    with pytest.raises(ValueError, match="z_cyl"):
        sim.force_field_manager.add_cylindrical_confinement(r_cyl=1.0, z_cyl=z_cyl)
    assert sim.system.getNumForces() == 0
    assert "CylindricalConfinement" not in sim.force_field_manager.forceDict
