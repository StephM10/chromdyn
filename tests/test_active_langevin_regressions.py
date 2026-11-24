import os

import numpy as np
import pytest
from openmm import Context, CustomExternalForce, Platform, System, Vec3

from chromdyn.integrators import ActiveLangevinIntegrator


def one_step(integ):
    system = System()
    system.addParticle(1)
    force = CustomExternalForce("-x")
    force.addParticle(0, [])
    system.addForce(force)
    context = Context(system, integ, Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0]])
    context.setVelocities([[0, 0, 0]])
    integ.setPerDofVariableByName("p", [Vec3(1, 0, 0)])
    integ.step(1)
    state = context.getState(getPositions=True, getVelocities=True)
    # Keep the Context alive while its integrator globals are inspected.
    return context, (
        np.array(integ.getPerDofVariableByName("p")),
        state.getPositions(asNumpy=True)._value,
        state.getVelocities(asNumpy=True)._value,
    )


def test_changed_step_size_matches_direct_construction():
    changed = ActiveLangevinIntegrator(0, 0, 0.01)
    changed.setStepSize(0.02)
    direct = ActiveLangevinIntegrator(0, 0, 0.02)
    changed_context, actual = one_step(changed)
    direct_context, expected = one_step(direct)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(actual[0], [[np.exp(-0.02), 0, 0]])


def test_small_step_retains_thermal_variance():
    integ = ActiveLangevinIntegrator(1, 1, 1e-20)
    context, _ = one_step(integ)
    assert integ.getGlobalVariableByName("one_minus_c2") == pytest.approx(
        2e-20, rel=1e-14, abs=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1},
        {"gamma": -1},
        {"dt": 0},
        {"corr_time": 0},
        {"temperature": np.nan},
        {"gamma": np.inf},
    ],
)
def test_invalid_parameters_are_rejected_at_construction(kwargs):
    values = dict(temperature=1, gamma=1, dt=0.01, corr_time=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        ActiveLangevinIntegrator(**values)


def test_parameter_setters_reject_invalid_values_without_mutation():
    integ = ActiveLangevinIntegrator(0, 0, 0.01)
    with pytest.raises(ValueError):
        integ.setGlobalVariableByName("gamma", -1)
    with pytest.raises(ValueError):
        integ.setGlobalVariable(0, -1)  # kbT
    with pytest.raises(ValueError):
        integ.setStepSize(0)
    with pytest.raises(ValueError):
        integ.setPerDofVariableByName("t_corr", [Vec3(0, 1, 1)])
    with pytest.raises(ValueError):
        integ.setPerDofVariable(3, [Vec3(np.nan, 0, 0)])  # F_act
    assert integ.getGlobalVariableByName("gamma") == 0


@pytest.mark.parametrize("platform", ["Reference", "CUDA"])
def test_two_ou_half_steps_have_the_exact_full_step_variance(platform):
    if platform == "CUDA":
        if os.environ.get("CHROMDYN_TEST_GPU") != "1":
            pytest.skip("optional local GPU parity")
        if platform not in [
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]:
            pytest.skip("CUDA unavailable")
    count = 8192
    system = System()
    for _ in range(count):
        system.addParticle(1)
    integ = ActiveLangevinIntegrator(0, 0, 0.03, corr_time=0.1)
    integ.setRandomNumberSeed(891)
    properties = {"Precision": "double"} if platform == "CUDA" else {}
    context = Context(system, integ, Platform.getPlatformByName(platform), properties)
    context.setPositions(np.zeros((count, 3)))
    context.setVelocities(np.zeros((count, 3)))
    integ.setPerDofVariableByName("F_act", [Vec3(2, 2, 2)] * count)
    integ.step(1)
    sample = np.asarray(integ.getPerDofVariableByName("p")).ravel()
    expected_variance = 4 * (-np.expm1(-0.6))
    assert abs(sample.mean()) < 0.04 * np.sqrt(expected_variance)
    np.testing.assert_allclose(sample.var(), expected_variance, rtol=0.05)
