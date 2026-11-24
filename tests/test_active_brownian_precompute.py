"""Numerical regression checks for cached active Brownian integration.

Run with pytest -s on Windows: native CUDA compiler output needs real handles.
"""
import numpy as np
import pytest
from openmm import CustomIntegrator, CustomExternalForce, Context, Platform, System, Vec3, unit
from chromdyn import ChromatinDynamics, TopologyGenerator
from chromdyn.integrators import ActiveBrownianIntegrator


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("parameters", [
    {"collision_rate": 0}, {"corr_time": 0},
    {"temperature": -1}, {"timestep": float("nan")},
])
def test_invalid_constructor_parameters_are_rejected(precompute, parameters):
    with pytest.raises(ValueError):
        ActiveBrownianIntegrator(precompute_coefficients=precompute, **parameters)


@pytest.mark.parametrize("precompute", [False, True])
def test_invalid_activity_update_preserves_existing_state(precompute):
    it = ActiveBrownianIntegrator(temperature=0, precompute_coefficients=precompute)
    ctx = _context(it, 2)
    it.set_activity([1.0, 2.0], [0.1, 0.2])
    before = {name: np.asarray(it.getPerDofVariableByName(name)) for name in ("t_corr", "F_act")}
    for amplitudes, times in [([float("nan"), 2.0], [0.3, 0.4]), ([1.0, 2.0], [0, 0.2]), ([1.0], [0.1, 0.2])]:
        with pytest.raises(ValueError):
            it.set_activity(amplitudes, times)
        for name, expected in before.items():
            np.testing.assert_array_equal(it.getPerDofVariableByName(name), expected)
    with pytest.raises(ValueError):
        it.setGlobalVariableByName("g", 0)
    assert it.getGlobalVariableByName("g") == 1.0
    with pytest.raises(ValueError):
        it.setPerDofVariableByName("t_corr", [Vec3(0, 0, 0), Vec3(0, 0, 0)])
    np.testing.assert_array_equal(it.getPerDofVariableByName("t_corr"), before["t_corr"])


def _context(integrator, n, platform="Reference", constrained=False):
    system = System()
    for _ in range(n):
        system.addParticle(1.0)
    force = CustomExternalForce("-2*x+y")
    for i in range(n):
        force.addParticle(i, [])
    system.addForce(force)
    if constrained:
        for i in range(0, n, 2):
            system.addConstraint(i, i+1, 1.0)
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    context = Context(system, integrator, Platform.getPlatformByName(platform), properties)
    context.setPositions(np.array([[i, 0.0, 0.0] for i in range(n)]))
    return context


def _positions(context):
    return context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)


@pytest.mark.parametrize("precompute", [False, True])
def test_deterministic_updates_refresh_after_parameter_setters(precompute):
    it = ActiveBrownianIntegrator(temperature=0, timestep=0.01,
                                 precompute_coefficients=precompute)
    ctx = _context(it, 1)
    p = np.array([[1.0, 2.0, 3.0]])
    it.setPerDofVariableByName("p", [Vec3(*p[0])])
    x = np.zeros((1, 3))
    tau_index = next(i for i in range(it.getNumPerDofVariables())
                     if it.getPerDofVariableName(i) == "t_corr")
    g_index = next(i for i in range(it.getNumGlobalVariables())
                   if it.getGlobalVariableName(i) == "g")
    actions = [
        lambda: it.setStepSize(0.02*unit.picosecond),
        lambda: it.setPerDofVariableByName("t_corr", [Vec3(0.07, 0.1, 0.2)]),
        lambda: it.setPerDofVariable(tau_index, [Vec3(0.3, 0.4, 0.5)]),
        lambda: it.setGlobalVariableByName("g", 3.0),
        lambda: it.setGlobalVariable(g_index, 4.0),
    ]
    for update in actions:
        update()
        dt = it.getStepSize().value_in_unit(unit.picosecond)
        tau = np.asarray(it.getPerDofVariableByName("t_corr"))
        g = it.getGlobalVariableByName("g")
        p *= np.exp(-dt/tau)
        x += dt*(np.array([[2.0, -1.0, 0.0]]) + p)/g
        it.step(1)
        np.testing.assert_allclose(np.asarray(it.getPerDofVariableByName("p")), p, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(_positions(ctx), x, rtol=1e-12, atol=1e-12)


def test_explicit_refresh_after_bypassing_python_setters():
    it = ActiveBrownianIntegrator(temperature=0, timestep=0.01)
    ctx = _context(it, 1)
    CustomIntegrator.setStepSize(it, 0.03)
    CustomIntegrator.setPerDofVariableByName(it, "t_corr", [Vec3(0.3, 0.3, 0.3)])
    CustomIntegrator.setGlobalVariableByName(it, "g", 2.0)
    it.setPerDofVariableByName("p", [Vec3(1, 1, 1)])
    it.refresh_coefficients()
    it.step(1)
    expected_p = np.exp(-0.1)*np.ones((1, 3))
    expected_x = 0.03*(np.array([[2, -1, 0]])+expected_p)/2
    np.testing.assert_allclose(_positions(ctx), expected_x, atol=1e-12)


@pytest.mark.parametrize("setter", ["activity", "by_name", "by_index"])
def test_changed_activity_has_correct_one_step_ou_variance(setter):
    n = 16384
    it = ActiveBrownianIntegrator(temperature=0, timestep=0.003)
    it.setRandomNumberSeed({"activity": 301, "by_name": 302, "by_index": 303}[setter])
    ctx = _context(it, n)
    amplitudes = np.repeat([1.0, 3.0, 0.5, 6.0], n//4)
    taus = np.repeat([0.03, 0.1, 1.0, 100.0], n//4)
    if setter == "activity":
        it.set_activity(amplitudes, taus)
    else:
        for name, values in [("t_corr", taus), ("F_act", amplitudes)]:
            vectors = [Vec3(v, v, v) for v in values]
            if setter == "by_name":
                it.setPerDofVariableByName(name, vectors)
            else:
                index = next(i for i in range(it.getNumPerDofVariables())
                             if it.getPerDofVariableName(i) == name)
                it.setPerDofVariable(index, vectors)
    it.step(1)
    p = np.asarray(it.getPerDofVariableByName("p"))
    for group in range(4):
        sample = p[group*(n//4):(group+1)*(n//4)].ravel()
        F = amplitudes[group*(n//4)]
        tau = taus[group*(n//4)]
        variance = F*F*(-np.expm1(-2*0.003/tau))
        assert abs(sample.mean()) < 0.06*np.sqrt(variance)
        np.testing.assert_allclose(sample.var(), variance, rtol=0.08)


def test_changed_temperature_and_friction_have_correct_diffusion():
    n = 8192
    it = ActiveBrownianIntegrator(temperature=0, timestep=0.001)
    it.setRandomNumberSeed(304)
    ctx = _context(it, n)
    it.setStepSize(0.02)
    kbT_index = next(i for i in range(it.getNumGlobalVariables())
                    if it.getGlobalVariableName(i) == "kbT")
    it.setGlobalVariable(kbT_index, 0.4)
    it.setGlobalVariableByName("g", 2.0)
    before = _positions(ctx)
    it.step(1)
    increments = _positions(ctx)-before
    expected_mean = 0.02*np.array([2.0, -1.0, 0.0])/2
    np.testing.assert_allclose(increments.mean(axis=0), expected_mean, atol=0.006)
    np.testing.assert_allclose(increments.var(axis=0), 2*0.4*0.02/2, rtol=0.06)


@pytest.mark.parametrize("precompute", [False, True])
def test_public_activity_updates_refresh_without_resetting_ou_state(precompute):
    gen = TopologyGenerator()
    gen.gen_top([1])
    sim = ChromatinDynamics(gen.topology, platform_name="Reference", write_logs=False, console_stream=False)
    sim.simulation_setup(init_struct=np.zeros((1, 3)), integrator="active-brownian",
                         temperature=0, timestep=0.01, friction=1,
                         save_pos=False, save_energy=False, save_forcefield=False,
                         stability_check=False, precompute_coefficients=precompute)
    it = sim.integrator_manager.integrator
    assert it.precompute_coefficients is precompute
    sim.set_activity([0.0], [2.0])
    it.setPerDofVariableByName("p", [Vec3(1, 1, 1)])
    it.step(1)
    sim.set_activity([0.0], [1.0])
    it.step(1)
    np.testing.assert_allclose(np.asarray(it.getPerDofVariableByName("p")), np.exp(-0.015), atol=1e-12)


@pytest.mark.parametrize("constrained", [False, True])
def test_cuda_cached_and_uncached_trajectories_agree(constrained):
    if "CUDA" not in [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]:
        pytest.skip("CUDA unavailable")
    results = []
    for cached in [False, True]:
        it = ActiveBrownianIntegrator(temperature=24.0, timestep=0.001,
                                     precompute_coefficients=cached)
        it.setRandomNumberSeed(12345)
        ctx = _context(it, 8, platform="CUDA", constrained=constrained)
        it.set_activity([1, 0, 2, 0, 3, 0, 4, 0], [0.03, 0.1, 1, 10, 100, 1, 0.03, 100])
        it.step(200)
        results.append((_positions(ctx), np.asarray(it.getPerDofVariableByName("p"))))
        del ctx, it
    for actual, expected in zip(results[1], results[0]):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("precompute", [False, True])
def test_recorded_thermal_kick_is_the_displacement_used_for_that_step(precompute):
    dt, gamma = 0.001, 2.0
    integ = ActiveBrownianIntegrator(temperature=24, timestep=dt, collision_rate=gamma,
                                     precompute_coefficients=precompute, store_thermal_kick=True)
    integ.setRandomNumberSeed(2048)
    ctx = _context(integ, 4)
    integ.set_activity([6., 0., 6., 0.], [1.]*4)
    for _ in range(3):
        before = _positions(ctx)
        integ.step(1)
        ou = np.asarray(integ.getPerDofVariableByName("p"))
        kick = np.asarray(integ.getPerDofVariableByName("thermal_kick"))
        assert np.any(kick)
        np.testing.assert_allclose(_positions(ctx)-before, dt/gamma*(np.array([2., -1., 0.])+ou)+kick,
                                   atol=1e-12, rtol=1e-12)
