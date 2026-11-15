"""Regression checks for short runs and public activity validation."""

import numpy as np
import pytest

from chromdyn import ChromatinDynamics, TopologyGenerator


def _simulation():
    generator = TopologyGenerator()
    generator.gen_top([1])
    return ChromatinDynamics(
        generator.topology, platform_name="Reference",
        write_logs=False, console_stream=False,
    )


def _setup(sim):
    sim.simulation_setup(
        init_struct=np.zeros((1, 3)), integrator="active-brownian",
        temperature=0, save_pos=False, save_energy=False,
        save_forcefield=False, stability_check=False,
    )


def test_short_run_survives_zero_measured_elapsed_time(monkeypatch):
    sim = _simulation()
    _setup(sim)
    monkeypatch.setattr("chromdyn.chromatin_dynamics.time.perf_counter", lambda: 1.0)
    sim.run(1, report=False)
    assert sim.simulation.currentStep == 1


def test_activity_requires_simulation_setup():
    with pytest.raises(RuntimeError, match="simulation_setup"):
        _simulation().set_activity([0.0], [1.0])


@pytest.mark.parametrize("amplitudes,times", [
    ([[1.0]], [1.0]), ([1.0], [0.0]), ([float("nan")], [1.0]),
])
def test_invalid_public_activity_does_not_modify_parameters(amplitudes, times):
    sim = _simulation()
    _setup(sim)
    sim.set_activity([2.0], [3.0])
    with pytest.raises(ValueError):
        sim.set_activity(amplitudes, times)
    params = sim.get_active_params()
    np.testing.assert_array_equal(params["F_act"], [[2.0]*3])
    np.testing.assert_array_equal(params["t_corr"], [[3.0]*3])
