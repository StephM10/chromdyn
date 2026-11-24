import logging
from types import SimpleNamespace

import pytest
from openmm import CustomIntegrator, LangevinIntegrator, System, unit

from chromdyn.integrators import ActiveBrownianIntegrator, ActiveLangevinIntegrator
from chromdyn.reporters import StabilityReporter


def report_input(integ, kinetic=0, potential=100):
    system = System()
    system.addParticle(1)
    calls = []
    context = SimpleNamespace(
        setVelocitiesToTemperature=lambda *args: calls.append(args)
    )
    simulation = SimpleNamespace(
        system=system, integrator=integ, context=context, currentStep=100
    )
    state = SimpleNamespace(
        getKineticEnergy=lambda: kinetic * unit.kilojoule_per_mole,
        getPotentialEnergy=lambda: potential * unit.kilojoule_per_mole,
    )
    return simulation, state, calls


def test_active_langevin_uses_live_thermal_energy(tmp_path):
    it = ActiveLangevinIntegrator(10, 1, 0.01)
    it.setGlobalVariableByName("kbT", 0.008314 * 20)
    sim, state, calls = report_input(it)
    reporter = StabilityReporter(
        tmp_path / "report.txt",
        potential_threshold=5,
        logger=logging.getLogger(__name__),
    )
    try:
        reporter.report(sim, state)
        assert calls[0][0] == pytest.approx(20)
    finally:
        reporter.saveFile.close()


def test_overdamped_report_does_not_reset_velocities(tmp_path):
    sim, state, calls = report_input(ActiveBrownianIntegrator(temperature=10))
    p = tmp_path / "report.txt"
    reporter = StabilityReporter(
        p, potential_threshold=5, logger=logging.getLogger(__name__)
    )
    try:
        reporter.report(sim, state)
        assert calls == []
        assert "INSTABILITY" in p.read_text()
    finally:
        reporter.saveFile.close()


def test_zero_temperature_performs_finite_checks_without_division(tmp_path):
    sim, state, calls = report_input(LangevinIntegrator(0, 1, 0.01))
    reporter = StabilityReporter(
        tmp_path / "report.txt", logger=logging.getLogger(__name__)
    )
    try:
        reporter.report(sim, state)
        assert calls == []
    finally:
        reporter.saveFile.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_energy_is_reported_and_stops_run(tmp_path, bad):
    sim, state, calls = report_input(LangevinIntegrator(10, 1, 0.01), potential=bad)
    p = tmp_path / "report.txt"
    reporter = StabilityReporter(p, logger=logging.getLogger(__name__))
    try:
        with pytest.raises(FloatingPointError, match="finite"):
            reporter.report(sim, state)
        assert calls == []
        assert "Non-finite" in p.read_text()
    finally:
        reporter.saveFile.close()


def test_unknown_custom_integrator_requires_explicit_energy_reference(tmp_path):
    sim, state, _ = report_input(CustomIntegrator(0.01))
    reporter = StabilityReporter(
        tmp_path / "report.txt", logger=logging.getLogger(__name__)
    )
    try:
        with pytest.raises(ValueError, match="reference_energy"):
            reporter.report(sim, state)
    finally:
        reporter.saveFile.close()


def test_unknown_custom_integrator_with_reference_does_not_invent_temperature(tmp_path):
    sim, state, calls = report_input(CustomIntegrator(0.01))
    p = tmp_path / "reference.txt"
    reporter = StabilityReporter(p, reference_energy=1, potential_threshold=5)
    try:
        reporter.report(sim, state)
        assert calls == []
        assert "INSTABILITY" in p.read_text()
    finally:
        reporter.close()
