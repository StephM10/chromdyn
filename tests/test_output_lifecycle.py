import logging

import numpy as np
import pytest

from chromdyn import ChromatinDynamics, TopologyGenerator
from chromdyn.utilities import LogManager


def simulation(path, write_logs=False):
    gen = TopologyGenerator()
    gen.gen_top([2])
    return ChromatinDynamics(
        gen.topology,
        name="same",
        platform_name="Reference",
        output_dir=str(path),
        write_logs=write_logs,
        console_stream=False,
    )


def setup(sim, **kwargs):
    settings = dict(
        init_struct=np.array([[0.0, 0, 0], [1, 0, 0]]),
        save_pos=False,
        save_energy=False,
        save_forcefield=False,
        stability_check=False,
    )
    settings.update(kwargs)
    sim.simulation_setup(**settings)


@pytest.mark.parametrize("report", ["save_pos", "save_energy", "stability_check"])
def test_reports_can_be_saved_without_log_file(tmp_path, report):
    sim = simulation(tmp_path / "out")
    try:
        setup(sim, **{report: True})
        sim.run(1, verbose=False)
        sim.save_reports()
        assert sim.output_dir.is_dir()
        assert not list(sim.output_dir.glob("*.log"))
    finally:
        sim.save_reports()


def test_no_output_run_does_not_create_output_directory(tmp_path):
    output = tmp_path / "unused"
    sim = simulation(output)
    setup(sim)
    sim.run(1, verbose=False)
    assert not output.exists()


@pytest.mark.parametrize("raises", [False, True])
def test_report_false_restores_all_reporters_and_pause_state(
    tmp_path, monkeypatch, raises
):
    sim = simulation(tmp_path)
    setup(sim, stability_check=True)

    class External:
        is_paused = True

    external = External()
    sim.simulation.reporters.append(external)
    before = sim.simulation.reporters

    def step(n):
        assert sim.simulation.reporters == []
        if raises:
            raise RuntimeError("expected step failure")

    monkeypatch.setattr(sim.simulation, "step", step)
    try:
        if raises:
            with pytest.raises(RuntimeError, match="expected step"):
                sim.run(1, report=False, verbose=False)
        else:
            assert sim.run(1, report=False, verbose=False) is None
        assert sim.simulation.reporters is before
        assert external.is_paused
    finally:
        sim.save_reports()


def test_save_reports_closes_all_owned_files_and_detaches_them(tmp_path):
    sim = simulation(tmp_path)
    setup(sim, save_pos=True, save_energy=True, stability_check=True)
    reporters = list(sim.reporters.values())
    external = object()
    sim.simulation.reporters.append(external)
    sim.save_reports()
    sim.save_reports()
    assert all(
        r.saveFile.closed if hasattr(r.saveFile, "closed") else not r.saveFile.id.valid
        for r in reporters
    )
    assert sim.simulation.reporters == [external]


def test_simulation_logs_are_isolated_and_closed_explicitly(tmp_path):
    a = simulation(tmp_path / "a", write_logs=True)
    b = simulation(tmp_path / "b", write_logs=True)
    a.logger.info("only-in-A")
    b.logger.info("only-in-B")
    handlers = [*a.logger.handlers, *b.logger.handlers]
    a.close()
    b.close()
    text_a = (tmp_path / "a" / "same.log").read_text()
    text_b = (tmp_path / "b" / "same.log").read_text()
    assert "only-in-A" in text_a and "only-in-B" not in text_a
    assert "only-in-B" in text_b and "only-in-A" not in text_b
    assert all(h.stream is None for h in handlers if isinstance(h, logging.FileHandler))


def test_module_logger_configuration_is_not_replaced():
    logger = logging.getLogger("chromdyn.audit.logger")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        assert LogManager().get_logger(logger.name) is logger
        assert handler in logger.handlers
        assert logger.level == logging.WARNING
    finally:
        logger.removeHandler(handler)
