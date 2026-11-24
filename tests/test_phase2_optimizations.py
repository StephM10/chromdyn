"""Regression and parity checks for Phase 2 optimizations."""

import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from chromdyn import TopologyGenerator, hic_utils, traj_utils
from chromdyn.hic_utils import HiCManager
from chromdyn.reporters import SaveStructure
from chromdyn.traj_utils import Analyzer


def _brute_force_spatial_vel_corr(coords, velocities, bead_types, dist_range, sampling_step, num_bins):
    bins = np.linspace(0, dist_range, num_bins + 1, dtype=np.float32)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    unique_types = np.unique(bead_types)
    type_pairs = sorted(
        {"-".join(sorted((str(t1), str(t2)))) for t1 in unique_types for t2 in unique_types}
    )
    sums = {k: np.zeros(num_bins, dtype=np.float64) for k in ["general"] + type_pairs}
    counts = {k: np.zeros(num_bins, dtype=np.int64) for k in ["general"] + type_pairs}

    n_frames, n_beads, _ = coords.shape
    for f in range(0, n_frames, sampling_step):
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                d = np.linalg.norm(coords[f, i] - coords[f, j])
                if d < bins[-1]:
                    b = int(np.digitize(d, bins[1:]))
                    if b < num_bins:
                        dot = float(np.dot(velocities[f, i], velocities[f, j]))
                        key = "-".join(sorted((str(bead_types[i]), str(bead_types[j]))))
                        sums["general"][b] += dot
                        counts["general"][b] += 1
                        sums[key][b] += dot
                        counts[key][b] += 1

    out = {"bin_centers": bin_centers}
    for k in ["general"] + type_pairs:
        arr = np.full(num_bins, np.nan, dtype=np.float64)
        nz = counts[k] > 0
        arr[nz] = sums[k][nz] / counts[k][nz]
        out[k] = arr
    return out


def _brute_force_msd(positions):
    M, N, _ = positions.shape
    out = np.zeros((M, N), dtype=np.float64)
    for lag in range(1, M):
        disp = positions[lag:] - positions[:-lag]
        out[lag] = np.mean(np.sum(disp**2, axis=2), axis=0)
    return out


def _brute_force_vacf(velocities, bead_types, sampling_step=1):
    sampled = velocities[::sampling_step]
    M, N, _ = sampled.shape
    per_bead = np.zeros((N, M), dtype=np.float64)
    for lag in range(M):
        dots = np.sum(sampled[: M - lag] * sampled[lag:], axis=2)
        per_bead[:, lag] = np.mean(dots, axis=0)
    out = {"general": np.mean(per_bead, axis=0)}
    for t in np.unique(bead_types):
        out[t] = np.mean(per_bead[bead_types == t], axis=0)
    return out


@pytest.mark.parametrize("chunk_size", [None, 3])
@pytest.mark.parametrize("sampling_step", [1, 2])
def test_spatial_vel_corr_matches_brute_force(chunk_size, sampling_step):
    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 5.0, size=(5, 11, 3))
    vels = rng.normal(size=(5, 11, 3))
    bead_types = np.array(["A", "B", "C", "A", "B", "C", "A", "B", "A", "C", "B"])

    expected = _brute_force_spatial_vel_corr(
        coords, vels, bead_types, dist_range=4.5, sampling_step=sampling_step, num_bins=12
    )
    actual = Analyzer.calculate_spatial_vel_corr(
        coords,
        vels,
        bead_types,
        dist_range=4.5,
        sampling_step=sampling_step,
        num_bins=12,
        platform="CPU",
        chunk_size=chunk_size,
    )

    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("n_frames", [17, 37])
@pytest.mark.parametrize("sampling_step", [1, 2])
def test_msd_matches_brute_force_on_prime_lengths(n_frames, sampling_step):
    rng = np.random.default_rng(7)
    steps = rng.normal(scale=0.3, size=(n_frames, 7, 3))
    positions = np.cumsum(steps, axis=0)
    bead_types = np.array(["A", "B", "A", "C", "B", "A", "C"])

    sliced = positions[::sampling_step]
    expected_msd = _brute_force_msd(sliced)
    actual_msd = Analyzer.compute_msd(
        positions, batch_size=2, platform="cpu", sampling_step=sampling_step
    )
    np.testing.assert_allclose(actual_msd, expected_msd, rtol=1e-11, atol=1e-11)

    avg = Analyzer.compute_msd_average(
        positions,
        bead_types=bead_types,
        batch_size=2,
        platform="cpu",
        sampling_step=sampling_step,
    )
    np.testing.assert_allclose(avg["msd"], np.mean(expected_msd, axis=1), rtol=1e-11, atol=1e-11)
    for t in ["A", "B", "C"]:
        np.testing.assert_allclose(
            avg["type_msd"][t],
            np.mean(expected_msd[:, bead_types == t], axis=1),
            rtol=1e-11,
            atol=1e-11,
        )


def test_msd_large_coordinate_offset_avoids_cancellation():
    rng = np.random.default_rng(101)
    steps = rng.normal(scale=0.05, size=(31, 5, 3))
    unshifted = np.cumsum(steps, axis=0)
    shifted = unshifted + 1e5

    expected = _brute_force_msd(unshifted)
    actual = Analyzer.compute_msd(shifted, batch_size=2, platform="cpu")
    avg = Analyzer.compute_msd_average(shifted, batch_size=2, platform="cpu")

    np.testing.assert_array_equal(actual[0], 0.0)
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(avg["msd"], np.mean(expected, axis=1), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("sampling_step", [1, 2])
def test_vacf_matches_brute_force(sampling_step):
    rng = np.random.default_rng(19)
    velocities = rng.normal(size=(19, 6, 3))
    bead_types = np.array(["A", "B", "A", "B", "A", "B"])

    expected = _brute_force_vacf(velocities, bead_types, sampling_step=sampling_step)
    actual = Analyzer.calculate_vacf(
        velocities, bead_types, sampling_step=sampling_step, platform="CPU"
    )
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


def test_writhe_vectorized_matches_legacy_and_handles_edge_cases():
    rng = np.random.default_rng(2026)
    for n_pts in [12, 30, 50]:
        curve = rng.normal(size=(n_pts, 3))
        vec = Analyzer.compute_writhe_single_curve(curve, closed=True, method="vectorized")
        leg = Analyzer.compute_writhe_single_curve(curve, closed=True, method="legacy")
        assert vec == pytest.approx(leg, rel=1e-12, abs=1e-12)

        vec_open = Analyzer.compute_writhe_single_curve(curve, closed=False, method="vectorized")
        leg_open = Analyzer.compute_writhe_single_curve(curve, closed=False, method="legacy")
        assert vec_open == pytest.approx(leg_open, rel=1e-12, abs=1e-12)

    c1 = rng.normal(size=(18, 3))
    c2 = rng.normal(size=(22, 3)) + np.array([2.5, 0.0, 0.0])
    assert Analyzer.compute_writhe_between_curves(c1, c2, method="vectorized") == pytest.approx(
        Analyzer.compute_writhe_between_curves(c1, c2, method="legacy"),
        rel=1e-12,
        abs=1e-12,
    )

    traj = rng.normal(size=(3, 15, 3))
    w_vec = Analyzer.compute_writhe_trajectory(traj, closed=True, processes=1, method="vectorized")
    w_leg = Analyzer.compute_writhe_trajectory(traj, closed=True, processes=1, method="legacy")
    np.testing.assert_allclose(w_vec, w_leg, rtol=1e-12, atol=1e-12)

    planar = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert Analyzer.compute_writhe_single_curve(planar, closed=True, method="vectorized") == 0.0

    with pytest.raises(ValueError, match="vectorized"):
        Analyzer.compute_writhe_single_curve(planar, method="unknown")
    with pytest.raises(ValueError, match="vectorized"):
        Analyzer.compute_writhe_between_curves(c1, c2, method="unknown")
    with pytest.raises(ValueError, match="vectorized"):
        Analyzer.compute_writhe_trajectory(traj, method="unknown")


def test_hic_gpu_streams_batches_and_accumulates_float64(tmp_path, monkeypatch):
    recorded_shapes = []
    allocated_dtypes = []
    freed_calls = []

    class FakeCuPyArray(np.ndarray):
        pass

    def _wrap(arr):
        return np.asarray(arr).view(FakeCuPyArray)

    def _asarray(obj, dtype=None):
        arr = np.asarray(obj, dtype=dtype)
        recorded_shapes.append(arr.shape)
        return _wrap(arr)

    def _zeros(shape, dtype=np.float64):
        allocated_dtypes.append(np.dtype(dtype))
        return _wrap(np.zeros(shape, dtype=dtype))

    fake_cp = SimpleNamespace(
        float32=np.float32,
        float64=np.float64,
        newaxis=np.newaxis,
        asarray=_asarray,
        array=_asarray,
        zeros=_zeros,
        arange=lambda *args, **kwargs: _wrap(np.arange(*args, **kwargs)),
        round=lambda x: _wrap(np.round(np.asarray(x))),
        where=lambda cond, a, b: _wrap(np.where(np.asarray(cond), np.asarray(a), np.asarray(b))),
        tanh=lambda x: _wrap(np.tanh(np.asarray(x))),
        linalg=SimpleNamespace(norm=lambda x, axis=-1: _wrap(np.linalg.norm(np.asarray(x), axis=axis))),
        asnumpy=lambda x: np.asarray(x).copy(),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: freed_calls.append(1)),
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1, CUDARuntimeError=RuntimeError)),
    )

    monkeypatch.setattr(hic_utils, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(hic_utils, "cp", fake_cp, raising=False)

    rng = np.random.default_rng(123)
    n_frames, n_beads = 10, 5
    coords = rng.uniform(0.0, 8.0, size=(n_frames, n_beads, 3)).astype(np.float32)
    boxes = np.tile(np.eye(3) * 10.0, (n_frames, 1, 1))

    # 1. Direct array helper
    gpu_arr = hic_utils._calc_HiC_from_traj_array_gpu(
        coords, mu=2.0, rc=2.0, p=3.0, boxes=boxes, batch_size=3
    )
    cpu_arr = hic_utils._calc_pbc_hic_cpu_serial(coords, boxes, mu=2.0, rc=2.0, p=3.0)
    np.testing.assert_allclose(gpu_arr, cpu_arr, atol=1e-6)
    assert all(shape[0] <= 3 for shape in recorded_shapes if len(shape) == 3)
    assert allocated_dtypes == [np.dtype(np.float64)]
    assert len(freed_calls) == 1

    # 2. CNDB non-PBC and PBC streaming paths
    cndb_path = tmp_path / "traj.cndb"
    gen = TopologyGenerator()
    gen.gen_top([n_beads])
    writer = SaveStructure(cndb_path, PBC=True, topology=gen.topology)
    try:
        for i in range(n_frames):
            ds = writer.saveFile.create_dataset(str(i), data=coords[i])
            ds.attrs["box"] = boxes[i]
    finally:
        writer.close()

    hm = HiCManager(logger=logging.getLogger(__name__))

    recorded_shapes.clear()
    gpu_nonpbc = hm.gen_hic_from_cndb(cndb_path, mu=2.0, rc=2.0, p=3.0, platform="CUDA", batch_size=3)
    cpu_nonpbc = hm.gen_hic_from_cndb(cndb_path, mu=2.0, rc=2.0, p=3.0, platform="CPU", parallel=False)
    np.testing.assert_allclose(gpu_nonpbc, cpu_nonpbc, atol=1e-6)
    assert all(shape[0] <= 3 for shape in recorded_shapes if len(shape) == 3)

    recorded_shapes.clear()
    gpu_pbc = hm.gen_pbc_hic_from_cndb(cndb_path, mu=2.0, rc=2.0, p=3.0, platform="CUDA", batch_size=3)
    cpu_pbc = hm.gen_pbc_hic_from_cndb(cndb_path, mu=2.0, rc=2.0, p=3.0, platform="CPU", parallel=False)
    np.testing.assert_allclose(gpu_pbc, cpu_pbc, atol=1e-6)
    assert all(shape[0] <= 3 for shape in recorded_shapes if len(shape) == 3)


def test_import_without_cupy_and_compute_rg_are_quiet(capsys):
    code = (
        "import sys, warnings; sys.modules['cupy'] = None; "
        "warnings.simplefilter('error'); "
        "import chromdyn; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"
    assert proc.stderr == ""

    Analyzer.compute_RG(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    captured = capsys.readouterr()
    assert captured.out == ""


def test_collect_distances_falls_back_cleanly_without_cupy(monkeypatch):
    monkeypatch.setattr(traj_utils, "CUPY_AVAILABLE", False)
    rng = np.random.default_rng(99)
    coords = rng.normal(size=(6, 4, 3))
    bead_types = np.array(["A", "B", "A", "B"])

    center_cpu = Analyzer.collect_distances_from_center_unified(
        coords, bead_types, batch_size=2, device="cpu"
    )
    center_gpu = Analyzer.collect_distances_from_center_unified(
        coords, bead_types, batch_size=2, device="gpu"
    )
    for k in center_cpu:
        np.testing.assert_allclose(center_gpu[k], center_cpu[k])

    com_cpu = Analyzer.collect_distances_from_com_unified(
        coords, bead_types, batch_size=2, device="cpu"
    )
    com_gpu = Analyzer.collect_distances_from_com_unified(
        coords, bead_types, batch_size=2, device="gpu"
    )
    for k in com_cpu:
        np.testing.assert_allclose(com_gpu[k], com_cpu[k])


@pytest.mark.skipif(
    os.environ.get("CHROMDYN_TEST_GPU") != "1", reason="optional local GPU parity"
)
def test_real_gpu_parity_for_msd_vacf_spatial_vel_corr():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("No CUDA device")

    rng = np.random.default_rng(55)
    positions = np.cumsum(rng.normal(scale=0.1, size=(29, 12, 3)), axis=0) + 1e4
    velocities = rng.normal(size=(29, 12, 3))
    bead_types = np.array(["A", "B", "C"] * 4)

    msd_cpu = Analyzer.compute_msd(positions, batch_size=4, platform="cpu")
    msd_gpu = Analyzer.compute_msd(positions, batch_size=4, platform="gpu")
    np.testing.assert_allclose(msd_gpu, msd_cpu, rtol=1e-4, atol=1e-5)

    avg_cpu = Analyzer.compute_msd_average(positions, bead_types=bead_types, batch_size=4, platform="cpu")
    avg_gpu = Analyzer.compute_msd_average(positions, bead_types=bead_types, batch_size=4, platform="gpu")
    np.testing.assert_allclose(avg_gpu["msd"], avg_cpu["msd"], rtol=1e-4, atol=1e-5)

    vacf_cpu = Analyzer.calculate_vacf(velocities, bead_types, platform="CPU")
    vacf_gpu = Analyzer.calculate_vacf(velocities, bead_types, platform="CUDA")
    for k in vacf_cpu:
        np.testing.assert_allclose(vacf_gpu[k], vacf_cpu[k], rtol=1e-5, atol=1e-6)

    svc_cpu = Analyzer.calculate_spatial_vel_corr(
        positions - 1e4, velocities, bead_types, dist_range=3.0, num_bins=10, platform="CPU", chunk_size=4
    )
    svc_gpu = Analyzer.calculate_spatial_vel_corr(
        positions - 1e4, velocities, bead_types, dist_range=3.0, num_bins=10, platform="CUDA", chunk_size=4
    )
    for k in svc_cpu:
        np.testing.assert_allclose(svc_gpu[k], svc_cpu[k], rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("name", ["active-langevin", "active-brownian"])
def test_active_integrator_manager_logs_set_activity(caplog, name):
    from chromdyn.integrators import IntegratorManager

    test_logger = logging.getLogger(f"chromdyn.test.{name}")
    test_logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger=test_logger.name):
        IntegratorManager(logger=test_logger, integrator=name)
    assert ".set_activity(" in caplog.text
    assert ".set_active_params(" not in caplog.text
