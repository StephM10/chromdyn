"""Public-API regression cases found during the final Phase 2 review."""

import os

import numpy as np
import pytest

from chromdyn.traj_utils import Analyzer


def _require_gpu():
    if os.environ.get("CHROMDYN_TEST_GPU") != "1":
        pytest.skip("optional local GPU parity")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("No CUDA device")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("entry", ["beads", "average"])
def test_gpu_ballistic_msd_keeps_small_lags_accurate(dtype, entry):
    _require_gpu()
    coords = np.zeros((20003, 1, 3), dtype=dtype)
    coords[:, 0, 0] = np.arange(len(coords))
    if entry == "beads":
        actual = Analyzer.compute_msd(coords, platform="gpu", batch_size=1)[:, 0]
    else:
        actual = Analyzer.compute_msd_average(coords, platform="gpu", batch_size=1)[
            "msd"
        ]
    # x(t)=t, so every displacement at lag k is exactly k: MSD(k)=k^2.
    np.testing.assert_allclose(actual[:8], np.arange(8) ** 2, rtol=1e-8, atol=5e-6)
    assert np.all(actual >= 0)


@pytest.mark.parametrize("offset", [0.0, 1e5])
def test_spatial_correlation_preserves_float64_bin_membership_on_gpu(offset):
    _require_gpu()
    coords = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]) + offset
    velocities = np.ones_like(coords)
    types = np.array(["A", "B"])
    cpu = Analyzer.calculate_spatial_vel_corr(
        coords, velocities, types, 1.0, num_bins=10, platform="CPU"
    )
    gpu = Analyzer.calculate_spatial_vel_corr(
        coords, velocities, types, 1.0, num_bins=10, platform="CUDA"
    )
    # 0.1 in float64 lies below the first float32 edge 0.10000000149...
    assert cpu["general"][0] == 3
    assert np.count_nonzero(np.isfinite(cpu["general"])) == 1
    for key in cpu:
        np.testing.assert_allclose(gpu[key], cpu[key], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("method", ["vectorized", "legacy"])
@pytest.mark.parametrize("entry", ["single", "between", "trajectory"])
def test_writhe_rejects_nonfinite_coordinates(bad, method, entry):
    curve = np.random.default_rng(123).normal(size=(5, 3))
    corrupt = curve.copy()
    corrupt[1, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        if entry == "single":
            Analyzer.compute_writhe_single_curve(corrupt, method=method)
        elif entry == "between":
            Analyzer.compute_writhe_between_curves(curve, corrupt, method=method)
        else:
            Analyzer.compute_writhe_trajectory(
                np.array([corrupt]), processes=1, method=method
            )
