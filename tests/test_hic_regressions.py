import logging
import multiprocessing
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from chromdyn.hic_utils import HiCManager
from chromdyn import hic_utils


def write_frames(path, count=21, padded=False):
    with h5py.File(path, "w") as f:
        f["types"] = np.array([b"A", b"A"])
        for i in range(count):
            distance = 0.5 if i < count - 1 else 4.0
            ds = f.create_dataset(
                f"{i:03d}" if padded else str(i), data=[[0.0, 0, 0], [distance, 0, 0]]
            )
            ds.attrs["box"] = np.eye(3) * 20


def manager():
    return HiCManager(logger=logging.getLogger(__name__))


def test_missing_gpu_falls_back_to_streaming_cpu(tmp_path, monkeypatch):
    p = tmp_path / "frames.cndb"
    write_frames(p, 3)
    monkeypatch.setattr(hic_utils, "CUPY_AVAILABLE", False)
    expected = manager().gen_hic_from_cndb(p, platform="CPU", parallel=False)
    actual = manager().gen_hic_from_cndb(p, platform="CUDA", parallel=False)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("reason", ["no_device", "driver"])
def test_gpu_capability_failure_falls_back_to_cpu(tmp_path, monkeypatch, reason):
    p = tmp_path / "frames.cndb"
    write_frames(p, 3)

    class DriverError(RuntimeError):
        pass

    def devices():
        if reason == "driver":
            raise DriverError("driver unavailable")
        return 0

    runtime = SimpleNamespace(getDeviceCount=devices, CUDARuntimeError=DriverError)
    monkeypatch.setattr(hic_utils, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        hic_utils,
        "cp",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
        raising=False,
    )
    expected = manager().gen_hic_from_cndb(p, platform="CPU", parallel=False)
    actual = manager().gen_hic_from_cndb(p, platform="CUDA", parallel=False)
    np.testing.assert_allclose(actual, expected)


def test_zero_padded_frame_keys_work_in_streaming_and_array_readers(tmp_path):
    p = tmp_path / "frames.cndb"
    write_frames(p, 3, padded=True)
    hm = manager()
    xyz = hm.cndb_to_numpy(p, skip_frames=2)
    assert xyz.shape == (2, 2, 3)
    result = hm.gen_hic_from_cndb(p, skip_frames=2, parallel=False)
    assert result[0, 1] == pytest.approx(
        np.mean([0.5 * (1 + np.tanh(3)), 0.5 * (1 + np.tanh(-4))])
    )


@pytest.mark.parametrize("power", [None, 3.0])
@pytest.mark.parametrize("parallel", [False, True])
def test_pbc_hic_includes_all_frames_and_excludes_self_contacts(
    tmp_path, monkeypatch, power, parallel
):
    p = tmp_path / "frames.cndb"
    write_frames(p)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 2)
    result = manager().gen_pbc_hic_from_cndb(
        p, platform="CPU", parallel=parallel, p=power
    )
    tail = 0.5 * (1 + np.tanh(-4)) if power is None else 0.5 * (2 / 4) ** power
    expected = (20 * 0.5 * (1 + np.tanh(3)) + tail) / 21
    np.testing.assert_allclose(np.diag(result), 0, atol=0)
    assert result[0, 1] == pytest.approx(expected, rel=2e-6)


def test_random_contact_map_is_symmetric_nonzero_and_has_zero_diagonal():
    result = manager().get_numpy_from_hic(8)
    assert result.shape == (8, 8)
    np.testing.assert_array_equal(result, result.T)
    np.testing.assert_array_equal(np.diag(result), 0)
    assert np.any(result > 0)
    assert np.all((0 <= result) & (result < 1))


def test_normalization_is_independent_of_excluded_diagonal_values():
    matrix = np.arange(36, dtype=float).reshape(6, 6) + 1
    matrix = matrix + matrix.T
    changed = matrix.copy()
    np.fill_diagonal(changed, 1000)
    hm = manager()
    np.testing.assert_allclose(hm.normalize_hic(changed), hm.normalize_hic(matrix))


@pytest.mark.skipif(
    os.environ.get("CHROMDYN_TEST_GPU") != "1", reason="optional local GPU parity"
)
def test_gpu_pbc_hic_matches_cpu(tmp_path):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("No CUDA device")
    p = tmp_path / "frames.cndb"
    write_frames(p)
    hm = manager()
    for power in [None, 3.0]:
        cpu = hm.gen_pbc_hic_from_cndb(p, platform="CPU", parallel=False, p=power)
        gpu = hm.gen_pbc_hic_from_cndb(p, platform="CUDA", batch_size=4, p=power)
        np.testing.assert_allclose(gpu, cpu, atol=1e-6)


def test_hic_map_and_shape_accessors_reflect_loaded_map(tmp_path):
    hm = manager()
    assert hm.get_hic_map() is None
    assert hm.get_shape() is None

    loaded = hm.get_numpy_from_hic(4)
    assert hm.get_hic_map() is loaded
    assert hm.get_shape() == (4, 4)

    txt = tmp_path / "hic.txt"
    np.savetxt(txt, [[0.0, 0.5], [0.5, 0.0]])
    loaded_txt = hm.get_numpy_from_hic(str(txt))
    assert hm.get_hic_map() is loaded_txt
    assert hm.get_shape() == (2, 2)
