import os

import h5py
import numpy as np
import pytest

from chromdyn import TopologyGenerator
from chromdyn.traj_utils import Trajectory, TrajectoryLoader
from chromdyn.reporters import SaveStructure


def write_cndb(path, keys, coords=None, box=None):
    if coords is None:
        coords = np.array([[[int(k), 0.0, 0.0], [int(k), 1.0, 0.0]] for k in keys])
    with h5py.File(path, "w") as f:
        f["types"] = np.array([b"A"] * len(coords[0]))
        for i, key in enumerate(keys):
            ds = f.create_dataset(str(key), data=coords[i])
            if box is not None:
                ds.attrs["box"] = box if np.ndim(box) == 2 else box[i]


def test_legacy_loader_orders_frames_numerically_and_keeps_id_modulo_sampling(tmp_path):
    p = tmp_path / "frames.cndb"
    write_cndb(p, range(12))
    np.testing.assert_array_equal(TrajectoryLoader.load(p)[:, 0, 0], np.arange(12))
    np.testing.assert_array_equal(
        TrajectoryLoader.load(p, d=2)[:, 0, 0], np.arange(0, 12, 2)
    )


@pytest.mark.parametrize("keys", [[0, 2, 10], [1, 2, 3], ["00", "02", "10"]])
@pytest.mark.parametrize("mode", ["memory", "batch", "memmap"])
def test_xyz_uses_sorted_frame_positions_in_all_storage_modes(tmp_path, keys, mode):
    p = tmp_path / "frames.cndb"
    write_cndb(p, keys)
    t = Trajectory(str(p))
    kwargs = {} if mode == "memory" else dict(frame_batch_size=1, dtype=np.float32)
    if mode == "memmap":
        kwargs.update(storage="memmap", memmap_path=str(tmp_path / "coords.dat"))
    try:
        xyz = t.xyz(frames=[1, None, 1], **kwargs)
        np.testing.assert_array_equal(xyz[:, 0, 0], [int(k) for k in keys[1:]])
        assert t.frame_keys == [str(k) for k in keys]
    finally:
        t.close()


@pytest.mark.parametrize(
    "start,end,stride,expected",
    [(0, None, 2, [1, 3, 5]), (1, None, 2, [2, 4]), (1, 5, 2, [2, 4])],
)
def test_distance_distribution_selects_frames_once(
    tmp_path, start, end, stride, expected
):
    p = tmp_path / "dist.cndb"
    coords = np.array([[[0.0, 0, 0], [i + 1.0, 0, 0]] for i in range(5)])
    write_cndb(p, range(5), coords)
    t = Trajectory(str(p))
    try:
        actual = t.get_distance_distribution_by_separation_and_type(
            start_frame=start, end_frame=end, sampling_step=stride, device="cpu"
        )
        np.testing.assert_allclose(actual["A-A"], expected)
    finally:
        t.close()


def test_wrapped_distances_use_minimum_image(tmp_path):
    p = tmp_path / "pbc.cndb"
    write_cndb(p, [0], np.array([[[0.1, 0, 0], [9.9, 0, 0]]]), np.eye(3) * 10)
    t = Trajectory(str(p))
    try:
        actual = t.get_distance_distribution_by_separation_and_type(
            wrapped=True, device="cpu"
        )
        np.testing.assert_allclose(actual["A-A"], [0.2], atol=1e-12)
    finally:
        t.close()


def test_contour_separation_does_not_pair_across_chains(tmp_path):
    p = tmp_path / "chains.cndb"
    write_cndb(
        p, [0], np.array([[[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0]]])
    )
    t = Trajectory(str(p))
    gen = TopologyGenerator()
    gen.gen_top([2, 2])
    t.topology = gen.topology
    try:
        actual = t.get_distance_distribution_by_separation_and_type(device="cpu")
        np.testing.assert_allclose(actual["A-A"], [1, 1])
    finally:
        t.close()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_native_topology_sparse_frames_and_boxes_stay_aligned(tmp_path, device):
    if device == "gpu":
        if os.environ.get("CHROMDYN_TEST_GPU") != "1":
            pytest.skip("optional local GPU parity")
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    p = tmp_path / "native.cndb"
    gen = TopologyGenerator()
    gen.gen_top([2, 2], types=["A", "B", "A", "B"])
    writer = SaveStructure(p, PBC=True, topology=gen.topology)
    try:
        for key, length in zip(["00", "04", "10"], [10.0, 20.0, 30.0]):
            ds = writer.saveFile.create_dataset(
                key, data=[[0.1, 0, 0], [length - 0.1, 0, 0], [4.0, 0, 0], [5.0, 0, 0]]
            )
            ds.attrs["box"] = np.eye(3) * length
    finally:
        writer.close()
    t = Trajectory(str(p))
    try:
        result = t.get_distance_distribution_by_separation_and_type(
            sampling_step=2, wrapped=True, device=device, batch_size=1
        )
        np.testing.assert_allclose(result["A-B"], [0.2, 1.0, 0.2, 1.0], atol=1e-12)
        assert t.xyz_wrapped(frames=[1, None, 2], xyz_cols=[0]).shape == (1, 4, 1)
    finally:
        t.close()


def test_empty_frame_selection_returns_empty_distribution(tmp_path):
    p = tmp_path / "frames.cndb"
    write_cndb(p, [0])
    t = Trajectory(str(p))
    try:
        assert (
            t.get_distance_distribution_by_separation_and_type(
                start_frame=1, device="cpu"
            )
            == {}
        )
    finally:
        t.close()
