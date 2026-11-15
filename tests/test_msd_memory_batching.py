import numpy as np
import h5py

from chromdyn.traj_utils import Analyzer, Trajectory


def _write_cndb(path, coords):
    with h5py.File(path, "w") as h5:
        for i, frame in enumerate(coords):
            h5.create_dataset(str(i), data=frame)
        h5.create_dataset("types", data=np.array([b"A", b"B", b"A"]))


def test_xyz_can_materialize_memmap_in_frame_batches(tmp_path):
    coords = np.arange(5 * 3 * 3, dtype=np.float64).reshape(5, 3, 3)
    cndb_path = tmp_path / "toy.cndb"
    memmap_path = tmp_path / "toy_xyz.dat"
    _write_cndb(cndb_path, coords)

    traj = Trajectory(str(cndb_path))
    observed = traj.xyz(
        frames=[0, None, 1],
        frame_batch_size=2,
        dtype=np.float32,
        storage="memmap",
        memmap_path=str(memmap_path),
    )

    assert isinstance(observed, np.memmap)
    assert observed.dtype == np.float32
    assert observed.shape == coords.shape
    np.testing.assert_allclose(observed, coords.astype(np.float32))


def test_compute_msd_average_matches_bead_resolved_msd_means():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    bead_types = np.array(["A", "B", "A"])
    bead_msd = Analyzer.compute_msd(positions, batch_size=2, platform="cpu")

    observed = Analyzer.compute_msd_average(
        positions,
        bead_types=bead_types,
        batch_size=2,
        platform="cpu",
    )

    np.testing.assert_allclose(observed["msd"], np.mean(bead_msd, axis=1))
    np.testing.assert_allclose(
        observed["type_msd"]["A"], np.mean(bead_msd[:, [0, 2]], axis=1)
    )
    np.testing.assert_allclose(observed["type_msd"]["B"], bead_msd[:, 1])
    assert observed["type_counts"] == {"A": 2, "B": 1}
