import numpy as np
import pytest
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for testing

from chromdyn.traj_utils import Analyzer
from chromdyn.visualization import visualize

class MockTrajectory:
    def __init__(self, positions):
        self.positions = positions # shape (T, N, 3)
        self.n_frames = positions.shape[0]
        # mock topology
        class MockTopology:
            def chains(self):
                class MockChain:
                    def atoms(self):
                        class MockAtom:
                            def __init__(self, index):
                                self.index = index
                        return [MockAtom(i) for i in range(positions.shape[1])]
                return [MockChain()]
        self.topology = MockTopology()

    def xyz(self, frames, bead_selection=None):
        if bead_selection is not None:
            return self.positions[frames[0]:frames[1]:frames[2]][:, bead_selection, :]
        return self.positions[frames[0]:frames[1]:frames[2]]

def test_mass_weighted_rg():
    # 1. Create a dummy coordinate set (1 frame, 2 particles)
    # P1 at (0, 0, 0), P2 at (10, 0, 0)
    pos = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    
    # 2. Geometric Rg (uniform mass)
    # COM is (5, 0, 0).
    # Deviations: P1 -> 5^2=25, P2 -> 5^2=25. Average deviation = 25.
    # Rg = sqrt(25) = 5.0
    rg_geom = Analyzer.compute_RG(pos, return_components=False, bead_masses=None)
    np.testing.assert_allclose(rg_geom[0], 5.0, err_msg="Geometric Rg failed")

    # 3. Mass-weighted Rg
    # P1 mass = 9.0, P2 mass = 1.0. Total mass = 10.0
    # COM = (0*9 + 10*1) / 10 = (1, 0, 0)
    # Deviations:
    # P1 -> (-1)^2 = 1. Weight = 9.
    # P2 -> (9)^2 = 81. Weight = 1.
    # Weighted average deviation = (1*9 + 81*1) / 10 = (9 + 81)/10 = 90/10 = 9.0
    # Rg = sqrt(9) = 3.0
    masses = np.array([9.0, 1.0])
    rg_mass = Analyzer.compute_RG(pos, return_components=False, bead_masses=masses)
    np.testing.assert_allclose(rg_mass[0], 3.0, err_msg="Mass-weighted Rg failed")
    print("test_mass_weighted_rg passed!")

def test_hetero_visualization(tmp_path):
    # 1. Create mock trajectory
    n_beads = 5
    pos = np.random.rand(1, n_beads, 3) * 10.0
    traj = MockTrajectory(pos)
    
    # 2. Create heterogeneous radii array
    radii = np.array([0.5, 1.0, 0.5, 2.0, 1.0])
    
    # 3. Test visualize function execution
    output_file = str(tmp_path / "test_viz.png")
    try:
        visualize(
            traj=traj,
            select_frame=0,
            r=radii,
            output_name=output_file
        )
        assert os.path.exists(output_file), "Visualization output file was not created"
    except Exception as e:
        pytest.fail(f"visualize with radii array crashed: {e}")
        
    print("test_hetero_visualization passed!")

if __name__ == "__main__":
    import tempfile
    test_mass_weighted_rg()
    with tempfile.TemporaryDirectory() as tmp_path:
        from pathlib import Path
        test_hetero_visualization(Path(tmp_path))
