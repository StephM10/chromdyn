import numpy as np
from pathlib import Path
from chromdyn.topology import TopologyGenerator
from chromdyn.chromatin_dynamics import ChromatinDynamics

def test_hetero_mass_radius(tmp_path):
    # 1. Generate topology
    top_gen = TopologyGenerator()
    chain_lens = [10]
    top_gen.gen_top(chain_lens=chain_lens, types="A")

    # 2. Setup masses and radii
    num_particles = sum(chain_lens)
    masses = np.random.uniform(1.0, 2.0, size=num_particles)
    radii = np.random.uniform(0.5, 1.0, size=num_particles)
    sigmas = radii * 2.0  # Just an example
    
    # 3. Initialize Dynamics
    # This should work after our modification
    sim = ChromatinDynamics(
        topology=top_gen.topology,
        name="test_hetero",
        output_dir=str(tmp_path),
        mass=masses,
        console_stream=False
    )
    
    # 4. Add forcefield with varying radii
    # This should work after our modification
    sim.force_field_manager.add_self_avoidance(radii=radii)
    sim.force_field_manager.add_LJ_repulsion(radii=sigmas)
    
    # 5. Setup simulation to test if OpenMM accepts the custom parameters
    sim.simulation_setup(
        integrator="langevin",
        temperature=120.0,
        timestep=0.01,
        friction=0.1,
        save_pos=False,
        save_energy=False,
        save_forcefield=True,
        stability_check=False
    )
    
    # Take 1 step to ensure everything compiles and runs
    sim.run(n_steps=1, verbose=True, report=False)
    
    print("Test passed! Heterogeneous mass and radius correctly executed.")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        test_hetero_mass_radius(tmp_path)
