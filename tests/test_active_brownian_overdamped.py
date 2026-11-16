import numpy as np
from openmm import Vec3

from chromdyn.chromatin_dynamics import ChromatinDynamics
from chromdyn.topology import TopologyGenerator


def test_active_brownian_zero_activity_ignores_state_velocity():
    gen = TopologyGenerator()
    gen.gen_top(chain_lens=[1], types=["A"])
    sim = ChromatinDynamics(
        gen.topology,
        name="active_brownian_zero_activity",
        platform_name="Reference",
        output_dir="output/test_active_brownian",
        console_stream=False,
        mass=1.0,
        write_logs=False,
    )
    sim.force_field_manager.apply_oscillatory_tweezer(
        mono_id=0, k_ot=0.0, A=0.0, omega=0.0, pos=(0.0, 0.0, 0.0)
    )
    sim.simulation_setup(
        init_struct=np.zeros((1, 3)),
        integrator="active-brownian",
        temperature=0.0,
        timestep=0.001,
        friction=1.0,
        save_pos=False,
        save_energy=False,
        save_forcefield=False,
        stability_check=False,
        advance_time=True,
    )
    sim.set_activity(F_seq=np.zeros(1), tau_seq=[1.0])
    sim.simulation.context.setVelocities([Vec3(10.0, 0.0, 0.0)])
    sim.simulation.context.setParameter("t", 0.0)

    sim.simulation.step(100)

    pos = sim.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    assert abs(pos[0, 0]._value) < 1e-12
    assert abs(sim.simulation.context.getParameter("t") - 0.1) < 1e-12
