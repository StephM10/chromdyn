import logging
from types import SimpleNamespace

import numpy as np
import pytest
from openmm import Context, Platform, System, Vec3, VerletIntegrator

from chromdyn import TopologyGenerator, save_pdb


@pytest.mark.parametrize("pbc", [False, True])
def test_pdb_positions_and_box_share_angstrom_units(tmp_path, pbc):
    gen = TopologyGenerator()
    gen.gen_top([2])
    system = System()
    system.addParticle(1)
    system.addParticle(1)
    system.setDefaultPeriodicBoxVectors(Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10))
    integ = VerletIntegrator(0.001)
    context = Context(system, integ, Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [1, 2, 3]])
    p = tmp_path / "units.pdb"
    obj = SimpleNamespace(
        name="units",
        output_dir=tmp_path,
        topology=gen.topology,
        logger=logging.getLogger(__name__),
        simulation=SimpleNamespace(context=context, currentStep=1),
    )
    save_pdb(obj, filename=str(p), PBC=pbc)
    lines = p.read_text().splitlines()
    atom = [s for s in lines if s.startswith("ATOM")][1]
    np.testing.assert_allclose(
        [float(atom[30:38]), float(atom[38:46]), float(atom[46:54])], [10, 20, 30]
    )
    if pbc:
        box = next(s for s in lines if s.startswith("CRYST1"))
        np.testing.assert_allclose(
            [float(box[6:15]), float(box[15:24]), float(box[24:33])], [100, 100, 100]
        )
