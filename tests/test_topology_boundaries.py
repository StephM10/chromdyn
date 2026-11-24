import pytest

from chromdyn import TopologyGenerator


@pytest.mark.parametrize(
    "lengths,per_res,sizes,bond_count",
    [
        ([2], 3, [2], 1),
        ([3, 2], 3, [3, 2], 3),
        ([8], 3, [3, 3, 2], 7),
        ([6], 3, [3, 3], 5),
    ],
)
def test_short_and_remainder_residues_preserve_chain_bonds(
    lengths, per_res, sizes, bond_count
):
    gen = TopologyGenerator()
    gen.gen_top(lengths, atoms_per_residue=per_res)
    assert [len(list(r.atoms())) for r in gen.topology.residues()] == sizes
    assert gen.topology.getNumBonds() == bond_count
    for a, b in gen.topology.bonds():
        assert a is not None and b is not None
        assert a.residue.chain is b.residue.chain
    for chain in gen.topology.chains():
        names = [r.name for r in chain.residues()]
        assert len(set(names)) == len(names)


def test_arbitrary_single_type_name_is_preserved():
    gen = TopologyGenerator()
    gen.gen_top([2], types="long chromatin type")
    assert gen.atom_types == ["long chromatin type"] * 2


def test_nonpositive_residue_size_is_rejected():
    with pytest.raises(ValueError, match="atoms_per_residue"):
        TopologyGenerator().gen_top([2], atoms_per_residue=0)
