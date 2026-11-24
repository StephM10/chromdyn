#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *
from openmm.app import Topology
from pathlib import Path
import numpy as np
from typing import List, Optional, Union


class TopologyGenerator:
    def __init__(self) -> None:
        """
        Initialize a new TopologyGenerator with an empty OpenMM Topology object.
        """
        self.topology: Topology = Topology()  # OpenMM Topology container
        self.atom_types: Optional[List[str]] = (
            None  # Will hold atom type names after topology generation
        )

    def gen_top(
        self,
        chain_lens: List[int],
        types: Union[str, List[str]] = "A",
        chain_names: Optional[List[str]] = None,
        isRing: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Generate a molecular topology with chains, residues, atoms, and bonds.

        Parameters:
        - chain_lens (List[int]): Number of atoms in each chain.
        - types (str | List[str]): Atom type assignment. Can be a single type, a list, 'unique', or a filename.
        - chain_names (List[str], optional): Names for each chain. Defaults to ['C1', 'C2', ...].
        - isRing (List[int], optional): Marks chains that should be closed with a bond from last to first atom.
        - kwargs:
            - atoms_per_residue (int): Number of atoms per residue. Default is 1.
        """
        # Number of atoms in each residue
        atoms_per_residue: int = int(kwargs.get("atoms_per_residue", 1))
        if atoms_per_residue < 1:
            raise ValueError("atoms_per_residue must be positive")

        # Default chain names if not provided
        if chain_names is None:
            chain_names = [f"C{xx}" for xx in range(1, len(chain_lens) + 1)]

        # Default isRing flag as 0 (non-cyclic) if not provided
        if isRing is None:
            isRing = [0] * len(chain_lens)

        atom_types: List[str] = []

        # Determine atom types from input
        if isinstance(types, str):
            type_file = Path(types)
            if type_file.exists():
                # Load from file if it exists
                atom_types = np.loadtxt(type_file, usecols=[1], dtype=str).tolist()
            elif types == "unique":
                # Assign unique types for each atom
                atom_types = [f"M{xx}" for xx in range(1, sum(chain_lens) + 1)]
            else:
                # Use single type for all atoms
                atom_types = [types] * sum(chain_lens)
        elif isinstance(types, list):
            atom_types = types

        # Validation checks
        assert len(chain_names) == len(
            chain_lens
        ), "chain_names do not match chain_lens"
        assert len(atom_types) == sum(
            chain_lens
        ), "types file does not match chain_lens"

        kk: int = 0  # Atom counter across all chains

        for cid, chain_len in enumerate(chain_lens):
            chain = self.topology.addChain(chain_names[cid])  # Add a new chain
            previous_atom = None
            for local_index in range(chain_len):
                resid, aid = divmod(local_index, atoms_per_residue)
                if aid == 0:
                    residue = self.topology.addResidue(f"L{resid+1}", chain)
                atom = self.topology.addAtom(
                    f"{chain_names[cid]}-L{resid+1}-{aid+1}",
                    f"{atom_types[kk]}",
                    residue,
                    f"{kk}",
                )
                if previous_atom is not None:
                    self.topology.addBond(previous_atom, atom)
                previous_atom = atom
                kk += 1

        # If marked as ring, connect last atom to first atom in each such chain
        for idx, chain in enumerate(self.topology.chains()):
            if isRing[idx] > 0:
                first_atom = list(chain.atoms())[0]
                last_atom = list(chain.atoms())[-1]
                self.topology.addBond(first_atom, last_atom)

        # Store atom types for future reference or export
        self.atom_types = atom_types

    def print_top(self) -> None:
        """
        Print the generated topology to stdout with formatted fields:
        Atom ID, Name, Type, Residue (Loci), and Chain ID.
        """
        atom_count: int = sum(1 for _ in self.topology.atoms())  # Total atom count
        print(f"{atom_count}\n")
        print(f"{'ID':<10} {'Name':<20} {'Type':<10} {'Loci':<10} {'Chain':<10}\n")

        for chain in self.topology.chains():
            for residue in chain.residues():
                for atom in residue.atoms():
                    print(
                        f"{atom.id:<10} {atom.name:<20} {atom.element:<10} "
                        f"{residue.name:<10} {chain.id:<10}\n"
                    )

    def save_top(self, filename: str = "topology.txt") -> None:
        """
        Save the generated topology to a file with formatted fields:
        Atom ID, Name, Type, Residue (Loci), and Chain ID.

        Parameters:
        - filename (str): Output file path. Defaults to 'topology.txt'.
        """
        atom_count: int = sum(1 for _ in self.topology.atoms())
        with open(filename, "w") as f:
            f.write(f"{atom_count}\n")
            f.write(
                f"{'ID':<10} {'Name':<20} {'Type':<10} {'Loci':<10} {'Chain':<10}\n"
            )
            for chain in self.topology.chains():
                for residue in chain.residues():
                    for atom in residue.atoms():
                        f.write(
                            f"{atom.id:<10} {atom.name:<20} {atom.element:<10} "
                            f"{residue.name:<10} {chain.id:<10}\n"
                        )
