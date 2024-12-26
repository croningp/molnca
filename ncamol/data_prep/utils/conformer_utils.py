from rdkit import Chem
from rdkit.Chem import AllChem


def optimize_molecule(
    molecule: Chem.rdchem.Mol, numConfs: int = 20
) -> Chem.rdchem.Mol:
    molecule = Chem.AddHs(molecule)
    if len(AllChem.EmbedMultipleConfs(molecule, numConfs=numConfs)) == 0:
        return None
    conv_energy = AllChem.MMFFOptimizeMoleculeConfs(
        molecule, numThreads=10, maxIters=500
    )

    molecule = sorted(
        molecule.GetConformers(), key=lambda x: conv_energy[x.GetId()][-1]
    )[0]
    return molecule


def get_centering_vector(
    complex_path,
):
    complex = Chem.MolFromPDBFile(complex_path, removeHs=True)
    coords = complex.GetConformer().GetPositions().mean(axis=0)
    return coords


def generate_xyz_file_from_conformer(
    molecules: list[Chem.rdchem.Conformer],
    names: list[str],
    complex_path: str,
    save_path: str,
) -> None:
    """Turn conformers into xyz files
    Molecule will be centered at the center of the pdb mols in complex_path
    To make sure ligand and protein are centered at the same point in the complex
    if necessary

    Args:
        molecules (list[Chem.rdchem.Conformer]): conformers of molecules to be converted
        names (list[str]): names of the molecules
        complex_path (str): path to the complex pdb file for centering
        save_path (str): path to save the xyz files
    """

    for name, molecule in zip(names, molecules):
        center_vector = get_centering_vector(
            f"{complex_path}{name}.pdb",
        )
        owning_mol = molecule.GetOwningMol()

        with open(f"{save_path}{name}.xyz", "w") as f:
            f.write(f"{molecule.GetNumAtoms()}\n")
            f.write(f"{name}\n")
            for atom in owning_mol.GetAtoms():
                pos = molecule.GetAtomPosition(atom.GetIdx()) - center_vector
                f.write(f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n")
