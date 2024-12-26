import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def add_coordinates(mol, coords):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol


def voxel_grid_to_rdkit(
    voxel_grid: np.array,
    channels: dict = {0: "C", 1: "N", 2: "O", 3: "S", 4: "P", 5: "Cl", 6: "*"},
    voxel_size: float = 1,
) -> None:
    """
    Visualizes a voxel grid using py3dmol.
    """
    # create py3dmol view
    # view = py3Dmol.view(width=800, height=800)

    atoms = list()
    coordinates = list()
    # for each voxel
    for i, channel in enumerate(voxel_grid):
        for x in range(channel.shape[0]):
            for y in range(channel.shape[1]):
                for z in range(channel.shape[2]):
                    # if voxel is occupied
                    if channel[x, y, z] != 0:
                        # get voxel coordinates
                        x1 = x * voxel_size
                        y1 = y * voxel_size
                        z1 = z * voxel_size

                        # get voxel type
                        voxel_type = channels[i]

                        # add atom to list
                        atoms.append(voxel_type)
                        coordinates.append([x1, y1, z1])

    atom_smiles = ".".join(atoms)
    mol = Chem.MolFromSmiles(atom_smiles)
    AllChem.EmbedMolecule(mol)
    mol = add_coordinates(mol, coordinates)

    return mol


def show_protein_ligand(protein, ligand, view=None):
    # Create Py3Dmol view
    if view is None:
        view = py3Dmol.view(width=800, height=800)
    else:
        view.removeAllModels()

    # Add the first molecule with a specific color scheme
    view.addModel(Chem.MolToMolBlock(protein), "sdf")
    view.setStyle({"model": 0}, {"sphere": {"scale": "0.5"}})

    # Add the second molecule with a different color scheme
    view.addModel(Chem.MolToMolBlock(ligand), "sdf")
    view.setStyle(
        {"model": 0}, {"sphere": {"colorscheme": "cyanCarbon", "scale": "0.75"}}
    )

    # Set the view options
    view.zoomTo(sel=1)
    # rotate the view
    view.rotate(300, "y")
    view.rotate(-10, "x")
    view.rotate(-70, "z")
    view.show()
    # return view
