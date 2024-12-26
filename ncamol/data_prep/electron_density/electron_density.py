import os
import pickle

from rdkit import Chem
from tqdm import tqdm

from ncamol import XTB_PATH
from ncamol.data_prep.electron_density.utils.esp import ESP
from ncamol.data_prep.electron_density.utils.orbkit import \
    electron_density_from_molden
from ncamol.data_prep.electron_density.utils.xtb import run_xtb


class Edens_Dataset:
    def __init__(
        self,
        file_path,
        storage_path,
        n_points: int,
        step_size: float,
        xtb_path: str = XTB_PATH,
    ) -> None:
        """
        Class for processing the ChemBl dataset.

        Parameters
        ----------
        file_path : str
            Path to the xyz files.
        storage_path : str
            Path where the processed data will be stored.
        n_points : int
            number of voxel for electron density.
        step_size : float
            step size for voxel (in Angstroem).
        """

        self.file_path = file_path
        self.filenames = [
            file for file in os.listdir(file_path) if file.endswith(".xyz")
        ]
        self.storage_path = storage_path
        self.xtb_path = xtb_path
        self.n_points = n_points
        self.step_size = step_size

        self.esp = ESP(n_points, step_size)

    def _save_pkl(self, data, outdir):
        with open(outdir, "wb") as f:
            pickle.dump(data, f)

    def _read_xyz_file(self, filepath: str):
        """
        Read xyz file and return the molecule object.

        Parameters
        ----------
        filepath : str
            path of the xyz file.

        Returns
        -------
        molecule : Chem.rdchem.Mol
            Molecule object.
        """
        molecule = Chem.MolFromXYZFile(filepath)
        if molecule is None:
            return None, None, None

        atom_types = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        coordinates = molecule.GetConformer().GetPositions()
        coords = [
            [atom, str(coord[0]), str(coord[1]), str(coord[2])]
            for atom, coord in zip(atom_types, coordinates)
        ]

        return coords, atom_types, Chem.MolToSmiles(molecule)

    def _parse_single_xyz_file(
        self, filename: str, esp: bool = True, save_pkl: bool = True
    ) -> None:
        """
        Parse a single xyz file.
        """
        xyz_file = os.path.join(self.file_path, filename)
        file_id = filename.split(".")[0]
        out_dir = os.path.join(self.storage_path, file_id)
        if os.path.exists(out_dir):
            # check if files in the directory
            if os.path.exists(os.path.join(out_dir, "output.pkl")):
                print(f"File {file_id} already processed")
                return None
        else:
            os.makedirs(out_dir)

        # read xyz file and run xtb
        coordinates, atom_types, smiles = self._read_xyz_file(xyz_file)
        if coordinates is None:
            return None

        # catch internal errors
        try:
            run_xtb(self.xtb_path, xyz_file, out_dir, molden=True, esp=esp)
        except Exception as e:
            print(e, "Error in xtb calculation")
            return None

        # compute electron density from xtb output
        molden_input = os.path.join(out_dir, "molden.input")
        rho = electron_density_from_molden(
            molden_input, self.n_points, self.step_size
        )
        output_dict = {}
        output_dict["electron_density"] = rho
        output_dict["num_atoms"] = len(atom_types)
        output_dict["smiles"] = smiles

        # compute esp from xtb output
        if esp and os.path.exists(os.path.join(out_dir, "xtb_esp.dat")):
            espxtb_input = os.path.join(out_dir, "xtb_esp.dat")
            molecule_esp = self.esp.calculate_espcube_from_xtb(espxtb_input)
            output_dict["electrostatic_potential"] = molecule_esp
        else:
            return None

        if save_pkl:
            # save data
            esp_file = os.path.join(out_dir, "esp.pkl")
            ed_file = os.path.join(out_dir, "ed.pkl")
            output_file = os.path.join(out_dir, "output.pkl")

            self._save_pkl(molecule_esp, esp_file)
            self._save_pkl(rho, ed_file)
            self._save_pkl(output_dict, output_file)

        return None

    def _compute_electron_density(self, esp: bool = True) -> None:
        """
        Compute electron density for all molecules in the dataset.
        """
        for filename in tqdm(self.filenames):
            self._parse_single_xyz_file(filename, esp=esp)
        return None
