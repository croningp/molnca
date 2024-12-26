import math
import warnings

from Bio.PDB import Structure
import numpy as np

from ncamol.data_prep.utils.parameters import VAN_DER_WAALS_RADIUS


class Voxel:
    UNKOWN_RADII = 1.5
    CHANNELS = {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "P": 4,
        "Cl": 5,
        "other": 6,
    }


class Voxel_PDB(Voxel):
    """Voxel representation of a protein or Ligand.

    Voxelizes a protein.

    Parameters
    ----------
    protein: dict
        A protein object.
    voxelsize: float
        The size of a voxel (in Angstrom).
    grid_size: int
        The size of the grid (in Angstrom).
    grid_dim: int
        grid_size / voxelsize (i.e. the number of voxels in each dimension).
    aggregation: str, defaul 'surround'
        How to aggregate labels of a voxel.
        'surround' -> the voxel surrounding the atom is labeled (within atom vdw radius)
        'closest' -> the closest voxel to atom center is labeled
    Attributes
    ----------
    types: np.ndarray [0, 1, 2, 3, 4, 5, 6] -> [C, N, O, S, P, Cl, other]

    the radii of other are set to 1.5 Angstrom

    Returns
    -------
    np.ndarray
        The voxelized protein. Shape: ([num_channels | 7],grid_dim, grid_dim, grid_dim)

    """

    def __init__(
        self,
        structure: Structure.Structure,
        grid_size: int = 24,
        voxel_size: float = 1.0,
        aggregation: str = "surround",
    ) -> None:
        self.structure = structure
        self.grid_size = grid_size # size of the grid in Angstrom
        self.voxel_size = voxel_size # size of a voxel in Angstrom
        self.aggregation = aggregation
        self.grid_dim = np.ceil(grid_size / voxel_size).astype(np.int32) # number of voxels in each dimension

        self.min_dist = math.ceil(max(VAN_DER_WAALS_RADIUS.values()))  # minimum distance of atom to grid edge

        if aggregation == "surround":
            self.init_voxel_cord_grid()

    def init_voxel_cord_grid(self):
        """
        Initializes a grid of voxel coordinates.
        """
        self.voxels_center_coords = np.zeros(
            (self.grid_dim, self.grid_dim, self.grid_dim, 3)
        )

        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                for k in range(self.grid_dim):
                    self.voxels_center_coords[i, j, k] = np.array(
                        [i, j, k]
                    ) * self.voxel_size + (self.voxel_size * 0.5)
        return None

    def center_coords(self, grid_center) -> np.ndarray:
        """Center the coordinates of the atoms in the grid.

        Parameters
        ----------
        grid_center: np.ndarray
            The center of the grid.
            
        Returns
        -------
        np.ndarray
            The centered coordinates.
        """
        self.center_vector = grid_center - self.atom_coords.mean(axis=0)
        return self.atom_coords + self.center_vector

    def fill_voxels(
        self, grid: np.ndarray, ligand_grid: np.ndarray, lig_atom_ids: list[int]
    ) -> tuple[np.array]:
        # rasterize
        if self.aggregation == "closest":
            voxel_indices = (self.atom_coords / self.voxel_size).astype(np.int32)

        # fill grid
        for i, atom in enumerate(self.atom_types):
            if self.aggregation == "closest":
                if i in lig_atom_ids:
                    ligand_grid[self.CHANNELS[atom]][tuple(voxel_indices[i])] = 1
                else:
                    grid[self.CHANNELS[atom]][tuple(voxel_indices[i])] = 1

            elif self.aggregation == "surround":



                voxel_coords = np.argwhere(
                    np.linalg.norm(
                        self.atom_coords[i] - self.voxels_center_coords, axis=3
                    )
                    <= self.atom_radii[i]
                )

                

                for coord in voxel_coords:
                    if i in lig_atom_ids:
                        ligand_grid[self.CHANNELS[atom]][*coord] = 1
                    else:
                        grid[self.CHANNELS[atom]][*coord] = 1
            else:
                raise ValueError(
                    f"Aggregation method {self.aggregation} not supported.",
                    "Please use 'closest' or 'surround'.",
                )

        return grid, ligand_grid

    def _voxelize(self, lig_name: str = "") -> np.ndarray:
        """Voxelizes a protein.

        Returns
        -------
        np.ndarray
            A 3D array of shape (gridsize, gridsize, gridsize) containing the voxelized protein.

        """
        lig_atom_ids = list()
        ligand_grid = np.array

        self.atom_types = [
            atom.get_name()[0]
            if atom.get_name()[0] in VAN_DER_WAALS_RADIUS
            else "other"
            for atom in self.structure.get_atoms()
        ]

        self.atom_radii = [
            VAN_DER_WAALS_RADIUS[atom]
            if atom in VAN_DER_WAALS_RADIUS
            else self.UNKOWN_RADII
            for atom in self.atom_types
        ]
        self.atom_coords = np.array(
            [atom.get_coord() for atom in self.structure.get_atoms()],
            dtype=np.float32,
        )

        # create empty grid with of size n_atom_types, grid_size, grid_size, grid_size
        grid = np.zeros(
            shape=(len(self.CHANNELS), self.grid_dim, self.grid_dim, self.grid_dim)
        )

        # second grid for ligand
        ligand_grid = np.array
        if lig_name:
            lig_atom_ids = self.get_ligand_ids(lig_name)
            ligand_grid = np.zeros(
                shape=(
                    len(self.CHANNELS),
                    self.grid_dim,
                    self.grid_dim,
                    self.grid_dim,
                )
            )

        # get center of grid
        # center in grid
        grid_center = np.array(
            [self.grid_size / 2, self.grid_size / 2, self.grid_size / 2]
        )
        self.atom_coords = self.center_coords(grid_center)
        # check if any atom is outside of the grid

        if np.any(self.atom_coords < self.min_dist) or np.any(self.atom_coords > self.grid_size - self.min_dist):
            warnings.warn(
                f"Protein does not fit in grid! Min coords {self.atom_coords.min(axis=0)}, Max coords {self.atom_coords.max(axis=0)}"
            )
            return None, None

        return self.fill_voxels(grid, ligand_grid, lig_atom_ids)

    def get_ligand_ids(self, lig_name: str) -> list[int]:
        """Returns the ids of the atoms in the ligand.

        Parameters
        ----------
        lig_name: str
            The name of the ligand.

        Returns
        -------
        list
            A list of atom ids.
        """
        lig_ids = []
        atom_id = 0

        for residue in self.structure.get_residues():
            for atom in residue.get_atoms():
                if atom.get_parent().get_resname() == lig_name:
                    lig_ids.append(atom_id)
                atom_id += 1

        return lig_ids


class Voxel_Ligand(Voxel):
    """Voxel representation of a Ligand.
    Function for dataset preparation for a set of ligands that are not associated with a protein.

    Voxelizes a ligand.

    Parameters
    ----------
    voxelsize: float
        The size of a voxel (in Angstrom).
    grid_size: int
        The size of the grid (in Angstrom).
    grid_dim: int
        grid_size / voxelsize (i.e. the number of voxels in each dimension).
    aggregation: str, defaul 'surround'
        How to aggregate labels of a voxel.
        'surround' -> the voxel surrounding the atom is labeled (within atom vdw radius)
        'closest' -> the closest voxel to atom center is labeled
    Attributes
    ----------
    types: np.ndarray [0, 1, 2, 3, 4, 5, 6] -> [C, N, O, S, P, Cl, other]

    the radii of other are set to 1.5 Angstrom
    Returns
    -------
    np.ndarray
        The voxelized protein. Shape: ([num_channels | 7],grid_dim, grid_dim, grid_dim)
    """

    def __init__(
        self,
        structure: Structure.Structure,
        grid_size: int = 24,
        voxel_size: float = 1.0,
        aggregation: str = "surround",
        resolution: str = "atom",
        center_vector: np.ndarray = None,
    ) -> None:
        self.structure = structure
        self.grid_size = grid_size # size of the grid in Angstrom
        self.voxel_size = voxel_size # size of a voxel in Angstrom
        self.aggregation = aggregation
        self.grid_dim = np.ceil(grid_size / voxel_size).astype(np.int32) # number of voxels in each dimension
        self.center_vector = center_vector

        self.min_dist = math.ceil(max(VAN_DER_WAALS_RADIUS.values()))  # minimum distance of atom to grid edge

        if aggregation == "surround":
            self.init_voxel_cord_grid()

    def init_voxel_cord_grid(self):
        """
        Initializes a grid of voxel coordinates.
        """
        self.voxels_center_coords = np.zeros(
            (self.grid_dim, self.grid_dim, self.grid_dim, 3)
        )

        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                for k in range(self.grid_dim):
                    self.voxels_center_coords[i, j, k] = np.array(
                        [i, j, k]
                    ) * self.voxel_size + (self.voxel_size * 0.5)
        return None

    def center_coords(self, grid_center) -> np.ndarray:
        """Center the coordinates of the atoms in the grid.

        Parameters
        ----------
        grid_center: np.ndarray
            The center of the grid.

        Returns
        -------
        np.ndarray
            The centered coordinates.
        """
        if self.center_vector is None:
            self.center_vector = grid_center - self.atom_coords.mean(axis=0)
            return self.atom_coords + self.center_vector
        else:
            return self.atom_coords + self.center_vector

    def fill_voxels(self, grid) -> np.ndarray:
        # rasterize
        if self.aggregation == "closest":
            voxel_indices = (self.atom_coords / self.voxel_size).astype(np.int32)

        # fill grid
        for i, atom in enumerate(self.atom_types):
            if self.aggregation == "closest":
                grid[self.CHANNELS[atom]][tuple(voxel_indices[i])] = 1
            elif self.aggregation == "surround":
                voxel_coords = np.argwhere(
                    np.linalg.norm(
                        self.atom_coords[i] - self.voxels_center_coords, axis=3
                    )
                    <= self.atom_radii[i]
                )
                for coord in voxel_coords:
                    grid[self.CHANNELS[atom]][*coord] = 1
            else:
                raise ValueError(
                    f"Aggregation method {self.aggregation} not supported.",
                    "Please use 'closest' or 'surround'.",
                )
        return grid

    def _voxelize(
        self,
    ) -> np.ndarray:
        """Voxelizes a ligand.


        Returns
        -------
        np.ndarray
            A 3D array of shape (gridsize, gridsize, gridsize) containing the voxelized protein.

        """

        self.atom_types = [
            atom.get_name()[0]
            if atom.get_name()[0] in VAN_DER_WAALS_RADIUS
            else "other"
            for atom in self.structure.get_atoms()
        ]
        
        self.atom_radii = [
            VAN_DER_WAALS_RADIUS[atom]
            if atom in VAN_DER_WAALS_RADIUS
            else self.UNKOWN_RADII
            for atom in self.atom_types
        ]
        self.atom_coords = np.array(
            [atom.get_coord() for atom in self.structure.get_atoms()],
            dtype=np.float32,
        )

        # check max distance between two atoms in atom_coords
        max_dist = 0
        for i in range(len(self.atom_coords)):
            for j in range(i + 1, len(self.atom_coords)):
                dist = np.linalg.norm(self.atom_coords[i] - self.atom_coords[j])
                if dist > max_dist:
                    max_dist = dist

        # create empty grid with of size n_atom_types, grid_size, grid_size, grid_size
        grid = np.zeros(
            shape=(len(self.CHANNELS), self.grid_dim, self.grid_dim, self.grid_dim)
        )
        # get center of grid
        grid_center = np.array(
            [self.grid_size / 2, self.grid_size / 2, self.grid_size / 2]
        )

        # center in grid
        self.atom_coords = self.center_coords(grid_center)
        # print center
        if np.any(self.atom_coords < self.min_dist) or np.any(self.atom_coords > self.grid_size - self.min_dist):
            warnings.warn(
                f"Ligand does not fit in grid! Min coords {self.atom_coords.min(axis=0)}, Max coords {self.atom_coords.max(axis=0)}"
            )
            return None, None


        return self.fill_voxels(grid)
