from collections import Counter
import os
from pathlib import Path

from Bio.PDB import (PDBIO, Chain, Model, NeighborSearch, PDBList, PDBParser,
                     Select, Structure)
import numpy as np

from ncamol.data_prep.utils.axis_alignment import rotate_molecule
from ncamol.data_prep.utils.parameters import THREE_LETTER_AMINOACIDS


class ConformationOutputSelector(Select):  # Inherit methods from Select class
    keepAltID: str = "A"  # Alt location ID to be output.

    def accept_atom(self, atom):
        if (not atom.element == "H") and ( # !!! note might change this to not exlude hydrogens
            (not atom.is_disordered()) or atom.get_altloc() == self.keepAltID
        ):
            atom.set_altloc(" ")  # Eliminate alt location ID before output.
            return True
        else:  # Alt location was not one to be output.
            return False


class PDB:
    MIN_ATOMS = 8  # minimum number of atoms in a residue to be considered a ligand

    def __init__(self, pdb_id: str, pdb_path: str) -> None:
        self.pdb_list = PDBList()
        self.pdb_id = pdb_id
        self.pdb_path = pdb_path

        self.ligand_names = []
        self.ligand_counts = Counter()

        if not os.path.exists(self.pdb_path):
            os.makedirs(self.pdb_path)

        # download and store file
        fname = self.pdb_list.retrieve_pdb_file(
            self.pdb_id, pdir=self.pdb_path, file_format="pdb"
        )

        os.rename(fname, self.pdb_path + f"{self.pdb_id}.pdb")
        self.pdb_file = self.pdb_path + f"{self.pdb_id}.pdb"
        self.fname = f"{self.pdb_id}.pdb"

        # preprocess file
        self.clean_pdb()

    def clean_pdb(self):
        """Remove multioccupancy atoms from a PDB file.
        And remove all residues with less than or equal to self.MIN_ATOMS atoms.
        And water molecules.
        And retrieve Ligand Names
        """
        structure = self.get_pdb_file()

        for model in structure:
            for chain in model:
                for residue in list(chain):
                    if self.remove_water(residue):
                        chain.detach_child(residue.id)
                        continue
                    if residue.get_resname() not in THREE_LETTER_AMINOACIDS:
                        to_small = self.check_min_residue_size(residue)
                        if to_small:
                            chain.detach_child(residue.id)
                            continue
                        else:
                            self.ligand_names.append(residue.get_resname())
                            continue

        # Update ligand counts
        self.ligand_counts = Counter(self.ligand_names)

        # Save the cleaned structure to a new PDB file
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.pdb_file, select=ConformationOutputSelector())

    def get_pdb_file(self) -> Structure.Structure:
        """Return the path to the PDB file."""
        parser = PDBParser()
        structure = parser.get_structure(self.pdb_id, self.pdb_file)
        return structure

    def remove_water(self, residue) -> bool:
        """Remove all water molecules from a model.
        And remove all residues with less than or equal to self.MIN_ATOMS atoms.
        """
        return residue.get_resname() == "HOH"

    def check_min_residue_size(self, residue) -> bool:
        """Check if residue has less than or equal to self.MIN_ATOMS atoms."""
        return len(residue) <= self.MIN_ATOMS

    def retrieve_binding_pocket(
        self,
        storage_dir_pocket: str,
        storage_dir_lig_and_pocket: str,
        storage_dir_lig: str,
        model_id: int = 0,
        cutoff: float = 8,
    ) -> None:
        """
        Retrieve the binding pocket of a ligand in a PDB file.
        For each ligand stored in self.ligand_names, find the nearby residues within a cutoff distance.
        Save the binding pocket and the ligand in a new PDB file.

        :param model_id: The model ID of the PDB file.
        :param cutoff: The cutoff distance to find nearby residues.
        :param storage_dir_pocket: The directory to store the binding pocket.
        :param storage_dir_lig_and_pocket: The directory to store the ligand and binding pocket.
        :return: The number of binding pockets found.
        """
        counter: int = 0
        if not os.path.exists(storage_dir_pocket):
            os.makedirs(storage_dir_pocket)

        if not os.path.exists(storage_dir_lig_and_pocket):
            os.makedirs(storage_dir_lig_and_pocket)

        if not os.path.exists(storage_dir_lig):
            os.makedirs(storage_dir_lig)

        # a pdb files can have multiple models which are different conformations of the same protein
        # just using 0 as default for now

        # for every ligand in the pdb file, find the nearby residues
        for ligand_name, count in self.ligand_counts.items():
            for ligand_id in range(count):

                # LIGAND AND POCKET
                structure = self.get_pdb_file()
                model = structure[model_id]

                nearby_residues, ligand = self.find_near_neighbours(
                    model=model,
                    ligand_name=ligand_name,
                    ligand_id=ligand_id,
                    cutoff=cutoff,
                )

                # save ligand and pocket
                geometric_center = ligand.center_of_mass(geometric=True)
                nearby_residues = self.center_around_zero(
                    geometric_center=geometric_center,
                    nearby_residues=nearby_residues,
                )

                # files should be stored with name 00_pdb_id.pdb where 00_ is a counter for the number of files
                # check which files already exist and increment the counter
                counter = 0
                while os.path.exists(
                    f"{storage_dir_pocket}{self.pdb_id}_{str(counter).zfill(2)}.pdb"
                ):
                    counter += 1

                file_name = f"{self.pdb_id}_{str(counter).zfill(2)}.pdb"
                new_file_name_pocket_ligand = f"{storage_dir_lig_and_pocket}{file_name}"

                io = PDBIO()
                io.set_structure(self.generate_pocket_pdb(nearby_residues, align=True))
                io.save(new_file_name_pocket_ligand)

                # POCKET 
                # save pocket without ligand. Second call to find_near_neighbours is necessary because
                # generator returned by the first call is exhausted
                structure = self.get_pdb_file()
                model = structure[model_id]
                nearby_residues, ligand = self.find_near_neighbours(
                    model=model,
                    ligand_name=ligand_name,
                    ligand_id=ligand_id,
                    cutoff=cutoff,
                )
                nearby_residues = self.center_around_zero(
                    geometric_center=geometric_center,
                    nearby_residues=nearby_residues,
                )

                nearby_residues.remove(ligand)
                new_file_name_pocket = f"{storage_dir_pocket}{file_name}"

                io = PDBIO()
                io.set_structure(self.generate_pocket_pdb(nearby_residues, align=True))
                io.save(new_file_name_pocket)

                # LIGAND
                # save ligand without pocket
                structure = self.get_pdb_file()
                model = structure[model_id]
                nearby_residues, ligand = self.find_near_neighbours(
                    model=model,
                    ligand_name=ligand_name,
                    ligand_id=ligand_id,
                    cutoff=0, # <- exclude all surrounding residues
                )

                nearby_residues = self.center_around_zero(
                    geometric_center=geometric_center,
                    nearby_residues=nearby_residues,
                )
                new_file_name_ligand = f"{storage_dir_lig}{self.pdb_id}_{str(counter).zfill(2)}.pdb"

                io = PDBIO()
                io.set_structure(self.generate_pocket_pdb(nearby_residues, align=True))
                io.save(new_file_name_ligand)

        return None

    def retrieve_ligand(self, storage_dir_lig: str, model_id: int = 0) -> None:
        """
        Retrieve the ligand in a PDB file.
        Save the ligand in a new PDB file.

        :param model_id: The model ID of the PDB file.
        :param storage_dir_lig: The directory to store the ligand.
        :return: The number of ligands found.
        """
        counter: int = 0
        if not os.path.exists(storage_dir_lig):
            os.makedirs(storage_dir_lig)

        # for every ligand in the pdb file, find the nearby residues
        for ligand_name, count in self.ligand_counts.items():
            for ligand_id in range(count):

                # LIGAND
                structure = self.get_pdb_file()
                model = structure[model_id]

                nearby_residues, ligand = self.find_near_neighbours(
                    model=model,
                    ligand_name=ligand_name,
                    ligand_id=ligand_id,
                    cutoff=0, # <- exclude all surrounding residues
                )

                # save ligand without pocket
                geometric_center = ligand.center_of_mass(geometric=True)
                nearby_residues = self.center_around_zero(
                    geometric_center=geometric_center,
                    nearby_residues=nearby_residues,
                )

                # files should be stored with name 00_pdb_id.pdb where 00_ is a counter for the number of files
                # check which files already exist and increment the counter
                counter = 0
                while os.path.exists(
                    f"{storage_dir_lig}{self.pdb_id}_{str(counter).zfill(2)}.pdb"
                ):
                    counter += 1

                file_name = f"{self.pdb_id}_{str(counter).zfill(2)}.pdb"
                new_file_name_ligand = f"{storage_dir_lig}{file_name}"

                io = PDBIO()
                io.set_structure(self.generate_pocket_pdb(nearby_residues, align=True))
                io.save(new_file_name_ligand)

        return None

    def find_near_neighbours(
        self,
        model: Model.Model,
        ligand_name: str,
        ligand_id: int = 0,
        cutoff: float = 5,
    ) -> tuple:
        """Return a set of all residues within cutoff angstroms of the ligand."""
        ligand = [
            residue
            for residue in model.get_residues()
            if residue.get_resname() == ligand_name
        ][ligand_id]

        ns = NeighborSearch(list(model.get_atoms()))
        # geometric center
        ligand_com = ligand.center_of_mass(geometric=True)

        # get all residues within cutoff angstroms of the ligand
        # but only save the residues which are Amino Acids or the ligand itself
        nearby_residues = set()
        if cutoff == 0:
            nearby_residues.add(ligand)
            return nearby_residues, ligand
        
        nearby_atoms = ns.search(ligand_com, cutoff, level="A")
        nearby_residues.update(
            [
                atom.get_parent()
                for atom in list(nearby_atoms)
                if atom.get_parent().get_resname()
                in THREE_LETTER_AMINOACIDS + [ligand_name]
            ]
        )
        return nearby_residues, ligand

    def generate_pocket_pdb(
        self, nearby_residues: set, align: bool = False
    ) -> Structure.Structure:
        """Generate a new PDB file containing the ligand and nearby residues."""
        nearby_structures = Structure.Structure("nearby")
        nearby_model = Model.Model(0)
        nearby_chain = Chain.Chain("A")

        # if clashes, change residue id
        residue_id = 999
        for residue in nearby_residues:
            try:
                nearby_chain.add(residue)
            except Exception as e:
                residue.id = (" ", residue_id, " ")
                nearby_chain.add(residue)
                residue_id -= 1

        nearby_model.add(nearby_chain)
        nearby_structures.add(nearby_model)

        if align:
            coords = rotate_molecule(self.get_coordinates(nearby_structures))
            nearby_structures = self.update_coordinates(nearby_structures, coords)

        return nearby_structures

    def center_around_zero(self, geometric_center, nearby_residues):
        """Center nearby residues around the geometric center of the ligand."""

        for residue in nearby_residues:
            for atom in residue:
                atom.set_coord(atom.get_coord() - geometric_center)

        return nearby_residues

    def get_coordinates(self, structure: Structure.Structure) -> np.array:
        """Return the coordinates of a PDB file."""
        coordinates = []
        model = structure[0]

        for i, atom in enumerate(model.get_atoms()):
            coordinates.append(atom.get_coord())

        return np.array(coordinates)

    def update_coordinates(self, structure: Structure.Structure, coords: np.array):
        """Update the coordinates of a PDB file."""
        model = structure[0]

        for i, atom in enumerate(model.get_atoms()):
            atom.set_coord(coords[i])

        return structure
