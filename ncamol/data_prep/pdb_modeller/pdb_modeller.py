import os

from Bio.PDB import PDBList, PDBParser
from modeller import Alignment, Environ, Model, log
from modeller.automodel import LoopModel, refine
from tools.utils.parameters import THREE_LETTER_AMINOACIDS_TO_ONE_LETTER


def load_pdb_id(id: str, outdir: str) -> None:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pdbl = PDBList()
    fname = pdbl.retrieve_pdb_file(id, pdir=outdir, file_format="pdb")
    os.rename(fname, outdir + f"{id}.pdb")
    return None


def generate_alignment_file(
    pdb_id: str, pdbfile: str, sequencefile: str, alignmentfile: str
) -> bool:
    """Generate an alignment file from a PDB file.
    PDB files have META data in which line starting with "REMARK 465" contain missing residues
    in the structure.
    The structure of lines with REMARD 465 with missing residues is:
    REMARK 465   M RES C SSSEQI                                 | [start of missing residues]
    REMARK 465     [THREE LETTER AA] [CHAIN ID]    [RESID]      | [missing residue]
    """

    missing_residues = get_missing_residues(pdbfile)
    if not missing_residues:
        print("No missing residues detected")
        return False

    missing_residues = [
        [three_letter_to_one_aa(residue[0]), residue[-1]]
        for residue in missing_residues
    ]
    residues, first_position, structureX = get_residues(sequencefile)
    # if residues are missing before start of structure, those dictate the first position
    first_position = min(first_position, int(missing_residues[0][-1]))

    write_alignment(
        pdb_id, alignmentfile, residues, missing_residues, first_position, structureX
    )
    return True


def write_alignment(
    pdb_id: str,
    alignmentfile: str,
    residues: list[str],
    missing_residues: list[list[str]],
    first_position: int,
    structureX: str,
) -> None:
    """Write the alignment file."""
    line_length = 75

    with open(alignmentfile, "w") as f:
        alignment_string_missing = ""
        alignment_string_complete = ""

        combined_residues = order_residues(
            residues.copy(), missing_residues.copy(), first_position
        )
        # print(combined_residues)
        for residue in combined_residues:
            if residue in missing_residues:
                alignment_string_missing += "-"
                alignment_string_complete += residue[0]
            else:
                alignment_string_missing += residue[0]
                alignment_string_complete += residue[0]

        alignment_string_missing_chunks = "\n".join(
            [
                alignment_string_missing[i : i + line_length]
                for i in range(0, len(alignment_string_missing), line_length)
            ]
        )

        alignment_string_complete_chunks = "\n".join(
            [
                alignment_string_complete[i : i + line_length]
                for i in range(0, len(alignment_string_complete), line_length)
            ]
        )

        # write missing residues string
        f.write(f">P1;{alignmentfile.split('/')[-1].split('.')[0]}\n")
        f.write(structureX)
        f.write(alignment_string_missing_chunks)

        # write complete residues string
        f.write(f"\n>P1;{alignmentfile.split('/')[-1].split('.')[0]}_fill\n")
        f.write(f"sequence:{pdb_id}::::::::\n")
        f.write(alignment_string_complete_chunks)
    return None


def order_residues(
    residues: list[str], missing_residues: list[list[str]], first_position: int
) -> list[str]:
    """
    Generates a list of residues in the order they appear in the structure.
    Also generates a list of residues where for each missing residue, the
    corresponding residue in the structure is replaced with a '-'.

    residues: list of residues in the alignment file; format [AA, position]
    missing_residues: list of missing residues in the structure: format [AA, position]
    first_position: position of the first residue in the structure
    """
    print(len(residues))
    print(missing_residues)
    ordered_residues = []
    num_residues = len(residues) + len(missing_residues)
    for i in range(first_position, num_residues + first_position):
        if missing_residues:
            if i == int(missing_residues[0][-1]):
                ordered_residues.append(missing_residues[0])
                missing_residues.pop(0)
            else:
                ordered_residues.append([residues[0], i])
                residues.pop(0)
        else:
            ordered_residues.append([residues[0], i])
            residues.pop(0)
    return ordered_residues


def get_missing_residues(pdbfile: str) -> list[list[str]]:
    """Get the missing residues from the PDB file.
    Are stored in the REMARK 465 META data.
    """

    missing_residues = []
    with open(pdbfile, "r") as f:
        start_of_missing = False
        for line in f.readlines():
            if line.startswith("REMARK 465   M RES C SSSEQI"):
                start_of_missing = True
                continue
            elif start_of_missing and line.startswith("REMARK 465"):
                line = line.strip().split()
                missing_residues.append([line[2], int(line[-1])])
    return missing_residues


def get_residues(aligmentfile: str) -> tuple[list[str], int]:
    """
    Get the residues from the alignment file and the id of the first AA present in
    the alignment file.
    """
    first_position = None
    structureX_line: str

    residues = []
    with open(aligmentfile, "r") as f:
        start_of_residues = False
        for line in f.readlines():
            if line.startswith("structureX"):
                structureX_line = line
                first_position = int(line.split(":")[2])
                start_of_residues = True
            elif start_of_residues and not line.startswith(">"):
                line = list(line.strip())
                residues.extend(line)
    return residues, first_position, structureX_line


def three_letter_to_one_aa(residue: str) -> str:
    return THREE_LETTER_AMINOACIDS_TO_ONE_LETTER[residue]


def model_structure(code, outdir):
    cwd = os.getcwd()
    os.chdir(outdir)

    try:
        log.verbose()
        env = Environ()
        env.io.atom_files_directory = ["."]

        a = LoopModel(env, alnfile=f"{code}.ali", knowns=code, sequence=f"{code}_fill")
        a.starting_model = 1
        a.ending_model = 1

        a.loop.starting_model = 1
        a.loop.ending_model = 2
        a.loop.md_level = refine.fast

        a.make()
    except Exception as e:
        print(f"Failed to model structure for {code}")
        print(e)
    finally:
        # go back to the original directory
        os.chdir(cwd)
    return None


def check_number_of_chains(code: str, outdir: str = "../data/pdb/") -> None:
    """Check if the pdb file has more than one chain."""
    pdbfile = f"{outdir}/{code}.pdb"
    parser = PDBParser()
    structure = parser.get_structure(code, pdbfile)[0]
    if len(structure) > 1:
        assert False, """
            PDB file has more than one chain.
            Currently, only PDB files with single chains are supported.    
        """
    return None


def main(code: str, outdir: str = "../data/pdb/"):
    outdir = f"{outdir}{code}/"
    load_pdb_id(code, outdir=outdir)
    check_number_of_chains(
        code, outdir=outdir
    )  # assertion error if more than one chain

    e = Environ()
    m = Model(e, file=f"{outdir}{code}")
    aln = Alignment(e)
    aln.append_model(m, align_codes=code)
    aln.write(file=f"{outdir}{code}" + ".seq")

    model_missing_residues = generate_alignment_file(
        code, f"{outdir}/{code}.pdb", f"{outdir}/{code}.seq", f"{outdir}/{code}.ali"
    )
    if model_missing_residues:
        model_structure(code, outdir)
