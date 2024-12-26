import os
import subprocess as sub


def run_xtb(
    xtb_file: str,
    xyz_file: str,
    save_folder: str,
    molden: bool = False,
    esp: bool = False,
    opt: bool = False,
    scc: int = 500,
):
    """Run XTB job on given xtb input file, saving output to a given
    save folder.

    Args:
        xtb_file (str): path to xtb executable
        xyz_file (str): xtb input file to run geometry optimisation on.
        save_folder (str): Folder to save XTB output files to.
        molden (bool): If True will generate molden input file.
        esp (bool): If True will generate electrostatic potential.
        opt (bool): If true will perform geometry optmisation.

    """
    os.makedirs(save_folder, exist_ok=True)
    cmd = [os.path.abspath(xtb_file), os.path.abspath(xyz_file)]
    if molden:
        cmd.append("--molden")
    if esp:
        cmd.append("--esp")
    if opt:
        cmd.append("--opt")
    # cmd.append(f"--scc  {scc}")
    sub.Popen(
        cmd,
        cwd=save_folder,
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    ).communicate()

    # sometimes --esp fails to create the data, in which case we need to repeat the call
    if esp:
        datasize = os.path.getsize(save_folder + "/xtb_esp.dat")

        # if it is zero, it means it failed
        if datasize == 0:
            # repeat command with basic options
            cmd = [os.path.abspath(xtb_file), os.path.abspath(xyz_file)]
            sub.Popen(
                cmd,
                cwd=save_folder,
                stdout=sub.PIPE,
                stderr=sub.PIPE,
            ).communicate()
            # repeat command with --esp
            cmd.append("--esp")
            sub.Popen(
                cmd,
                cwd=save_folder,
                stdout=sub.PIPE,
                stderr=sub.PIPE,
            ).communicate()
