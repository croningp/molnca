import numpy as np


class ESP:
    """
    Class for calculating electrostatic potential grid.

    Arguments
    ---------
        n_points: (int, default = 64)
            number of points on the grid along each axis

        step_size: (float, default = 0.5)
            distance (in Bohr) between grid points on a given axis

    Methods
    --------
        calculate_esp_grid: Calculate ESP grid using specified
            grid params and molden input file from XTB.
    """

    def __init__(self, n_points=64, step_size=0.5):
        self.n_points = n_points
        self.step_size = step_size

    def calculate_espcube_from_xtb(self, esp_xtb):
        """Given the electrstatic potential array built using xtb (option --esp), this
        function will place the sparse array into a cube.

        Args:
            esp_xtb: File as generated using "xtb --esp"

        Returns:
            cube with positions filled using data from the xtb array
        """

        # read xtb file, which contains sparse xyz and their charge
        data = np.genfromtxt(esp_xtb)
        # create canvas cube with all 0s
        cube = np.zeros((self.n_points, self.n_points, self.n_points))
        # use step size to see how big is a voxel
        factor = 1 / self.step_size
        center = self.n_points // 2

        for entry in data:
            x, y, z, e = entry
            # adjust coordinates to cube
            x = int(x * factor + center)
            y = int(y * factor + center)
            z = int(z * factor + center)
            cube[x, y, z] += e

        return cube
