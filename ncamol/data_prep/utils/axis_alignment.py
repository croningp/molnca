import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def rotate_molecule(coords):
    """Rotate a molecule to align with the principal axes.

    Parameters
    ----------
    coords: np.ndarray
        The coordinates of the molecule.

    Returns
    -------
    coords: np.ndarray
        The coordinates of the molecule after rotation.

    """
    pca = PCA(n_components=3)
    pca.fit(coords)
    coords = pca.transform(coords)
    rot = R.align_vectors(
        np.array([[1, 0, 0]], dtype=np.int32), [pca.components_[0]]
    )  # align with x-axis
    coords = rot[0].apply(coords)
    return coords
