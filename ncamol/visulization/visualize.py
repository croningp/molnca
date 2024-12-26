import k3d
import numpy as np


def plot_ESP(voxel_data: list[np.array], min_val: int | None =None, max_val: int | None =None, **kwargs):
    """Atom channel voxel data visualization using k3d.

    Args:
        data (np.array): shape atom_channel x x_dim x y_dim x z_dim
    Returns:
        None
    """
    dims = voxel_data[0].shape[-1]

    plot = k3d.plot()

    if not kwargs.get("color_map"):
        color_map = {
            0: 0x383838, 
            1: 0x4c4cee, 
            2: 0xfb2b2b, 
            3: 0xffff52, 
            4: 0xacf598, 
            5: 0x82f5ff,
            6: 0x4c4cee,
        }
    else:
        color_map = kwargs.get("color_map")

    for channel_index in range(voxel_data.shape[0]):
        data = voxel_data[channel_index]
        voxels = (data > 0).astype(np.uint8)

        # color = color_map.get(channel_index, "")
        color = color_map.get(channel_index, 0xffffff)
        plt_voxels_heart = k3d.voxels(voxels,
                                    color_map=color,
                                    outlines=False,
                                    bounds=[-0, dims, -0, dims, -0, dims])

        plot += plt_voxels_heart
    plot.display()
    return None