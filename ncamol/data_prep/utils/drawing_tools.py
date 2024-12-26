from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def draw_box(ax, dim: int = 32) -> None:
    """Draw a box around the origin."""
    # Plot the box
    ax.plot([0, dim], [0, 0], [0, 0], color="k")
    ax.plot([0, 0], [0, dim], [0, 0], color="k")
    ax.plot([0, 0], [0, 0], [0, dim], color="k")
    ax.plot([0, dim], [0, 0], [dim, dim], color="k")
    ax.plot([0, 0], [0, dim], [dim, dim], color="k")
    ax.plot([0, dim], [dim, dim], [0, 0], color="k")
    ax.plot([0, 0], [dim, dim], [0, dim], color="k")
    ax.plot([dim, dim], [0, dim], [0, 0], color="k")
    ax.plot([dim, dim], [0, 0], [0, dim], color="k")
    ax.plot([dim, dim], [0, dim], [dim, dim], color="k")
    ax.plot([dim, dim], [dim, dim], [0, dim], color="k")
    ax.plot([dim, 0], [dim, dim], [dim, dim], color="k")
    return None


def plot_voxels(
    protein=None,
    ligand=None,
    show_channels="all",
    dim: int = 35,
    channel_mapping={
        0: "C",
        1: "N",
        2: "O",
        3: "S",
        4: "P",
        5: "F",
        6: "other",
    },
    custom_color_map: dict | None = None,
    point_cloud: bool = False,
    title="Target",
    show_animation=False,
    show_box=False,
    save_svg: None | str = None,
    zoom: float = 1.0,
    alpha: float = 1.0,
):
    color_mapping = {
        "protein": {
            "C": "grey",
            "N": "blue",
            "O": "red",
            "S": "yellow",
            "P": "orange",
            "F": "green",
            "other": "magenta",
        },
        "ligand": {
            "C": "grey",
            "N": "blue",
            "O": "red",
            "S": "yellow",
            "P": "orange",
            "F": "green",
            "other": "magenta",
        },
    }

    if custom_color_map is not None:
        for key, value in custom_color_map.items():
            color_mapping[key] = value

    plt.clf()
    ax = plt.axes(projection="3d")
    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(6, 6)

    def plot_channels(voxels, channels, ax, key, alpha=1, edgec="black"):
        if not point_cloud:
            for channel in channels:
                ax.voxels(
                    voxels[channel][:],
                    facecolors=color_mapping[key][channel_mapping[channel]],
                    edgecolor=edgec,
                    shade=False,
                    alpha=alpha,
                )
        else:
            plot_points(voxels, channels, ax, key)

    def plot_points(voxels, channels, ax, key, alpha=1):
        for channel in channels:
            x, y, z = (voxels[channel] > 0).nonzero()
            ax.scatter(
                x,
                y,
                z,
                c=color_mapping[key][channel_mapping[channel]],
                alpha=alpha,
                s=50,
                marker="o",
                # edgecolors="k",
            )

    if protein is not None:
        if show_channels == "all":
            plot_channels(protein, range(7), ax, "protein")
        else:
            plot_channels(protein, show_channels, ax, "protein")

    if ligand is not None:
        if show_channels == "all":
            plot_channels(ligand, range(7), ax, "ligand", alpha=alpha)
        else:
            plot_channels(
                ligand,
                show_channels,
                ax,
                "ligand",
                alpha=alpha,
                # edgec=color_mapping["ligand"]["C"],
            )

    # make planes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # make gridlines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    # limit x axis
    ax.set_xlim(dim - (zoom * dim), dim - (dim - (dim * zoom)))
    ax.set_ylim(dim - (zoom * dim), dim - (dim - (dim * zoom)))
    ax.set_zlim(dim - (zoom * dim), dim - (dim - (dim * zoom)))

    # remove axis
    ax.set_axis_off()
    ax.view_init(elev=15, azim=120)
    if show_box:
        draw_box(ax, dim=dim)

    # plt.title(title)
    if show_animation:

        def animate(i):
            ax.view_init(elev=0.0, azim=3.6 * i)
            return (fig,)

        # Animate
        fig.set_size_inches(5, 5)
        ani = animation.FuncAnimation(
            fig, animate, frames=100, interval=(60 * 100) / 120, blit=False
        )
        animation.Animation.save(
            ani,
            "../data/images/atom_channels.mp4",
            fps=20,
            dpi=800,
        )

        return ani

    if save_svg is not None:
        plt.savefig(save_svg, format="svg", dpi=800, transparent=True)
        plt.close()

    plt.show()


def show_electron_densities(
    voxel_array,
    type="ed",
    point_cloud=True,
    dim=35,
    edens_threshold=0.1,
    show_animation=False,
    show_box=False,
):
    def plot_points(voxels, ax):
        if type == "ed":
            x_pos, y_pos, z_pos = (voxels > edens_threshold).nonzero()
            pos_color_mask = np.where(
                (voxels > edens_threshold).nonzero(), "#0000FF", None
            )[0]
        elif type == "esp":
            x_pos, y_pos, z_pos = (voxels > 0).nonzero()
            pos_color_mask = np.where(
                (voxels > 0).nonzero(), "#0000FF", "#FF0000"
            )[0]
            x_neg, y_neg, z_neg = (voxels < 0).nonzero()
            neg_color_mask = np.where(
                (voxels < 0).nonzero(), "#FF0000", "#0000FF"
            )[0]
        else:
            raise ValueError(
                """Type must be either 'ed' for electron density or 'esp' for
                electrostatic potential"""
            )

        ax.scatter(
            x_pos,
            y_pos,
            z_pos,
            c=pos_color_mask,
            alpha=1,
            s=50,
            marker="o",
            edgecolors="k",
        )
        if type == "esp":
            ax.scatter(
                x_neg,
                y_neg,
                z_neg,
                c=neg_color_mask,
                alpha=1,
                s=50,
                marker="o",
                edgecolors="k",
            )

    color_mask = np.where(voxel_array > 0, "#0000FF", "#FF0000")

    plt.clf()
    ax = plt.figure().add_subplot(
        projection="3d",
    )
    fig = plt.gcf()

    if point_cloud:
        plot_points(voxel_array, ax)
    else:
        if type == "ed":
            voxel_array = np.where(voxel_array < edens_threshold, 0, 1)

        ax.voxels(
            voxel_array,
            facecolors=color_mask,
            edgecolor="k",
            shade=True,
            alpha=0.5,
        )

    if show_box:
        draw_box(ax, dim=dim)

    # limit x axis
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dim)
    ax.set_zlim(0, dim)

    # make planes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # make gridlines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    # remove axis
    ax.set_axis_off()

    if show_animation:

        def animate(i):
            ax.view_init(elev=0.0, azim=3.6 * i)
            return (fig,)

        # Animate
        fig.set_size_inches(5, 5)
        ani = animation.FuncAnimation(
            fig, animate, frames=100, interval=100, blit=False
        )
        animation.Animation.save(
            ani, "../data/images/edens.mp4", fps=20, dpi=800
        )

        return ani
    plt.show()
    return None
