import numpy as np
import matplotlib.pyplot as plt

def visualize_rod(r_collection, t_collection=None, tangent_scale=0.1, show_tangent=True):
    """
    Plot Cosserat rod centerline using r_collection.

    Parameters
    ----------
    r_collection : np.ndarray
        Shape should be (n_elem + 1, 3). Each row is a node position [x, y, z].
    t_collection : np.ndarray, optional
        Shape should be (n_elem, 3). Each row is the tangent direction of one segment.
    tangent_scale : float
        Length scale for tangent arrows.
    show_tangent : bool
        Whether to show tangent vectors.
    """

    r = np.asarray(r_collection)

    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r_collection must have shape (n_nodes, 3)")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    # Plot rod centerline
    ax.plot(x, y, z, "-o", linewidth=2, markersize=4, label="Rod centerline")

    # Plot tangent arrows
    if show_tangent and t_collection is not None:
        t = np.asarray(t_collection)

        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError("t_collection must have shape (n_elem, 3)")

        for i in range(min(len(t), len(r) - 1)):
            start = r[i]
            direction = t[i]

            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            ax.quiver(
                start[0], start[1], start[2],
                direction[0], direction[1], direction[2],
                length=tangent_scale,
                normalize=True
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rod Visualization")

    # Make axes equal
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max()

    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.legend()
    plt.show()