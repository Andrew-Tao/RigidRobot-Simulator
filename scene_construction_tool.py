import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _draw_cuboid(
    ax,
    origin,
    size,
    color="#888888",
    alpha=0.22,
    edgecolor="#555555",
    linewidth=1.0,
):
    """
    Draw an axis-aligned solid rectangular cuboid.

    Parameters
    ----------
    origin : array-like, shape (3,)
        Lower-left-bottom corner: [x0, y0, z0]
    size : array-like, shape (3,)
        [width_x, width_y, height_z]
    """
    origin = np.asarray(origin, dtype=float)
    size = np.asarray(size, dtype=float)

    x0, y0, z0 = origin
    dx, dy, dz = size

    # 8 vertices
    p000 = np.array([x0,      y0,      z0])
    p100 = np.array([x0 + dx, y0,      z0])
    p110 = np.array([x0 + dx, y0 + dy, z0])
    p010 = np.array([x0,      y0 + dy, z0])

    p001 = np.array([x0,      y0,      z0 + dz])
    p101 = np.array([x0 + dx, y0,      z0 + dz])
    p111 = np.array([x0 + dx, y0 + dy, z0 + dz])
    p011 = np.array([x0,      y0 + dy, z0 + dz])

    faces = [
        [p000, p100, p110, p010],  # bottom
        [p001, p101, p111, p011],  # top
        [p000, p100, p101, p001],  # front
        [p010, p110, p111, p011],  # back
        [p000, p010, p011, p001],  # left
        [p100, p110, p111, p101],  # right
    ]

    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly)

    return np.vstack([p000, p100, p110, p010, p001, p101, p111, p011])

def _draw_cylinder(
    ax,
    center_xy,
    radius,
    z0,
    height,
    color="#6C757D",
    alpha=0.28,
    resolution=40,
    edgecolor=None,
):
    """
    Draw a solid vertical cylinder.

    Parameters
    ----------
    center_xy : array-like, shape (2,)
        Cylinder center in xy plane
    radius : float
        Cylinder radius
    z0 : float
        Bottom z
    height : float
        Cylinder height
    """
    cx, cy = np.asarray(center_xy, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    z = np.linspace(z0, z0 + height, 2)

    Theta, Z = np.meshgrid(theta, z)
    X = cx + radius * np.cos(Theta)
    Y = cy + radius * np.sin(Theta)

    # side surface
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color, linewidth=0, shade=True)

    # top and bottom caps
    rr = np.linspace(0.0, radius, resolution // 2 + 1)
    Theta2, RR = np.meshgrid(theta, rr)
    Xcap = cx + RR * np.cos(Theta2)
    Ycap = cy + RR * np.sin(Theta2)

    Zbot = np.full_like(Xcap, z0)
    Ztop = np.full_like(Xcap, z0 + height)

    ax.plot_surface(Xcap, Ycap, Zbot, alpha=alpha, color=color, linewidth=0, shade=True)
    ax.plot_surface(Xcap, Ycap, Ztop, alpha=alpha, color=color, linewidth=0, shade=True)

    # useful bounding points for autoscaling
    pts = np.array([
        [cx - radius, cy - radius, z0],
        [cx + radius, cy + radius, z0 + height],
    ])
    return pts

def _draw_scene_objects(ax, cuboids=None, cylinders=None):
    """
    Draw scene objects and return points for axis autoscaling.
    """
    all_pts = []

    if cuboids is not None:
        for cuboid in cuboids:
            pts = _draw_cuboid(
                ax,
                origin=cuboid["origin"],
                size=cuboid["size"],
                color=cuboid.get("color", "#888888"),
                alpha=cuboid.get("alpha", 0.22),
                edgecolor=cuboid.get("edgecolor", "#555555"),
                linewidth=cuboid.get("linewidth", 1.0),
            )
            all_pts.append(pts)

    if cylinders is not None:
        for cyl in cylinders:
            pts = _draw_cylinder(
                ax,
                center_xy=cyl["center_xy"],
                radius=cyl["radius"],
                z0=cyl["z0"],
                height=cyl["height"],
                color=cyl.get("color", "#6C757D"),
                alpha=cyl.get("alpha", 0.28),
                resolution=cyl.get("resolution", 40),
            )
            all_pts.append(pts)

    if len(all_pts) == 0:
        return None

    return np.vstack(all_pts)