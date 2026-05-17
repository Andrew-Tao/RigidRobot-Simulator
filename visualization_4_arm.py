import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scene_construction_tool import _draw_scene_objects

EPS = 1e-9


ARM_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
]


def _set_equal_3d_axes(ax, points: np.ndarray, pad: float = 0.18) -> None:
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    span = max(span, 1.0)
    half = 0.5 * span * (1.0 + pad)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _clean_3d_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((0.985, 0.985, 0.99, 1.0))
    ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.88, 1.0))
    ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.88, 1.0))
    ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.88, 1.0))
    ax.tick_params(labelsize=9, pad=2)


def _draw_vector(ax, origin, vec, color="k", label=None, linewidth=1.8, alpha=1.0):
    origin = np.asarray(origin, dtype=float)
    vec = np.asarray(vec, dtype=float)
    ax.quiver(
        origin[0], origin[1], origin[2],
        vec[0], vec[1], vec[2],
        arrow_length_ratio=0.14,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
    )
    if label:
        tip = origin + vec
        ax.text(tip[0], tip[1], tip[2], label, fontsize=9, color=color)


def _basis_from_normal(normal):
    normal = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(normal)
    if nrm < EPS:
        raise ValueError("plane_normal must be nonzero.")
    ez = normal / nrm

    refs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    ref = refs[int(np.argmin([abs(np.dot(ez, r)) for r in refs]))]
    ex = np.cross(ref, ez)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    return ex, ey, ez


def _draw_mount_plane(ax, base_points, plane_normal=None, plane_origin=None):
    base_points = np.asarray(base_points, dtype=float)
    centroid = base_points.mean(axis=0) if plane_origin is None else np.asarray(plane_origin, dtype=float)

    if plane_normal is None:
        centered = base_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_normal = vh[-1]
    plane_normal = np.asarray(plane_normal, dtype=float)
    nrm = np.linalg.norm(plane_normal)
    if nrm < EPS:
        return centroid

    ex, ey, ez = _basis_from_normal(plane_normal)
    rel = base_points - centroid
    uv = np.column_stack((rel @ ex, rel @ ey))
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    du = max(umax - umin, 1.0) * 0.18 + 0.02
    dv = max(vmax - vmin, 1.0) * 0.18 + 0.02

    corners_2d = np.array([
        [umin - du, vmin - dv],
        [umax + du, vmin - dv],
        [umax + du, vmax + dv],
        [umin - du, vmax + dv],
    ])
    corners_3d = np.array([centroid + u * ex + v * ey for u, v in corners_2d])

    poly = Poly3DCollection(
        [corners_3d],
        facecolors=(0.82, 0.90, 0.98, 0.18),
        edgecolors=(0.55, 0.68, 0.80, 0.7),
        linewidths=1.0,
    )
    ax.add_collection3d(poly)

    # light cross grid on plane
    for frac in (0.25, 0.5, 0.75):
        u = (1 - frac) * (umin - du) + frac * (umax + du)
        p0 = centroid + u * ex + (vmin - dv) * ey
        p1 = centroid + u * ex + (vmax + dv) * ey
        ax.plot(*np.column_stack((p0, p1)), color=(0.55, 0.68, 0.80, 0.32), linewidth=0.9)

        v = (1 - frac) * (vmin - dv) + frac * (vmax + dv)
        p0 = centroid + (umin - du) * ex + v * ey
        p1 = centroid + (umax + du) * ex + v * ey
        ax.plot(*np.column_stack((p0, p1)), color=(0.55, 0.68, 0.80, 0.32), linewidth=0.9)

    _draw_vector(ax, centroid, 0.22 * ez, color="#4C78A8", label="n", linewidth=1.9, alpha=0.9)
    return centroid


def _arm_summary_text(arm_states, sigma_total):
    lines = [f"total σ = {sigma_total:.4f}"]
    for i, state in enumerate(arm_states):
        lines.append(
            f"arm {i+1}:  D={state['distance']:.3f}   A={state['alignment']:.3f}   σ={state['sigma']:.3f}"
        )
    return "\n".join(lines)


def plot_four_arm_static(
    arm_states,
    target,
    plane_normal=None,
    title=None,
    tangent_scale=0.20,
    show_sigma_text=True,
    cuboids=None,
    cylinders=None,
    show=True,
):
    """
    Plot four two-segment arms in 3D with cleaner styling.

    Parameters
    ----------
    arm_states : list[dict]
        One dict per arm. Each dict should contain:
            base, joint, tip, tangent,
            seg1_points, seg2_points,
            distance, alignment, sigma
    target : array-like, shape (3,)
    plane_normal : array-like, optional
        If given, a mounting plane and a normal vector are drawn.
    """
    if len(arm_states) != 4:
        raise ValueError("arm_states must contain exactly 4 entries.")

    target = np.asarray(target, dtype=float)
    fig = plt.figure(figsize=(11.5, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axes(ax)

    sigma_total = 0.0
    all_points = [target[None, :]]
    base_points = np.array([np.asarray(s["base"], dtype=float) for s in arm_states])

    plane_centroid = _draw_mount_plane(ax, base_points, plane_normal=plane_normal)
    all_points.append(base_points)
    all_points.append(plane_centroid[None, :])

    sigma_total = 0.0
    all_points = [target[None, :]]
    base_points = np.array([np.asarray(s["base"], dtype=float) for s in arm_states])

    plane_centroid = _draw_mount_plane(ax, base_points, plane_normal=plane_normal)
    all_points.append(base_points)
    all_points.append(plane_centroid[None, :])

    # NEW: draw user-defined cuboids/cylinders
    scene_pts = _draw_scene_objects(ax, cuboids=cuboids, cylinders=cylinders)
    if scene_pts is not None:
        all_points.append(scene_pts)

    for i, state in enumerate(arm_states):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        seg1 = np.asarray(state["seg1_points"], dtype=float)
        seg2 = np.asarray(state["seg2_points"], dtype=float)
        base = np.asarray(state["base"], dtype=float)
        joint = np.asarray(state["joint"], dtype=float)
        tip = np.asarray(state["tip"], dtype=float)
        tangent = np.asarray(state["tangent"], dtype=float)
        sigma_i = float(state.get("sigma", np.nan))
        sigma_total += sigma_i if np.isfinite(sigma_i) else 0.0

        ax.plot(
            seg1[:, 0], seg1[:, 1], seg1[:, 2],
            color=color, linewidth=4.0, solid_capstyle="round", alpha=0.95,
            label=f"Arm {i+1}" if i == 0 else None,
        )
        ax.plot(
            seg2[:, 0], seg2[:, 1], seg2[:, 2],
            color=color, linewidth=4.0, solid_capstyle="round", alpha=0.95,
        )

        # subtle skeleton underlay for depth
        arm_polyline = np.vstack((seg1, seg2[1:]))
        ax.plot(
            arm_polyline[:, 0], arm_polyline[:, 1], arm_polyline[:, 2],
            color="black", linewidth=6.0, alpha=0.06,
        )

        ax.scatter(base[0], base[1], base[2], s=90, color=color, edgecolors="white", linewidths=1.5, zorder=10)
        ax.scatter(joint[0], joint[1], joint[2], s=70, color=color, edgecolors="white", linewidths=1.2, zorder=11)
        ax.scatter(tip[0], tip[1], tip[2], s=140, marker="o", color=color, edgecolors="white", linewidths=1.8, zorder=12)

        _draw_vector(ax, tip, tangent_scale * tangent, color=color, linewidth=2.0, alpha=0.95)
        gap = target - tip
        ax.plot(
            [tip[0], target[0]],
            [tip[1], target[1]],
            [tip[2], target[2]],
            linestyle="--", linewidth=1.3, color=color, alpha=0.45,
        )

        base_offset = np.array([0.015, 0.015, 0.015])
        tip_offset = np.array([0.02, 0.02, 0.02])
        ax.text(*(base + base_offset), f"B{i+1}", fontsize=10, weight="bold", color=color)
        ax.text(*(tip + tip_offset), f"A{i+1}", fontsize=10, weight="bold", color=color)

        all_points.extend([seg1, seg2, base[None, :], joint[None, :], tip[None, :], (tip + tangent_scale * tangent)[None, :]])

    ax.scatter(
        target[0], target[1], target[2],
        s=260, marker="*", color="#D62728", edgecolors="white", linewidths=1.4,
        label="Target", zorder=20,
    )

    all_points = np.vstack(all_points)
    _set_equal_3d_axes(ax, all_points)

    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=8)
    ax.view_init(elev=24, azim=-52)

    if title is None:
        title = "Four-arm two-segment optimizer"
    ax.set_title(title, fontsize=14, pad=16, weight="bold")

    if show_sigma_text:
        summary = _arm_summary_text(arm_states, sigma_total)
        ax.text2D(
            0.02, 0.98, summary,
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#D9DCE3", alpha=0.95),
        )
    else:
        ax.text2D(
            0.02, 0.98, f"total σ = {sigma_total:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D9DCE3", alpha=0.95),
        )

    legend_handles = [
        plt.Line2D([0], [0], color=ARM_COLORS[i], lw=4, label=f"Arm {i+1}")
        for i in range(4)
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", linestyle="None", markersize=13,
                   markerfacecolor="#D62728", markeredgecolor="white", label="Target")
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_base_layout(
    base_points,
    plane_origin=None,
    plane_normal=None,
    cuboids=None,
    cylinders=None,
    show=True,
):
    if show == False:
        return 0

    """Cleaner plot showing the four arm bases and the mounting plane."""
    base_points = np.asarray(base_points, dtype=float)
    if base_points.shape != (4, 3):
        raise ValueError("base_points must have shape (4, 3).")

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axes(ax)

    if plane_origin is None:
        plane_origin = base_points.mean(axis=0)
    else:
        plane_origin = np.asarray(plane_origin, dtype=float)

    centroid = _draw_mount_plane(ax, base_points, plane_normal=plane_normal, plane_origin=plane_origin)

    scene_pts = _draw_scene_objects(ax, cuboids=cuboids, cylinders=cylinders)
    all_points = [base_points, plane_origin[None, :], centroid[None, :]]
    if scene_pts is not None:
        all_points.append(scene_pts)
    _set_equal_3d_axes(ax, np.vstack(all_points))

    ax.plot(
        np.r_[base_points[:, 0], base_points[0, 0]],
        np.r_[base_points[:, 1], base_points[0, 1]],
        np.r_[base_points[:, 2], base_points[0, 2]],
        color="#7A869A", linewidth=1.2, linestyle="--", alpha=0.5,
    )

    for i, p in enumerate(base_points):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        ax.scatter(p[0], p[1], p[2], s=120, color=color, edgecolors="white", linewidths=1.6, zorder=10)
        ax.text(p[0] + 0.02, p[1] + 0.02, p[2] + 0.02, f"B{i+1}", fontsize=10, weight="bold", color=color)

    ax.scatter(
        plane_origin[0], plane_origin[1], plane_origin[2],
        s=90, marker="X", color="#444444", edgecolors="white", linewidths=1.2, zorder=11,
    )
    ax.text(
        plane_origin[0] + 0.02, plane_origin[1] + 0.02, plane_origin[2] + 0.02,
        "origin", fontsize=9, color="#444444"
    )

    all_points = np.vstack((base_points, plane_origin[None, :], centroid[None, :]))
    _set_equal_3d_axes(ax, all_points)
    ax.view_init(elev=24, azim=-52)
    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=8)
    ax.set_title("Four-arm base layout", fontsize=14, pad=16, weight="bold")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
