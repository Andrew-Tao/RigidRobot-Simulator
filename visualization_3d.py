import numpy as np
import plotly.graph_objects as go


EPS = 1e-9

VALID_AXES = ("theta1", "kappa1", "theta2", "kappa2")
AXIS_LABELS = {
    "theta1": "theta1 (rad)",
    "kappa1": "kappa1 (1/length)",
    "theta2": "theta2 (rad)",
    "kappa2": "kappa2 (1/length)",
}


def _equal_3d_scene_ranges(points: np.ndarray, pad: float = 0.15):
    """Return equal xyz axis ranges for Plotly 3D scene."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    span = max(span, 1.0)
    half = 0.5 * span * (1.0 + pad)

    return dict(
        x=[center[0] - half, center[0] + half],
        y=[center[1] - half, center[1] + half],
        z=[center[2] - half, center[2] + half],
    )


def plot_two_segment_arm_3d_interactive(
    seg1_points: np.ndarray,
    seg2_points: np.ndarray,
    base: np.ndarray,
    joint: np.ndarray,
    tip: np.ndarray,
    target: np.ndarray,
    tangent_tip: np.ndarray,
    gap_vec: np.ndarray,
    sigma: float,
    align: float,
    distance: float,
    theta1: float,
    kappa1: float,
    theta2: float,
    kappa2: float,
    tangent_scale: float = 0.25,
    show: bool = True,
):
    """
    Interactive Plotly visualization of a two-segment 3D arm.
    Supports mouse rotate / zoom / pan.
    """

    base = np.asarray(base).reshape(3,)
    joint = np.asarray(joint).reshape(3,)
    tip = np.asarray(tip).reshape(3,)
    target = np.asarray(target).reshape(3,)
    tangent_tip = np.asarray(tangent_tip).reshape(3,)
    gap_vec = np.asarray(gap_vec).reshape(3,)

    seg1_points = np.asarray(seg1_points)
    seg2_points = np.asarray(seg2_points)

    fig = go.Figure()

    # Segment 1
    fig.add_trace(go.Scatter3d(
        x=seg1_points[:, 0],
        y=seg1_points[:, 1],
        z=seg1_points[:, 2],
        mode="lines",
        line=dict(width=8),
        name="Segment 1",
    ))

    # Segment 2
    fig.add_trace(go.Scatter3d(
        x=seg2_points[:, 0],
        y=seg2_points[:, 1],
        z=seg2_points[:, 2],
        mode="lines",
        line=dict(width=8),
        name="Segment 2",
    ))

    # Key points
    fig.add_trace(go.Scatter3d(
        x=[base[0]],
        y=[base[1]],
        z=[base[2]],
        mode="markers",
        marker=dict(size=6, color="black"),
        name="Base",
    ))

    fig.add_trace(go.Scatter3d(
        x=[joint[0]],
        y=[joint[1]],
        z=[joint[2]],
        mode="markers",
        marker=dict(size=6, color="purple"),
        name="Joint",
    ))

    fig.add_trace(go.Scatter3d(
        x=[tip[0]],
        y=[tip[1]],
        z=[tip[2]],
        mode="markers",
        marker=dict(size=7, color="orange"),
        name="Tip",
    ))

    fig.add_trace(go.Scatter3d(
        x=[target[0]],
        y=[target[1]],
        z=[target[2]],
        mode="markers",
        marker=dict(size=7, color="red", symbol="x"),
        name="Target",
    ))

    # Tangent vector at tip
    tangent_norm = np.linalg.norm(tangent_tip)
    if tangent_norm > EPS:
        tangent_unit = tangent_tip / tangent_norm
        tangent_end = tip + tangent_scale * tangent_unit

        fig.add_trace(go.Scatter3d(
            x=[tip[0], tangent_end[0]],
            y=[tip[1], tangent_end[1]],
            z=[tip[2], tangent_end[2]],
            mode="lines",
            line=dict(width=6, color="green"),
            name="Tip tangent",
        ))

    # Gap vector: tip -> target
    if np.linalg.norm(gap_vec) > EPS:
        gap_end = tip + gap_vec
        fig.add_trace(go.Scatter3d(
            x=[tip[0], gap_end[0]],
            y=[tip[1], gap_end[1]],
            z=[tip[2], gap_end[2]],
            mode="lines",
            line=dict(width=4, color="gray", dash="dash"),
            name="Tip to target",
        ))

    # Dotted connection tip-target
    fig.add_trace(go.Scatter3d(
        x=[tip[0], target[0]],
        y=[tip[1], target[1]],
        z=[tip[2], target[2]],
        mode="lines",
        line=dict(width=3, color="gray", dash="dot"),
        name="Tip-target line",
        showlegend=False,
    ))

    all_points = np.vstack([seg1_points, seg2_points, base, joint, tip, target])
    ranges = _equal_3d_scene_ranges(all_points)

    title_text = (
        "Two-Segment 3D Constant-Curvature Arm<br>"
        f"theta1={theta1:.3f}, kappa1={kappa1:.3f}, "
        f"theta2={theta2:.3f}, kappa2={kappa2:.3f}<br>"
        f"distance={distance:.6f}, alignment={align:.6f}, sigma={sigma:.6f}"
    )

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis=dict(title="x", range=ranges["x"]),
            yaxis=dict(title="y", range=ranges["y"]),
            zaxis=dict(title="z", range=ranges["z"]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=90),
        legend=dict(x=0.02, y=0.98),
    )

    if show:
        fig.show()

    return fig

import numpy as np
import matplotlib.pyplot as plt


def _set_equal_3d_axes(ax, points: np.ndarray, pad: float = 0.15) -> None:
    """Set equal scaling for a 3D axis based on a set of points."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    span = max(span, 1.0)
    half = 0.5 * span * (1.0 + pad)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)



def plot_two_segment_arm_3d(
    seg1_points: np.ndarray,
    seg2_points: np.ndarray,
    base: np.ndarray,
    joint: np.ndarray,
    tip: np.ndarray,
    target: np.ndarray,
    tangent_tip: np.ndarray,
    gap_vec: np.ndarray,
    sigma: float,
    align: float,
    distance: float,
    theta1: float,
    kappa1: float,
    theta2: float,
    kappa2: float,
    tangent_scale: float = 0.25,
    gap_scale: float = 1.0,
    show: bool = True,
):
    """Visualize the two-segment 3D arm, target, tip tangent, and tip-to-target vector."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(seg1_points[:, 0], seg1_points[:, 1], seg1_points[:, 2], linewidth=4, label="Segment 1")
    ax.plot(seg2_points[:, 0], seg2_points[:, 1], seg2_points[:, 2], linewidth=4, label="Segment 2")

    ax.scatter(*base, color="black", s=60, label="Base")
    ax.scatter(*joint, color="purple", s=60, label="Joint")
    ax.scatter(*tip, color="orange", s=80, label="Tip")
    ax.scatter(*target, color="red", s=80, marker="x", label="Target")

    ax.quiver(
        tip[0], tip[1], tip[2],
        tangent_tip[0], tangent_tip[1], tangent_tip[2],
        length=tangent_scale,
        normalize=True,
        linewidth=2.0,
        color="green",
        label="Tip tangent",
    )

    if np.linalg.norm(gap_vec) > EPS:
        ax.quiver(
            tip[0], tip[1], tip[2],
            gap_vec[0], gap_vec[1], gap_vec[2],
            length=gap_scale,
            normalize=False,
            linewidth=1.8,
            color="gray",
            linestyle="dashed",
            label="Tip to target",
        )

    ax.plot(
        [tip[0], target[0]],
        [tip[1], target[1]],
        [tip[2], target[2]],
        linestyle=":",
        linewidth=1.8,
        color="gray",
    )

    all_points = np.vstack([seg1_points, seg2_points, base, joint, tip, target])
    _set_equal_3d_axes(ax, all_points)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        "Two-Segment 3D Constant-Curvature Arm\n"
        f"theta1={theta1:.3f}, kappa1={kappa1:.3f}, theta2={theta2:.3f}, kappa2={kappa2:.3f}\n"
        f"distance={distance:.6f}, alignment={align:.6f}, sigma={sigma:.6f}"
    )
    ax.legend(loc="best")

    if show:
        plt.show()

    return fig, ax


def plot_sigma_workspace_3d(
    axis_names,
    axis_values,
    sigma_values,
    fixed_params,
    show: bool = True,
    marker_size: float = 600.0,
    alpha: float = 0.55,
):
    """
    Plot a 3D workspace map of sigma values.

    Parameters
    ----------
    axis_names : tuple[str, str, str]
        Which three parameters define the x/y/z axes. Each must be one of
        ('theta1', 'kappa1', 'theta2', 'kappa2').
    axis_values : tuple[np.ndarray, np.ndarray, np.ndarray]
        Flattened x, y, z coordinates corresponding to sigma_values.
    sigma_values : np.ndarray
        Flattened sigma values used for the color map.
    fixed_params : dict
        The remaining parameter(s) held fixed.
    """
    if len(axis_names) != 3:
        raise ValueError("axis_names must contain exactly three entries.")
    if len(set(axis_names)) != 3:
        raise ValueError("axis_names must be unique.")
    for name in axis_names:
        if name not in VALID_AXES:
            raise ValueError(f"Invalid axis name '{name}'. Valid choices: {VALID_AXES}")

    x, y, z = axis_values
    sigma_values = np.asarray(sigma_values, dtype=float)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    threshold = 0.2  # choose your value

    mask = sigma_values >= threshold

    x_plot = x[mask]
    y_plot = y[mask]
    z_plot = z[mask]
    sigma_plot = sigma_values[mask]
    print(sigma_plot.shape)

    scatter = ax.scatter(
        x_plot, y_plot, z_plot,
        c=sigma_plot,
        cmap="hot",
        s=80,
        alpha=0.7,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.08)
    cbar.set_label("sigma")

    ax.set_xlabel(AXIS_LABELS[axis_names[0]])
    ax.set_ylabel(AXIS_LABELS[axis_names[1]])
    ax.set_zlabel(AXIS_LABELS[axis_names[2]])

    fixed_text = ", ".join(f"{k}={v:.4f}" for k, v in fixed_params.items())
    if fixed_text:
        title = f"3D Sigma Workspace: {axis_names[0]}, {axis_names[1]}, {axis_names[2]}\nFixed: {fixed_text}"
    else:
        title = f"3D Sigma Workspace: {axis_names[0]}, {axis_names[1]}, {axis_names[2]}"
    ax.set_title(title)

    if show:
        plt.show()

    return fig, ax
