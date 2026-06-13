"""
3D animation tool for the Spirob slender disk-chain robot.

Main entry point: animate_slender_robot(...)

Saves as MP4 when ffmpeg is on PATH; falls back to GIF (PillowWriter) otherwise.
Install ffmpeg for MP4: brew install ffmpeg  |  conda install -c conda-forge ffmpeg
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# ── helpers ─────────────────────────────────────────────────────────────────

def _disk_circle_world(posture: np.ndarray, radius: float, n_pts: int = 36) -> np.ndarray:
    """Return (n_pts+1, 3) vertices of a disk rim in world frame."""
    theta = np.linspace(0, 2 * np.pi, n_pts + 1)
    local = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(n_pts + 1)])
    R, t = posture[:3, :3], posture[:3, 3]
    return (R @ local.T).T + t


def _get_colors(n: int, cmap_name: str = 'plasma') -> list:
    """Return n distinct RGBA colours from a matplotlib colormap."""
    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


_CABLE_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                 '#1abc9c', '#e67e22', '#8e44ad', '#16a085', '#d35400']


# ── main public function ─────────────────────────────────────────────────────

def animate_slender_robot(
    time_collection,
    posture_collection,
    force_collection=None,
    hole_offset=None,
    disk_radius: float = 0.03,
    output_path: str = 'slender_robot.mp4',
    fps: int = 15,
    force_scale: float = 1.0,
    skip_frames: int = 1,
    show_tension_arrows: bool = True,
    view_yaw: float = -60.0,
    view_pitch: float = 30.0,
    view_roll: float = 0.0,
):
    """
    Generate an MP4 animation of a slender disk-chain robot.

    Parameters
    ----------
    time_collection    : array-like (T,)
        Simulation timestamps in seconds.
    posture_collection : array-like (T, N, 4, 4)
        SE3 posture matrices for each disk at each timestep.
        Position = posture[:3, 3], rotation = posture[:3, :3].
    force_collection   : array-like (T, N, 6) or None
        Total generalised force [fx,fy,fz,tx,ty,tz] in LOCAL frame on each disk.
        When provided, net-force arrows are drawn at each disk centre in world frame.
    hole_offset        : array-like (C, 3) or None
        Offset of each cable hole from the disk centre, expressed in world frame
        (no rotation applied — consistent with CableDrivenForce convention).
        C is the number of cables (typically 3).
        When provided, cable paths and tension-direction arrows are drawn.
    disk_radius        : float
        Disk radius used for rendering circles.
    output_path        : str
        Destination .mp4 file path.
    fps                : int
        Frames per second in the output video.
    force_scale        : float
        Multiplicative scale for the net-force arrow length.
    skip_frames        : int
        Render every skip_frames-th timestep.  Increase for large simulations.
    show_tension_arrows: bool
        Draw small arrows at each cable hole showing tension direction.
        Automatically disabled when N > 20 to keep the scene readable.
    view_yaw   : float
        Camera azimuth angle in degrees (rotation around world Z). Default -60.
    view_pitch : float
        Camera elevation angle in degrees above the horizontal plane. Default 30.
    view_roll  : float
        Camera roll angle in degrees around the line of sight. Default 0.
    """
    posture_collection = np.asarray(posture_collection, dtype=float)
    time_collection    = np.asarray(time_collection,    dtype=float)
    T, N = posture_collection.shape[:2]
    frame_indices = np.arange(0, T, max(1, skip_frames))
    n_frames = len(frame_indices)

    disk_colors = _get_colors(N, 'plasma')
    ho = np.asarray(hole_offset, dtype=float) if hole_offset is not None else None

    # ── scene bounds ────────────────────────────────────────────────────────
    all_pos = posture_collection[:, :, :3, 3]           # (T, N, 3)
    if ho is not None:
        # Hole positions: centre + offset, no rotation (matches CableDrivenForce)
        all_holes = all_pos[:, :, np.newaxis, :] + ho[np.newaxis, np.newaxis, :, :]
        all_pts = np.concatenate([all_pos.reshape(-1, 3), all_holes.reshape(-1, 3)], axis=0)
    else:
        all_pts = all_pos.reshape(-1, 3)

    pad = max(disk_radius * 4, 0.02)
    mid = all_pts.mean(axis=0)
    half = max(
        all_pts[:, 0].max() - all_pts[:, 0].min(),
        all_pts[:, 1].max() - all_pts[:, 1].min(),
        all_pts[:, 2].max() - all_pts[:, 2].min(),
    ) / 2 + pad
    xl = (mid[0] - half, mid[0] + half)
    yl = (mid[1] - half, mid[1] + half)
    zl = (mid[2] - half, mid[2] + half)

    _draw_arrows = show_tension_arrows and N <= 20

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 9))
    ax  = fig.add_subplot(111, projection='3d')

    def draw_frame(t_idx: int):
        ax.cla()
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_zlim(*zl)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(f'Slender Robot  —  t = {time_collection[t_idx]:.3f} s', fontsize=11)
        ax.view_init(elev=view_pitch, azim=view_yaw, roll=view_roll)

        centers = posture_collection[t_idx, :, :3, 3]  # (N, 3)

        # Base anchor
        ax.scatter([0], [0], [0], color='black', s=130, marker='D', zorder=6, label='Base')

        # Backbone dashed line: base → disk 1 → ... → disk N
        bb = np.vstack([[0.0, 0.0, 0.0], centers])
        ax.plot(bb[:, 0], bb[:, 1], bb[:, 2], 'k--', lw=1.2, alpha=0.35)

        # Disk rims
        label_step = max(1, N // 10)
        for i in range(N):
            circle = _disk_circle_world(posture_collection[t_idx, i], disk_radius)
            col = disk_colors[i]
            ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color=col, lw=2.0)
            ax.scatter(*centers[i], color=col, s=22, zorder=5)
            if N <= 20 or i % label_step == 0:
                ax.text(centers[i, 0], centers[i, 1], centers[i, 2],
                        f' {i + 1}', fontsize=6, color=col)

        # Cable paths and tension arrows
        if ho is not None:
            C = len(ho)
            arrow_len = disk_radius * 1.8
            for ci in range(C):
                offset     = ho[ci]
                base_hole  = offset.copy()                  # world-frame base hole
                disk_holes = centers + offset               # (N, 3)
                cable_pts  = np.vstack([base_hole, disk_holes])
                col = _CABLE_COLORS[ci % len(_CABLE_COLORS)]

                ax.plot(cable_pts[:, 0], cable_pts[:, 1], cable_pts[:, 2],
                        color=col, lw=2.0, alpha=0.8, label=f'Cable {ci + 1}')
                ax.scatter(disk_holes[:, 0], disk_holes[:, 1], disk_holes[:, 2],
                           color=col, s=28, marker='o', alpha=0.9, zorder=6)

                if _draw_arrows:
                    for i in range(N):
                        hole_i    = disk_holes[i]
                        prev_hole = base_hole if i == 0 else disk_holes[i - 1]

                        d = prev_hole - hole_i
                        nd = np.linalg.norm(d)
                        if nd > 1e-9:
                            ax.quiver(*hole_i, *(d / nd * arrow_len),
                                      color=col, arrow_length_ratio=0.4,
                                      lw=1.0, alpha=0.65)

                        if i < N - 1:
                            d = disk_holes[i + 1] - hole_i
                            nd = np.linalg.norm(d)
                            if nd > 1e-9:
                                ax.quiver(*hole_i, *(d / nd * arrow_len),
                                          color=col, arrow_length_ratio=0.4,
                                          lw=1.0, alpha=0.65)

        # Net-force arrows at each disk centre (local → world frame)
        if force_collection is not None:
            fc = np.asarray(force_collection, dtype=float)
            for i in range(N):
                R_i     = posture_collection[t_idx, i, :3, :3]
                f_world = R_i @ fc[t_idx, i, :3]
                f_mag   = np.linalg.norm(f_world)
                if f_mag > 1e-9:
                    scale = min(f_mag * force_scale, disk_radius * 4) / f_mag
                    ax.quiver(*centers[i], *(f_world * scale),
                              color='orange', arrow_length_ratio=0.35, lw=1.8,
                              label='Net force' if i == 0 else '')

        # Deduplicated legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        ax.legend(seen.values(), seen.keys(), loc='upper right', fontsize=7)

    def _update(frame_num: int):
        draw_frame(frame_indices[frame_num])

    print(f"Rendering {n_frames} frames for {N} disks...")
    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=int(1000 / fps))

    if shutil.which('ffmpeg') is not None:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        save_path = output_path
    else:
        # ffmpeg not available — fall back to animated GIF via Pillow (no external tool needed)
        base, _ = os.path.splitext(output_path)
        save_path = base + '.gif'
        writer = animation.PillowWriter(fps=fps)
        print("ffmpeg not found — saving as GIF instead.")
        print("To get MP4:  brew install ffmpeg   OR   conda install -c conda-forge ffmpeg")

    ani.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Animation saved → {save_path}")
