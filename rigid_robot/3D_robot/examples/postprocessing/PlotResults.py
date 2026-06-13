import numpy as np
import matplotlib.pyplot as plt


def plot_disk_chain_results(time_collection, posture_collection, orientation_collection, force_collection, title="Robot Simulation"):
    N_disks = posture_collection.shape[1]
    cmap = plt.cm.get_cmap('viridis', N_disks)
    show_all_labels = N_disks <= 10
    label_set = {0, N_disks // 2, N_disks - 1}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    for i in range(N_disks):
        label = f"disk {i + 1}" if (show_all_labels or i in label_set) else None
        color = cmap(i / max(N_disks - 1, 1))
        pos_i = posture_collection[:, i, :3, 3]
        axes[0, 0].plot(time_collection, pos_i[:, 0], color=color, label=label)
        axes[0, 1].plot(time_collection, pos_i[:, 2], color=color, label=label)
        axes[1, 0].plot(time_collection, orientation_collection[:, i, 0], color=color, label=label)
        axes[1, 1].plot(time_collection, orientation_collection[:, i, 2], color=color, label=label)

    plot_cfg = [
        (axes[0, 0], "Time (s)", "Position X (m)",  "Position X"),
        (axes[0, 1], "Time (s)", "Position Z (m)",  "Position Z"),
        (axes[1, 0], "Time (s)", "Theta X (rad)",   "Orientation Theta X"),
        (axes[1, 1], "Time (s)", "Theta Z (rad)",   "Orientation Theta Z"),
    ]
    for ax, xlabel, ylabel, title_ in plot_cfg:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()

    force_labels = ["fx (N)", "fy (N)", "fz (N)", "tx (N·m)", "ty (N·m)", "tz (N·m)"]
    force_titles = ["Force X", "Force Y", "Force Z", "Torque X", "Torque Y", "Torque Z"]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle(f"{title}: Internal Forces & Torques on Each Disk")

    for idx in range(6):
        row, col = divmod(idx, 3)
        ax = axes2[row, col]
        for i in range(N_disks):
            label = f"disk {i + 1}" if (show_all_labels or i in label_set) else None
            color = cmap(i / max(N_disks - 1, 1))
            ax.plot(time_collection, force_collection[:, i, idx], color=color, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(force_labels[idx])
        ax.set_title(force_titles[idx])
        ax.legend(fontsize=8)
        ax.grid(True)

    fig2.tight_layout()
    plt.show()
