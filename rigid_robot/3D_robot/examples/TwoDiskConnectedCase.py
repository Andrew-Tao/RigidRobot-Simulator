"""
Simulation of a serial chain of rigid disk robots connected by springs and torsional springs.

This example models a vertical series of rigid disk robots hanging under gravity.
Each disk is connected to the next disk through a linear spring and a torsional spring,
allowing both translational and rotational coupling between neighboring rigid bodies.

The chain is anchored at the base point (0, 0, 0). The first disk, robot1, is attached
to this fixed base anchor. The remaining disks are connected sequentially, forming a
serial rigid-robot chain. The final disk, robot n, is the bottom disk of the hanging
system.


The purpose of this simulation is to study the dynamic response of a spring-connected
multi-rigid-body system under gravity, including the effects of both linear spring
forces and torsional spring moments.
"""


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.RigidRobot3D import RigidRobot3D, ConnectedRigidRobots3D
from robot3d.Simulator3D import Simulator3D, MutiRobotSimulator3D
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


lie3 = SE3LieAlgebra()

if __name__ == "__main__":

    robot_disk_1 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]),
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,
    )

    robot_disk_2 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.08]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]),
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,
    )
    robot_collection = [robot_disk_1, robot_disk_2]

    slender_robot = ConnectedRigidRobots3D(robots=robot_collection)
    slender_robot.add_connection((0, 1), to_base=True)
    slender_robot.add_connection((0, 1), to_base=False)

    simulator_slender = MutiRobotSimulator3D(
        time_step=0.01,
        duration=30,
        stepper='explicit_euler',
        control_logic=None,
    )

    simulator_slender.attach(slender_robot)
    simulator_slender.connected_robot.robots[1].control_input = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    while simulator_slender.run():
        if simulator_slender.current_time >= 0.5:
            simulator_slender.connected_robot.robots[1].control_input = np.zeros(6)
        simulator_slender.multi_robots_step()
        simulator_slender.multi_robot_record()

    # Data extraction
    time_collection = np.array(simulator_slender.time_collection)
    posture_collection = np.array(simulator_slender.posture_collection)
    orientation_collection = np.array(simulator_slender.orientation_collection)
    force_collection = np.array(simulator_slender.force_collection)

    force_robot_1 = force_collection[:, 0]  # shape: (T, 6)  [fx, fy, fz, tx, ty, tz]
    force_robot_2 = force_collection[:, 1]

    theta_x_1 = orientation_collection[:, 0, 0]
    theta_z_1 = orientation_collection[:, 0, 2]
    theta_x_2 = orientation_collection[:, 1, 0]
    theta_z_2 = orientation_collection[:, 1, 2]

    position_robot_1 = posture_collection[:, 0, :3, 3]
    position_robot_2 = posture_collection[:, 1, :3, 3]
    position_x_1 = position_robot_1[:, 0]
    position_z_1 = position_robot_1[:, 2]
    position_x_2 = position_robot_2[:, 0]
    position_z_2 = position_robot_2[:, 2]

    # --- Position & Orientation ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 7))
    fig1.suptitle("Two-Disk Robot: Position & Orientation")

    axes1[0, 0].plot(time_collection, position_x_1, label="Disk 1")
    axes1[0, 0].plot(time_collection, position_x_2, label="Disk 2")
    axes1[0, 0].set_xlabel("Time (s)")
    axes1[0, 0].set_ylabel("Position X (m)")
    axes1[0, 0].set_title("Position X")
    axes1[0, 0].legend()
    axes1[0, 0].grid(True)

    axes1[0, 1].plot(time_collection, position_z_1, label="Disk 1")
    axes1[0, 1].plot(time_collection, position_z_2, label="Disk 2")
    axes1[0, 1].set_xlabel("Time (s)")
    axes1[0, 1].set_ylabel("Position Z (m)")
    axes1[0, 1].set_title("Position Z")
    axes1[0, 1].legend()
    axes1[0, 1].grid(True)

    axes1[1, 0].plot(time_collection, theta_x_1, label="Disk 1")
    axes1[1, 0].plot(time_collection, theta_x_2, label="Disk 2")
    axes1[1, 0].set_xlabel("Time (s)")
    axes1[1, 0].set_ylabel("theta_x (rad)")
    axes1[1, 0].set_title("Orientation theta_x")
    axes1[1, 0].legend()
    axes1[1, 0].grid(True)

    axes1[1, 1].plot(time_collection, theta_z_1, label="Disk 1")
    axes1[1, 1].plot(time_collection, theta_z_2, label="Disk 2")
    axes1[1, 1].set_xlabel("Time (s)")
    axes1[1, 1].set_ylabel("theta_z (rad)")
    axes1[1, 1].set_title("Orientation theta_z")
    axes1[1, 1].legend()
    axes1[1, 1].grid(True)

    fig1.tight_layout()

    # --- Forces & Torques (all 6 components) ---
    force_labels = ["fx (N)", "fy (N)", "fz (N)", "tx (N.m)", "ty (N.m)", "tz (N.m)"]
    force_titles = ["Force X", "Force Y", "Force Z", "Torque X", "Torque Y", "Torque Z"]

    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle("Two-Disk Robot: Forces & Torques on Each Disk")

    for idx in range(6):
        row, col = divmod(idx, 3)
        ax = axes2[row, col]
        ax.plot(time_collection, force_robot_1[:, idx], label="Disk 1")
        ax.plot(time_collection, force_robot_2[:, idx], label="Disk 2")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(force_labels[idx])
        ax.set_title(force_titles[idx])
        ax.legend()
        ax.grid(True)

    fig2.tight_layout()
    plt.show()
