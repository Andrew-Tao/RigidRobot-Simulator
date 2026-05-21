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
from robot3d.CableDrivenForce import CableDrivenForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SlenderRobotVisualization import animate_slender_robot


def rotate_xy(v, degree):
    theta = np.deg2rad(degree)  # convert degree to radians

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    return Rz @ v


lie3 = SE3LieAlgebra()

if __name__ == "__main__":
    robot_radius = 0.03

    robot_disk_1 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_2 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.08]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_3 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.12]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_4 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04 * 4]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_5 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04 * 5]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_6 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04 * 6]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_disk_7 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04 * 7]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )

    robot_collection = [robot_disk_1,robot_disk_2,robot_disk_3,robot_disk_4,robot_disk_5,robot_disk_6,robot_disk_7]

    slender_robot = ConnectedRigidRobots3D(
        robots= robot_collection
    )

    # Add connection map for the robot
    k_s = 10.0
    k_t = 0.04

    slender_robot.add_connection((0,1), to_base= True, spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((0,1), to_base= False, spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((1,2), to_base= False,spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((2,3), to_base= False,spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((3,4), to_base= False,spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((4,5), to_base= False,spring_stiffness= k_s, torque_spring_stiffness= k_t)
    slender_robot.add_connection((5,6), to_base= False,spring_stiffness= k_s, torque_spring_stiffness= k_t)


    # Add external force - CableDrivenForce

    

    def control_logic(time: float):
        cable_driven_force = np.array([0.0, 0.0, 0.09]) if time < 20.0 else np.zeros(3)
        return cable_driven_force
    
    base_hole_offset = np.array([0.8 * robot_radius, 0.0, 0.0])
    hole_arrangment = np.array([rotate_xy(base_hole_offset, 120 * n) for n in range(3)])

    cable_driven_force = CableDrivenForce(
        control_input= control_logic,
        hole_offset= hole_arrangment
    )

    simulator_slender = MutiRobotSimulator3D(
        time_step = 0.01,
        duration = 20,
        stepper= 'explicit_euler',
        control_logic = None, 
    )

    simulator_slender.attach(slender_robot)
    simulator_slender.add_external_force(cable_driven_force)

    #simulator_slender.connected_robot.robots[1].control_input = np.array([0.1,0.0,0.0,0.0,0.0,0.0])

    while simulator_slender.run():
        if simulator_slender.current_time >= 0.5:
            simulator_slender.connected_robot.robots[1].control_input = np.zeros(6)
        simulator_slender.multi_robots_step()
        simulator_slender.multi_robot_record()

    # Data collection (place holder for now)
    #print(len(simulator_slender.time_collection))
    time_collection = np.array(simulator_slender.time_collection)
    
    posture_collection = np.array(simulator_slender.posture_collection)
    orientation_collection = np.array(simulator_slender.orientation_collection)
    force_collection = np.array(simulator_slender.force_collection)
    #print("force_colleciton", force_collection)

    N_disks = posture_collection.shape[1]

    # Use a colourmap so the plots work for any number of disks.
    # Show a full legend when N is small; only show first/middle/last when large.
    cmap = plt.cm.get_cmap('viridis', N_disks)
    show_all_labels = N_disks <= 10
    label_set = {0, N_disks // 2, N_disks - 1}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Spirob Slender Robot Simulation")

    for i in range(N_disks):
        label = f"disk {i + 1}" if (show_all_labels or i in label_set) else None
        color = cmap(i / max(N_disks - 1, 1))
        pos_i = posture_collection[:, i, :3, 3]            # (T, 3)
        axes[0, 0].plot(time_collection, pos_i[:, 0],                    color=color, label=label)
        axes[0, 1].plot(time_collection, pos_i[:, 2],                    color=color, label=label)
        axes[1, 0].plot(time_collection, orientation_collection[:, i, 0], color=color, label=label)
        axes[1, 1].plot(time_collection, orientation_collection[:, i, 2], color=color, label=label)

    plot_cfg = [
        (axes[0, 0], "Time (s)", "Position X (m)",  "Position X"),
        (axes[0, 1], "Time (s)", "Position Z (m)",  "Position Z"),
        (axes[1, 0], "Time (s)", "Theta X (rad)",   "Orientation Theta X"),
        (axes[1, 1], "Time (s)", "Theta Z (rad)",   "Orientation Theta Z"),
    ]
    for ax, xlabel, ylabel, title in plot_cfg:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # ── 3-D animation ────────────────────────────────────────────────────────
    animate_slender_robot(
        time_collection   = time_collection,
        posture_collection= posture_collection,
        force_collection  = force_collection,
        hole_offset       = hole_arrangment,
        disk_radius       = robot_radius,
        output_path       = 'slender_robot_simulation.mp4',  # falls back to .gif if ffmpeg missing
        fps               = 20,
        force_scale       = 0.5,
        skip_frames       = 5,
        view_yaw          = -60.0,   # degrees — rotate camera around world Z
        view_pitch        = 30.0,    # degrees — camera elevation above horizontal
        view_roll         = 0.0,     # degrees — roll around the line of sight
    )


