"""
Cantilever beam deflection case. This is a test case for the connected rigid robot model. We have two disks connected by a spring, and we apply a force on the second disk to see how the system behaves. The expected behavior is that the second disk will deflect due to the applied force, and the first disk will also move due to the connection between them. We can adjust the spring stiffness and damping coefficients to see how they affect the system's response.
Used for Benchmarking the connected rigid robot model, and for visualizing the dynamic response of a simple two-disk system under external force and spring connection. This case can also be used to validate the implementation of the spring forces and torques in the ConnectedRigidRobots3D class.
The analytical solution is provided by the Euler-Bernoulli beam theory for a cantilever beam with a point load at the free end. The deflection can be calculated using the formula:
delta = (F * L^3) / (3 * E * I)
where F is the applied force, L is the length of the beam, E is the Young
"""


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.RigidRobot3D import RigidRobot3D, ConnectedRigidRobots3D
from robot3d.Simulator3D import Simulator3D, MutiRobotSimulator3D
from robot3d.CableDrivenForce import CableDrivenForce, GravityForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SlenderRobotVisualization import animate_slender_robot

lie3 = SE3LieAlgebra()

def generate_series_robot_disks(
    n_disks, 
    length_between_disks,
    initial_position, 
    initial_orientation, 
    mass, 
    moment_inertia, 
    radius, 
    thickness, ):
    robot_disks = []
    for i in range(n_disks):
        robot_disk = RigidRobot3D(
            position = initial_position + np.array([0.0, 0.0, -(i+1) * length_between_disks]),
            orientation = initial_orientation,
            mass = mass,
            inertia = moment_inertia,
            linear_velocity = np.zeros(3),
            angular_velocity = np.zeros(3),
            radius = radius,
            thickness = thickness,
        )
        robot_disks.append(robot_disk)
    return robot_disks

def generate_series_connection_map(
    cantilever_beam,
    k_spring,
    damping_spring,
    k_tortional_spring,
    damping_tortional_spring,
    spring_original_length):

    cantilever_beam.add_connection(
            (0, 1),
            to_base = True,
            spring_stiffness = k_spring,
            spring_damping_coefficient = damping_spring,
            torque_spring_stiffness = k_tortional_spring,
            torque_spring_damping_coefficient = damping_tortional_spring,
            spring_original_length = spring_original_length,
        )

    for i in range(len(cantilever_beam.robots)-1):
        cantilever_beam.add_connection(
            (i, i + 1),
            to_base = False,
            spring_stiffness = k_spring,
            spring_damping_coefficient = damping_spring,
            torque_spring_stiffness = k_tortional_spring,
            torque_spring_damping_coefficient = damping_tortional_spring,
            spring_original_length = spring_original_length,
        )
    return 0


if __name__ == "__main__":

    n_elements = 2
    density = 1000  # kg/m^3
    total_length = 1.0 # meters
    disk_radius = 0.01 # meters
    E_module = 2.0 * 1e9 #Pa
    G_module = 7.7 * 1e8 #Pa

    segment_mass = density * np.pi * disk_radius**2 * (total_length / n_elements)
    segment_length = total_length / n_elements
    cross_section_area = np.pi * disk_radius**2
    I_x = np.pi * disk_radius**4 / 4
    I_y = np.pi * disk_radius**4 / 4
    I_z = np.pi * disk_radius**4 / 2
    k_spring = np.array([G_module * cross_section_area, 
                         G_module * cross_section_area,
                         E_module * cross_section_area]) / segment_length

    k_tortional_spring = np.array([
        E_module * I_x, E_module * I_y, G_module * I_z
    ]) / segment_length

    moment_inertia = np.array([I_x, I_y, I_z])* segment_mass / cross_section_area

    damping_spring = np.array([1.0, 1.0, 1.0])   
    damping_tortional_spring = np.array([2e-3, 2e-3, 2e-3])

    # Spirob_parameters
    n_elements = 7
    segment_length = 0.04

    k_spring = np.array([1.0,1.0,1.0])
    k_tortional_spring = np.array([0.01, 0.01, 0.01])

    damping_spring = np.array([1.0, 1.0, 1.0])   
    damping_tortional_spring = np.array([2e-3, 2e-3, 2e-3])
    segment_mass = 0.0855
    disk_radius = 0.03
    moment_inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5])

    
    robot_collection = generate_series_robot_disks(
        n_disks = n_elements,
        length_between_disks = segment_length,
        initial_position = np.array([0.0, 0.0, 0.0]),
        initial_orientation = np.array([0.0, 0.0, 0.0]),
        mass = segment_mass,
        moment_inertia = moment_inertia,
        radius = disk_radius,
        thickness = 0.025,
        )

    cantilever_beam = ConnectedRigidRobots3D(robots=robot_collection)

    

    generate_series_connection_map(
        cantilever_beam,
        k_spring,
        damping_spring,
        k_tortional_spring,
        damping_tortional_spring,
        spring_original_length = segment_length,
    )

    simulator_beam = MutiRobotSimulator3D(
        time_step=0.01,
        duration=20,
        stepper = 'explicit_euler',
        control_logic = None)

    simulator_beam.attach(cantilever_beam)


    for i in range(n_elements):
        simulator_beam.connected_robot.robots[i].control_input = np.array([0.0,0.005,0.0,0.0,0.0,0.0])
  
   

    while simulator_beam.run():
        simulator_beam.multi_robots_step()
        simulator_beam.multi_robot_record()


          # Data collection (place holder for now)
    #print(len(simulator_beam.time_collection))
    time_collection = np.array(simulator_beam.time_collection)

    posture_collection = np.array(simulator_beam.posture_collection)
    orientation_collection = np.array(simulator_beam.orientation_collection)
    force_collection = np.array(simulator_beam.force_collection)
    internal_force_collection = np.array(simulator_beam.internal_force_collection)
    #print("force_colleciton", force_collection)
    print(force_collection)

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

    # --- Forces & Torques (all 6 components) ---
    force_labels = ["fx (N)", "fy (N)", "fz (N)", "tx (N·m)", "ty (N·m)", "tz (N·m)"]
    force_titles = ["Force X", "Force Y", "Force Z", "Torque X", "Torque Y", "Torque Z"]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle("Spirob Slender Robot: Internal Forces & Torques on Each Disk (excl. control input)")

    for idx in range(6):
        row, col = divmod(idx, 3)
        ax = axes2[row, col]
        for i in range(N_disks):
            label = f"disk {i + 1}" if (show_all_labels or i in label_set) else None
            color = cmap(i / max(N_disks - 1, 1))
            ax.plot(time_collection, internal_force_collection[:, i, idx], color=color, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(force_labels[idx])
        ax.set_title(force_titles[idx])
        ax.legend(fontsize=8)
        ax.grid(True)

    fig2.tight_layout()
    plt.show()
    
    # ── 3-D animation ────────────────────────────────────────────────────────
    animate_slender_robot(
        time_collection   = time_collection,
        posture_collection= posture_collection,
        force_collection  = force_collection,
        disk_radius       = disk_radius,
        output_path       = 'slender_robot_simulation.mp4',  # falls back to .gif if ffmpeg missing
        fps               = 20,
        force_scale       = 0.5,
        skip_frames       = 5,
        view_yaw          = 0.0,   # degrees — rotate camera around world Z
        view_pitch        = 0.0,    # degrees — camera elevation above horizontal
        view_roll         = 0.0,     # degrees — roll around the line of sight
    )



    


