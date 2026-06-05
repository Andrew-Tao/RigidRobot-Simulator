import sys, os
from time import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from robot3d.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.RigidRobot3D import RigidRobot3D
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


def analytical_temoshenko_solution(x, F, E, I, G, A, L):

    sigma = -F / ( A * (4/3) * G)
    kappa = -F *L / (E * I * 2)
    gamma = F / (6 * E * I) 

    y = sigma * x + kappa * x**2 + gamma * x**3
    return y


if __name__ == "__main__":

    # -------------------- Initialization of the cantilever beam system --------------

    force = 15 # N total load
    radius = 0.25  # m
    n_elements = 25
    E_module = 1e6  # Pa
    G_module = 1e4  #Pa
    total_length = 3.0  # m

    I_x = np.pi * (radius**4) / 4  # m^4, moment of inertia for bending about x-axis
    I_y = np.pi * (radius**4) / 4  # m^4, moment of inertia for bending about y-axis
    I_z = I_x + I_y  # m^4, polar moment of inertia for a circular cross-section

    density = 5000  # kg/m^3
    time_step = 0.00225  # s
    duration =30# s

    damping_spring = np.array([1.0, 1.0, 1.0])  * 4000
    damping_tortional_spring = np.array([1.0, 1.0, 1.0]) * 0.035

    persistence_time = 2000 # s, time duration for which the load is applied
    ramp_up_time = 5  # s, time duration for ramping up the load

    # ---------------------------------------- End ---------------------------------


    total_volume = np.pi * (radius**2) * total_length  # m^3, volume of the beam
    total_mass = density * total_volume  # kg

    segment_mass = total_mass / n_elements
    segment_length = total_length / n_elements
    cross_section_area = np.pi * (radius**2)


    k_spring = np.array([G_module * (4/3) * cross_section_area, 
                            G_module * (4/3) * cross_section_area,
                            E_module * cross_section_area]) 

    k_tortional_spring = np.array([
        E_module * I_x, E_module * I_y, G_module * I_z
    ]) 

    I_x_mass = (1/12) * segment_mass * (3 * radius**2 + segment_length**2)
    I_y_mass = (1/12) * segment_mass * (3 * radius**2 + segment_length**2)
    I_z_mass = (1/2) * segment_mass * radius**2

    moment_inertia = np.array([I_x_mass, I_y_mass, I_z_mass])

    robot_collection = generate_series_robot_disks(
        n_disks = n_elements,
        length_between_disks = segment_length,
        initial_position = np.array([0.0, 0.0, 0.0]),
        initial_orientation = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, -1.0]]),
        mass = segment_mass,
        moment_inertia = moment_inertia,
        radius = radius,
        thickness = 0.025,
        )

    timoshenko_beam = ConnectedRigidRobots3D(robots=robot_collection)

    generate_series_connection_map(
        timoshenko_beam,
        k_spring,
        damping_spring,
        k_tortional_spring,
        damping_tortional_spring,
        spring_original_length = segment_length,
    )

    simulator_beam = MutiRobotSimulator3D(
        time_step=time_step,
        duration=duration,
        stepper = 'explicit_euler',
        control_logic = None)

    simulator_beam.attach(timoshenko_beam)


    while simulator_beam.run():

        t = simulator_beam.current_time
        if t <= ramp_up_time:
            current_load = force * (t / ramp_up_time)
        elif t < persistence_time:
            current_load = force
        else:
            current_load = 0.0
      
        simulator_beam.connected_robot.robots[-1].control_input = np.array([0.0, force, 0.0, 0.0, 0.0,0.0])

        simulator_beam.multi_robots_step()
        simulator_beam.multi_robot_record()


            # Data collection (place holder for now)
    #print(len(simulator_beam.time_collection))
    time_collection = np.array(simulator_beam.time_collection)

    posture_collection = np.array(simulator_beam.posture_collection)
    orientation_collection = np.array(simulator_beam.orientation_collection)
    force_collection = np.array(simulator_beam.force_collection)
    bending_internal_couple_collection = np.array(simulator_beam.bending_internal_couple_collection)
    shear_internal_couple_collection = np.array(simulator_beam.shear_internal_couple_collection)
    tau_x_base_collection = np.array(simulator_beam.tau_x_base_collection)
    strain_local_collection = np.array(simulator_beam.strain_local_collection)

    y_tip_collection = posture_collection[:, -1, 1, 3]  # Extract the y-position of the tip disk over time

    x_position_collection = posture_collection[-1, :, 2, 3] # The x direction is actually the z direction
    y_position_collection = posture_collection[-1, :, 1, 3]  # Extract the y-position of the tip disk over time

    y_position_analytical = analytical_temoshenko_solution(-x_position_collection, force, E_module, I_x, G_module, A=cross_section_area, L=total_length)




    #plt.plot(time_collection, bending_internal_couple_collection[:,0,0], label="Bending Internal Couple")
    #plt.plot(time_collection, shear_internal_couple_collection[:,0,0], label="Shear Internal Couple")
    #plt.plot(time_collection, tau_x_base_collection[:,0], label="Tau_x Base")
    #plt.plot(time_collection, force_collection[:,0,3], label="Total Tau 0")
    #plt.plot(time_collection, strain_local_collection[:,0,1], label="Total Tau 1")
    plt.figure()
    plt.plot(-x_position_collection, -y_position_collection, label = "simulation result")

    plt.plot(-x_position_collection, y_position_analytical, label = "analytical solution")

    plt.xlabel("X-Position (m)")
    plt.ylabel("Y-Position (m)")
    plt.title("Timoshenko Beam Configuration")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------- Plot 2 ----------------
    plt.figure()
    plt.plot(time_collection, -y_tip_collection, label="Tip Y-Position (Simulation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Y-Position (m)")
    plt.title("Plot 2: Tip Y-Position Over Time")
    plt.grid(True)
    plt.show()


    #print("force_colleciton", force_collection)
    #print(force_collection)

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
            ax.plot(time_collection, force_collection[:, i, idx], color=color, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(force_labels[idx])
        ax.set_title(force_titles[idx])
        ax.legend(fontsize=8)
        ax.grid(True)

    fig2.tight_layout()
    plt.show()


