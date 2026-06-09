"""
Cantilever beam deflection case. This is a test case for the connected rigid robot model. We have two disks connected by a spring, and we apply a force on the second disk to see how the system behaves. The expected behavior is that the second disk will deflect due to the applied force, and the first disk will also move due to the connection between them. We can adjust the spring stiffness and damping coefficients to see how they affect the system's response.
Used for Benchmarking the connected rigid robot model, and for visualizing the dynamic response of a simple two-disk system under external force and spring connection. This case can also be used to validate the implementation of the spring forces and torques in the ConnectedRigidRobots3D class.
The analytical solution is provided by the Euler-Bernoulli beam theory for a cantilever beam with a point load at the free end. The deflection can be calculated using the formula:
delta = (F * L^3) / (3 * E * I)
where F is the applied force, L is the length of the beam, E is the Young

Benchmarking with Pyelasics example Cantilever beam under nonconservative load of 20 N 
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from robot3d.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.RigidRobot3D import RigidRobot3D
from robot3d.CableDrivenForce import CableDrivenForce, GravityForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra, log_SO3
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SlenderRobotVisualization import animate_slender_robot
from BeamGenerator import generate_series_robot_disks, generate_series_connection_map

lie3 = SE3LieAlgebra()


def max_bending_angle_at_frame(posture_frame: np.ndarray) -> float:
    """
    posture_frame: (n_disks, 4, 4) SE3 postures at one time step.
    Returns the maximum bending angle (radians) between any two consecutive disks,
    computed as ||log_SO3(R_i^T @ R_{i+1})||.
    """
    max_angle = 0.0
    for i in range(len(posture_frame) - 1):
        R_i   = posture_frame[i,   :3, :3]
        R_ip1 = posture_frame[i+1, :3, :3]
        angle = np.linalg.norm(log_SO3(R_i.T @ R_ip1))
        if angle > max_angle:
            max_angle = angle
    return max_angle


if __name__ == "__main__":

    F = 3 # N total load
    persistence_time = 200 # s, time duration for which the load is applied
    width = 0.01  # m
    
# -------------------- Initialization of the cantilever beam system --------------

    F = 20 # N total load

    persistence_time = 200 # s, time duration for which the load is applied
    width = 0.01  # m
    base_area = width * width  # m^2
    
    n_elements = 25
    load = F / (n_elements * 20) # Why / 20 ? TODO: Why the Pyelasica mutipley load by np.mass[i]
    E_module = 1.2 * 1e7  # Pa
    poisson_ratio = 0 
    G_module = E_module / (2 * (1 + poisson_ratio)) # Pa
    total_length = 0.5  # m
    I_x = 0.01**4 / 12  # m^4, moment of inertia for a circular cross-section
    I_y = 0.01**4 / 12  # m^4
    I_z = I_x + I_y  # m^4, polar moment of inertia for a circular cross-section

    density = 1000  # kg/m^3
    time_step = 0.00016 # s
    duration = 0.0025 # s

    print("I",I_x)
    print(load)
    print("density",density)
    print("base_area",base_area)
    print("width",width)

    damping_spring = np.array([1.0, 1.0, 1.0])  * 0
    damping_tortional_spring = np.array([1.0, 1.0, 1.0]) * 0
    S_modifier = 1.0
    ramp_up_time = 0.001  # s, time duration for ramping up the load
    stepper_type = 'position_verlet'  # 'euler', 'velocity_verlet', or 'position_verlet'

# ---------------------------------------- End ---------------------------------

    total_volume = 0.01 **2 * total_length  # m^3, volume of the beam
    total_mass = density * total_volume  # kg

    segment_mass = total_mass / n_elements
    segment_length = total_length / n_elements
    cross_section_area = 0.01 **2  # m^2, cross-sectional area of the beam


    k_spring = np.array([G_module * (4/3) * cross_section_area, 
                         G_module * (4/3) * cross_section_area,
                         E_module * cross_section_area]) * S_modifier

    k_tortional_spring = np.array([
        E_module * I_x, E_module * I_y, G_module * I_z
    ]) 

    I_h = (1/12) * segment_mass * ((0.01)**2 + (0.01)**2)
    I_w = (1/12) * segment_mass * ((0.01)**2 + segment_length**2)
    I_d = (1/12) * segment_mass * ((0.01)**2 + segment_length**2)


    moment_inertia = np.array([I_w, I_d, I_h]) 


    robot_collection = generate_series_robot_disks(
        n_disks = n_elements,
        length_between_disks = segment_length,
        initial_position = np.array([0.0, 0.0, 0.0]),
        initial_orientation = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, -1.0]]),
        mass = segment_mass,
        moment_inertia = moment_inertia,
        radius = width,
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
        time_step=time_step,
        duration=duration,
        stepper = stepper_type,
        control_logic = None)

    simulator_beam.attach(cantilever_beam)


    while simulator_beam.run():

        t = simulator_beam.current_time
        if t <= ramp_up_time:
            current_load = load * (t / ramp_up_time)
        elif t < persistence_time:
            current_load = load
        else:
            current_load = 0.0

        for i in range(n_elements):
            simulator_beam.connected_robot.robots[i].control_input = np.array([0.0, current_load, 0.0, 0.0, 0.0, 0.0])

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
    damping_couple_collection = np.array(simulator_beam.damping_couple_collection)


    y_tip_collection = posture_collection[:, -1, 1, 3]  # Extract the y-position of the tip disk over time

    x_position_collection = posture_collection[-1, :, 1, 3] # The x direction is actually the z direction
    y_position_collection = posture_collection[-1, :, 2, 3]  # Extract the y-position of the tip disk over time


    analytical_position = np.load(os.path.join(os.path.dirname(__file__), "position_collection.npy"))
    plt.figure()
    plt.plot(x_position_collection, y_position_collection + (total_length / n_elements), label = "simulation result")
    plt.plot(analytical_position[0], analytical_position[1], label = "analytical solution")
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




    plt.plot(time_collection, bending_internal_couple_collection[:,0,0], label="Bending Internal Couple")
    plt.plot(time_collection, shear_internal_couple_collection[:,0,0], label="Shear Internal Couple")
    plt.plot(time_collection, tau_x_base_collection[:,0], label="Tau_x Base")
    plt.plot(time_collection, force_collection[:,0,3], label="Total Tau 0")
    plt.plot(time_collection, strain_local_collection[:,0,1], label="strain_local_collection")
    plt.plot(time_collection, damping_couple_collection[:,0], label="Damping Couple 0")
    plt.xlabel("Time (s)")
    plt.ylabel("Internal Couple (N·m)")
    plt.title("Internal Couples on the First Disk")
    plt.legend()
    plt.grid()
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

    # ── Max bending angle over time ──────────────────────────────────────────
    max_bending_collection = np.array([
        max_bending_angle_at_frame(posture_collection[t]) for t in range(len(time_collection))
    ])

    plt.figure()
    plt.plot(time_collection, np.degrees(max_bending_collection))
    plt.xlabel("Time (s)")
    plt.ylabel("Max bending angle (deg)")
    plt.title("Maximum inter-disk bending angle over time")
    plt.grid(True)
    plt.show()


    # ── 3-D animation ────────────────────────────────────────────────────────

    animate_slender_robot(
        time_collection   = time_collection,
        posture_collection= posture_collection,
        force_collection  = None,
        disk_radius       = width * 5,
        output_path       = 'slender_robot_simulation.mp4',  # falls back to .gif if ffmpeg missing
        fps               = 20,
        force_scale       = 0.5,
        skip_frames       = 1,
        view_yaw          = 0.0,   # degrees — rotate camera around world Z
        view_pitch        = 0.0,    # degrees — camera elevation above horizontal
        view_roll         = 0.0,     # degrees — roll around the line of sight
    )

    



    


