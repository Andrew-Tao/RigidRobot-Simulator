"""
Cantilever beam deflection case. This is a test case for the connected rigid robot model. We have two disks connected by a spring, and we apply a force on the second disk to see how the system behaves. The expected behavior is that the second disk will deflect due to the applied force, and the first disk will also move due to the connection between them. We can adjust the spring stiffness and damping coefficients to see how they affect the system's response.
Used for Benchmarking the connected rigid robot model, and for visualizing the dynamic response of a simple two-disk system under external force and spring connection. This case can also be used to validate the implementation of the spring forces and torques in the ConnectedRigidRobots3D class.
The analytical solution is provided by the Euler-Bernoulli beam theory for a cantilever beam with a point load at the free end. The deflection can be calculated using the formula:
delta = (F * L^3) / (3 * E * I)
where F is the applied force, L is the length of the beam, E is the Young
"""


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.force.CableDrivenForce import CableDrivenForce, GravityForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SlenderRobotVisualization import animate_slender_robot
from BeamGenerator import generate_series_connection_map, generate_series_robot_disks, generate_series_robot_disks_tapper
from robot3d.contact.DiskSelfContact import SelfAngleContact

lie3 = SE3LieAlgebra()


if __name__ == "__main__":

    # Spirob_parameters
    n_elements = 25
    L = 0.35 
    disk_radius = 0.03
    disk_thickness = 0.01

    density_TPU = 1.18e3
    area_disk = np.pi * (disk_radius**2)
    disk_mass = density_TPU * (area_disk * disk_thickness)
   
    segment_length = 0.35 / n_elements

    area_spine = 0.4 * 1e-4
    r_spine = (area_spine / np.pi)**0.5
    alpha_c = 4/3 # For cylinder cross section
    
    Ixx = (1/12) * disk_mass * (3 * disk_radius **2 + disk_thickness **2)
    Iyy = Ixx
    Izz = (1/2) * disk_mass * disk_radius**2


    Ix = np.pi * r_spine**4 / 4
    Iy = Ix
    Ixy = np.pi* r_spine**4 / 4

    moment_inertia = np.array([Ixx, Iyy, Izz])
    print("moment_inertia", moment_inertia)
    print("disk_mass", disk_mass)

    E_module = 50 * 1e6  #Pa
    G_module = 60 * 1e6  #Pa

    # TODO: k_s and k_t should vary across the arm

    k_s = np.array([alpha_c * G_module,alpha_c * G_module, E_module]) * area_spine 
    k_t = np.array([E_module * Ix * 1.5 , E_module*Ix * 1.5, G_module * Ixy]) 

    spring_damp = np.array([1.5, 1.5, 1.6]) * 27
    tor_spring_damp = np.array([7e-5, 7e-5, 2e-4]) * 9

    print("k_s",k_s)
    print("k_t",k_t)

    
    robot_collection = generate_series_robot_disks_tapper(
        n_disks = n_elements,
        base_length_between_disks = 0.02,
        initial_position = np.array([0.0, 0.0, 0.0]),
        initial_orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
        base_mass = disk_mass,
        base_moment_inertia = moment_inertia,
        base_radius = 0.035 * 0.5,
        tapper_angle= 0.025
        )
    


    cantilever_beam = ConnectedRigidRobots3D(robots=robot_collection)

    generate_series_connection_map(
        cantilever_beam,
        k_s,
        spring_damp,
        k_t,
        tor_spring_damp,
        # spring_original_length omitted → auto-measured from initial robot positions
    )

    simulator_beam = MutiRobotSimulator3D(
        time_step=0.0001,
        duration=1.0,
        stepper = 'position_verlet',
        control_logic = None)

    simulator_beam.attach(cantilever_beam)

    gravity = GravityForce()

    # Cable hole directions on the unit circle at 0°, 120°, 240°
    cable_directions = np.array([
        [ 1.0,                 0.0,  0.0],
        [-0.5,  np.sqrt(3)/2,        0.0],
        [-0.5, -np.sqrt(3)/2,        0.0],
    ])  # (C, 3)
    cable_fraction = 0.8  # holes at 80% of each disk's own radius
    disk_radii_arr = np.array([robot.radius for robot in robot_collection])  # (N,)
    # Per-disk hole offsets: (N, C, 3) — each disk scaled by its own radius
    hole_offset = disk_radii_arr[:, np.newaxis, np.newaxis] * cable_fraction * cable_directions[np.newaxis, :, :]

    def cable_control(time):
        if time < 0.5:
            return np.array([40.0, 0.0, 0.0])  # cable 1 active at 0.1 N, cables 2 & 3 slack
        else:
            return np.array([40.0, 40.0, 0.0])

    cable_force = CableDrivenForce(control_input=cable_control, hole_offset=hole_offset)
    simulator_beam.add_external_force(gravity)
    simulator_beam.add_external_force(cable_force)

    contact = SelfAngleContact()



    while simulator_beam.run():

        # ------------------- Apply Contact -------------------

        contact_tor_collection = contact.compute_force_collection(cantilever_beam) 
        #print("contact_tor_colleciton", contact_tor_collection.shape)
        for i in range(n_elements):
            tor_ct_x = contact_tor_collection[i,0]
            tor_ct_y = contact_tor_collection[i,1]
            simulator_beam.connected_robot.robots[i].control_input = np.array([0.0,0.0,0.0,tor_ct_x,tor_ct_y,0.0])
        
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
    
    # Per-disk radii from the tapered robot (each disk stores its own radius)
    disk_radii = np.array([robot.radius for robot in robot_collection])

    # ── 3-D animation ────────────────────────────────────────────────────────
    animate_slender_robot(
        time_collection   = time_collection,
        posture_collection= posture_collection,
        force_collection  = None,
        disk_radius       = disk_radii,
        output_path       = 'slender_robot_simulation.mp4',  # falls back to .gif if ffmpeg missing
        fps               = 20,
        force_scale       = 0.5,
        skip_frames       = 5,
        view_yaw          = 0.0,   # degrees — rotate camera around world Z
        view_pitch        = 0.0,    # degrees — camera elevation above horizontal
        view_roll         = 0.0,     # degrees — roll around the line of sight
    )






