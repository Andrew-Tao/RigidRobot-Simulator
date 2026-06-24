import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from robot3d.methods3D import SE3LieAlgebra, rpy_to_Q, log_SO3
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from BeamGenerator import generate_series_connection_map
from examples.postprocessing.SlenderRobotVisualization import animate_slender_robot
import numpy as np
import matplotlib.pyplot as plt
from DiskSelfContact import SelfContact



lie3 = SE3LieAlgebra()

def initialize_robot(segment_length):
     # position
    robot1_location = np.array([0.0, 0.0, -segment_length])
    robot2_location = robot1_location.copy(); robot2_location[2] -= segment_length

    # orientation
    intial_orientation = np.eye(3)
    intial_orientation[2,2] = - 1.0

    # properties 
    mass = 0.0333
    inertia  = np.array([7.78486660e-06, 7.78486660e-06, 1.50136713e-05])
    radius = 0.03
    thickness = 0.01

    robot1 = RigidRobot3D(
        robot1_location,
        intial_orientation,
        mass = mass, 
        inertia = inertia, 
        linear_velocity=np.zeros(3), 
        angular_velocity= np.zeros(3),
        radius = radius,
        thickness = thickness, 
    )

    robot2 = RigidRobot3D(
        robot2_location,
        intial_orientation,
        mass = mass, 
        inertia = inertia, 
        linear_velocity=np.zeros(3), 
        angular_velocity= np.zeros(3),
        radius = radius,
        thickness = thickness, 
    )
    return robot1, robot2

def compute_rotation(base_orientaiton, Q_matrix):
    # Input two SO3 orientatin matirx
    # Output (3) rotation vector
    #print("Q_matrix",Q_matrix)
    trans = base_orientaiton.T @ Q_matrix
    #print("trans",trans)
    skew = log_SO3(trans)
    #print("skew",skew)
    
    
    return skew

def compute_rotation_from_director_colleciton(base_orientaiton,director_colleciton):
    n_frame = director_colleciton.shape[0]
    n_robots = director_colleciton.shape[1]
    
    result = np.zeros((n_frame, n_robots,3))
    #print("result", result.shape)
    for i in range(n_frame):
        for j in range(n_robots):
            #print("shape",director_colleciton[i,j].shape)
            vector = compute_rotation(base_orientaiton,director_colleciton[i,j])

            result[i,j,:] = vector 
    return result


if __name__ == "__main__":

    # Spirob_parameters
    _params_path = os.path.join(os.path.dirname(__file__), "../../spirob/spirob_params.json")
    with open(_params_path) as f:
        p = json.load(f)

    n_elements      = p["n_elements"]
    L               = p["L"]
    disk_radius     = p["disk_radius"]
    disk_thickness  = p["disk_thickness"]
    density_TPU     = p["density_TPU"]
    area_spine      = p["area_spine"]
    alpha_c         = p["alpha_c"]
    E_module        = p["E_module"]
    G_module        = p["G_module"]
    area_disk       = p["area_disk"]
    disk_mass       = p["disk_mass"]
    segment_length  = p["segment_length"]
    r_spine         = p["r_spine"]
    moment_inertia  = np.array(p["moment_inertia"])
    k_s             = np.array(p["k_s"])
    k_t             = np.array(p["k_t"])
    spring_damp     = np.array(p["spring_damp"])
    tor_spring_damp = np.array(p["tor_spring_damp"])

    print("moment_inertia", moment_inertia)
    print("disk_mass", disk_mass)


    robot1, robot2 = initialize_robot(segment_length = segment_length)

    robot_collection = [robot1]

    robots_system = ConnectedRigidRobots3D(robots=robot_collection)

    simulator = MutiRobotSimulator3D(
        time_step= 0.001, 
        duration=5,
        stepper='position_verlet',
        control_logic= None
    )

    generate_series_connection_map(
        robots_system, 
        k_s,
        spring_damp, 
        k_t, 
        tor_spring_damp,
        spring_original_length= segment_length
    )

    simulator.attach(robots_system)

    # -----------------------Contact_Force--------------------------
    contact_collection = []
    contact = SelfContact()
    def compute_contact(slender_robot):
        contact_torque = contact.compute_force_collection(slender_robot= slender_robot)
        contact_collection.append(contact_torque)
        return contact_torque
   

    #print(simulator.connected_robot)
    # --- Torque control as functions of time (edit these) ---
    def torque_x(t):
        if t<2: 
            return 0.02         # N·m
        else: return 0

    def torque_y(t):
        return 0.0  # 90° out of phase

    while simulator.run():
        t = simulator.current_time
        contact_tor = compute_contact(slender_robot= robots_system) * 1
        #print(contact_tor)
        robots_system.robots[0].control_input = np.array([0.0, 0.0, 0.0, torque_x(t) + contact_tor[0], torque_y(t)+ contact_tor[1], 0.0]) 
        simulator.multi_robots_step()
        simulator.multi_robot_record()
        



    time_collection        = np.array(simulator.time_collection)
    posture_collection     = np.array(simulator.posture_collection)
    orientation_collection = np.array(simulator.orientation_collection)
    force_collection       = np.array(simulator.force_collection)
    director_collection = posture_collection[:,:,:3,:3]

    #print(director_collection)
    base_orientation = np.eye(3); base_orientation[2,2] = - 1.0
    #print("base",base_orientation)
    rotation_collection = compute_rotation_from_director_colleciton(base_orientation, director_collection)
    #print(rotation_collection)

    rotate_x_degree = rotation_collection[:,0,0] / np.pi * 180
    #print(rotate_x_degree)
    rotate_y_degree = rotation_collection[:,0,1] / np.pi * 180

    # --- Torque vs time ---
    torque_x_values = np.array([torque_x(t) for t in time_collection])
    torque_y_values = np.array([torque_y(t) for t in time_collection])

    contact_collection = np.array(contact_collection) 

    limit_line = np.zeros_like(time_collection) + 30
    print("Steady_Degree", rotate_x_degree[-1])


    plt.figure()
    plt.plot(time_collection, rotate_x_degree, label = "rx")
    plt.plot(time_collection, rotate_y_degree, label = "ry")
    plt.plot(time_collection, contact_collection[:,0], label = "contact_x")
    plt.plot(time_collection, contact_collection[:,1], label = "contact_y")
    plt.plot(time_collection, limit_line, label = "30 degree")
    plt.legend()
    plt.title("Angle and contact")
    plt.show()


    fig_t, ax_t = plt.subplots(figsize=(8, 4))
    ax_t.plot(time_collection, torque_x_values, label="torque_x")
    ax_t.plot(time_collection, torque_y_values, label="torque_y")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Torque (N·m)")
    ax_t.set_title("Applied Torque vs Time")
    ax_t.legend()
    ax_t.grid(True)
    plt.tight_layout()
    plt.show()

    plt.scatter(rotate_x_degree,rotate_y_degree)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rotation")
    plt.show()

   

    
    """

    # --- Position & orientation of each disk ---
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle("Two-Disk Contact Simulation")

    for i, label in enumerate(["disk 1"]):
        pos_i = posture_collection[:, i, :3, 3]
        axes[0, 0].plot(time_collection, pos_i[:, 0], label=label)
        axes[0, 1].plot(time_collection, pos_i[:, 2], label=label)
        axes[1, 0].plot(time_collection, orientation_collection[:, i, 0], label=label)
        axes[1, 1].plot(time_collection, orientation_collection[:, i, 2], label=label)
        axes[1, 2].plot(time_collection, orientation_collection[:, i, 1], label=label)


    for ax, xlabel, ylabel, title in [
        (axes[0, 0], "Time (s)", "Position X (m)",  "Position X"),
        (axes[0, 1], "Time (s)", "Position Z (m)",  "Position Z"),
        (axes[1, 0], "Time (s)", "Theta X (rad)",   "Orientation Theta X"),
        (axes[1, 1], "Time (s)", "Theta Z (rad)",   "Orientation Theta Z"),
        (axes[1, 2], "Time (s)", "Theta Y (rad)",   "Orientation Theta Y")
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    """



    animate_slender_robot(
        time_collection    = time_collection,
        posture_collection = posture_collection,
        force_collection   = force_collection,
        disk_radius        = disk_radius,
        output_path        = 'contact_simulation.mp4',
        fps                = 20,
        force_scale        = 0.5,
        skip_frames        = 5,
        view_yaw           = 0.0,
        view_pitch         = 0.0,
        view_roll          = 0.0,
    )






