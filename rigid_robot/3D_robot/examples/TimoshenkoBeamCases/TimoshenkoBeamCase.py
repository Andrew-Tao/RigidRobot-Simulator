import sys, os
from examples.postprocessing.PlotResults import plot_disk_chain_results
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.methods3D import SE3LieAlgebra
from examples.preprocessing.BeamGenerator import generate_series_robot_disks, generate_series_connection_map

lie3 = SE3LieAlgebra()

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


    plot_disk_chain_results(time_collection, posture_collection, orientation_collection, force_collection)


