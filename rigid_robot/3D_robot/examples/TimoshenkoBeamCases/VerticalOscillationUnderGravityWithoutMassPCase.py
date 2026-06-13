"""
Vetical oscillation of a cantilever beam under gravity. 
The beam is modelled as a series of connected rigid disks, with the first disk fixed in space and the rest free to move.
A downward force is applied to each disk to simulate the effect of gravity, 
and the resulting oscillations are observed and compared to analytical solutions for validation.
"""

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

def period_detect(time_series, signal_series):
    """
    Detect the period of oscillation from the time series and signal series.
    This is a simple implementation that detects peaks in the signal and calculates the average time between peaks.
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(signal_series)
    if len(peaks) < 2:
        return None  # Not enough peaks to determine period
    peak_times = time_series[peaks]
    periods = np.diff(peak_times)
    average_period = np.mean(periods)
    return average_period

def amplitude_detect(time_series, signal_series):
    """
    Detect the amplitude of oscillation as half the peak-to-peak distance:
    average of the maxima minus average of the minima, divided by two.
    This is robust to a constant offset (e.g. a static equilibrium deflection).
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(signal_series)
    troughs, _ = find_peaks(-signal_series)
    if len(peaks) == 0 or len(troughs) == 0:
        return None  # No peaks/troughs detected

    peak_to_peak = np.mean(signal_series[peaks]) - np.mean(signal_series[troughs])
    return peak_to_peak / 2

if __name__ == "__main__":

# -------------------- Initialization of the cantilever beam system --------------

    n_elements = 10

    mass_p = 0 # kg 
    mass_rod = 1 # kg
    E_module = 1e9  # Pa
    density = 1000  # kg/m^3

    G_module = (2/3) * E_module  # Pa
    total_length = 1.0  # m
    cross_section_area = mass_rod / (total_length * density)  # m^2, cross-sectional area of the beam

    radius = np.sqrt(cross_section_area / np.pi)  # m, radius of the circular cross-section
    print(f"Calculated radius of the beam's circular cross-section: {radius:.4f} m")
    print(f"Aspect ratio (length/radius): {total_length/ n_elements / radius:.2f}")
    I_x = (1/4) * np.pi * radius**4  # m^4, moment of inertia for a circular cross-section
    I_y = I_x  # m^4
    I_z = I_x + I_y  # m^4, polar moment of inertia


    persistence_time = 200 # s, time duration for which the load is applied
    g = 9.81 # m/s^2

    k = (E_module * cross_section_area / total_length)

    analytical_period = 2 * np.pi * np.sqrt((mass_p + mass_rod/3)/k)
    analytical_amplitude = g * (mass_p + mass_rod/2) / k

    print(f"Analytical period of oscillation: {analytical_period:.4f} s")
    print(f"Analytical amplitude of oscillation: {analytical_amplitude:.4f} m")

    time_step = 0.0001  # s
    #time_step = analytical_period / 1e6
    duration = 2 # s


    damping_spring = np.array([1.0, 1.0, 1.0])  * 1.5 * 0 
    damping_tortional_spring = np.array([1.0, 1.0, 1.0]) * 0.04 * 0

    ramp_up_time = 0.001  # s, time duration for ramping up the load
    stepper_type = "explicit_euler"  # "Euler" or "RK4"
    stepper_type = "position_verlet"  # "Euler", "RK4", or "position_verlet"


# ---------------------------------------- End ---------------------------------

    total_volume = cross_section_area * total_length  # m^3, volume of the beam
    total_mass = density * total_volume  # kg

    segment_mass = total_mass / n_elements
    segment_length = total_length / n_elements
    load = segment_mass * g  # N, load applied to each disk to simulate gravity
    load_p = mass_p * g



    k_spring = np.array([G_module * (4/3) * cross_section_area, 
                         G_module * (4/3) * cross_section_area,
                         E_module * cross_section_area])

    k_tortional_spring = np.array([
        E_module * I_x, E_module * I_y, G_module * I_z
    ]) 

    I_w = (1/12) * segment_mass * ( 3 * radius**2 + segment_length**2 )
    I_d = (1/12) * segment_mass * ( 3 * radius**2 + segment_length**2 )
    I_h = (1/2) * segment_mass * radius**2


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
        radius = radius,
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
            if i == n_elements - 1:  # Apply the load to the last disk (tip of the beam)
                simulator_beam.connected_robot.robots[i].control_input = np.array([0.0, 0.0, -current_load - load_p, 0.0, 0.0, 0.0])
            else:   
                simulator_beam.connected_robot.robots[i].control_input = np.array([0.0, 0.0, -load, 0.0, 0.0, 0.0])

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



   
    simulation_period = period_detect(time_collection, posture_collection[:, -1, 2, 3])
    simulation_amplitude = amplitude_detect(time_collection, posture_collection[:, -1, 2, 3] - total_length)
    analytical_period = 2 * np.pi * np.sqrt((mass_p + mass_rod/3)/k)
    analytical_amplitude = g * (mass_p + mass_rod/2) / k
    print(f"Detected period of oscillation from simulation: {simulation_period:.4f} s")
    print(f"Detected amplitude of oscillation from simulation: {simulation_amplitude:.4f} m")
   
    


    z_tip_collection = posture_collection[:, -1, 2, 3] - posture_collection[0, -1, 2, 3]  # Extract the z-position of the tip disk over time
   
    star_delta_L = g * (mass_p + mass_rod/2) / k  # Analytical solution for the static deflection of the cantilever beam under the applied load
    star_T = 2 * np.pi * np.sqrt((mass_p + mass_rod/3)/k) #Analytical solution for the period of oscillation of the cantilever beam under the applied load
    y_axis_value = z_tip_collection / star_delta_L
    x_axis_value = time_collection / star_T

    y_analytical =  (1 + np.sin(2 * np.pi * time_collection / star_T - (np.pi / 2))) 

    plt.figure()
    plt.plot(x_axis_value, y_axis_value, label = "simulation result")
    plt.plot(x_axis_value, y_analytical, label = "analytical solution", color='black', linestyle='dashed')
    plt.xlabel("Time (s)")
    plt.ylabel("Z-Position (m)")
    plt.title("Vertical Oscillation of Cantilever Beam Tip Under Gravity")
    plt.legend()
    plt.grid()
    plt.show()

 


    """
    plt.plot(time_collection, bending_internal_couple_collection[:,0,0], label="Bending Internal Couple")
    plt.plot(time_collection, shear_internal_couple_collection[:,0,0], label="Shear Internal Couple")
    plt.plot(time_collection, tau_x_base_collection[:,0], label="Tau_x Base")
    plt.plot(time_collection, force_collection[:,0,3], label="Total Tau 0")
    plt.plot(time_collection, strain_local_collection[:,0,1], label="Total Tau 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Internal Couple (N·m)")
    plt.title("Internal Couples on the First Disk")
    plt.legend()
    plt.grid()
    plt.show()
    """


    plot_disk_chain_results(time_collection, posture_collection, orientation_collection, force_collection)

    
    # ── 3-D animation ────────────────────────────────────────────────────────
    """
    animate_slender_robot(
        time_collection   = time_collection,
        posture_collection= posture_collection,
        force_collection  = None,
        disk_radius       = width * 5,
        output_path       = 'slender_robot_simulation.mp4',  # falls back to .gif if ffmpeg missing
        fps               = 20,
        force_scale       = 0.5,
        skip_frames       = 5,
        view_yaw          = 0.0,   # degrees — rotate camera around world Z
        view_pitch        = 0.0,    # degrees — camera elevation above horizontal
        view_roll         = 0.0,     # degrees — roll around the line of sight
    )
    """
    



    


