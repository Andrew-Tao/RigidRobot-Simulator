"""
Cantilever beam deflection case. This is a test case for the connected rigid robot model. We have two disks connected by a spring, and we apply a force on the second disk to see how the system behaves. The expected behavior is that the second disk will deflect due to the applied force, and the first disk will also move due to the connection between them. We can adjust the spring stiffness and damping coefficients to see how they affect the system's response.
Used for Benchmarking the connected rigid robot model, and for visualizing the dynamic response of a simple two-disk system under external force and spring connection. This case can also be used to validate the implementation of the spring forces and torques in the ConnectedRigidRobots3D class.
The analytical solution is provided by the Euler-Bernoulli beam theory for a cantilever beam with a point load at the free end. The deflection can be calculated using the formula:
delta = (F * L^3) / (3 * E * I)
where F is the applied force, L is the length of the beam, E is the Young
"""


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.methods3D import SE3LieAlgebra

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


def run_simulation(n_elements, time_step, duration):
    """Run the cantilever beam simulation and return posture_collection.

    Returns
    -------
    posture_collection : np.ndarray, shape (T, n_elements, 4, 4)
    """

    F = 0.0004
    persistence_time = 200
    width = 0.01

    load = F / n_elements
    E_module = 1.2 * 1e7 / 1e6 * 2.0
    poisson_ratio = 0
    G_module = E_module / (2 * (1 + poisson_ratio))
    total_length = 0.5
    I_x = 0.01**4 / 12 * 1000
    I_y = 0.01**4 / 12 * 1000
    I_z = I_x + I_y

    density = 1000
    damping_spring = np.array([1.0, 1.0, 1.0]) / 30
    damping_tortional_spring = np.array([2e-3, 2e-3, 2e-3]) / 40

    total_volume = 0.01**2 * total_length
    total_mass = density * total_volume

    segment_mass = total_mass / n_elements
    segment_length = total_length / n_elements
    cross_section_area = 0.01**2

    k_spring = np.array([G_module * (4/3) * cross_section_area,
                         G_module * (4/3) * cross_section_area,
                         E_module * cross_section_area])

    k_tortional_spring = np.array([
        E_module * I_x, E_module * I_y, G_module * I_z
    ])

    I_h = (1/12) * segment_mass * ((0.01)**2 + (0.01)**2)
    I_w = (1/12) * segment_mass * ((0.01)**2 + segment_length**2)
    I_d = (1/12) * segment_mass * ((0.01)**2 + segment_length**2)
    moment_inertia = np.array([I_w, I_d, I_h])

    robot_collection = generate_series_robot_disks(
        n_disks=n_elements,
        length_between_disks=segment_length,
        initial_position=np.array([0.0, 0.0, 0.0]),
        initial_orientation=np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, -1.0]]),
        mass=segment_mass,
        moment_inertia=moment_inertia,
        radius=width,
        thickness=0.025,
    )

    cantilever_beam = ConnectedRigidRobots3D(robots=robot_collection)

    generate_series_connection_map(
        cantilever_beam,
        k_spring,
        damping_spring,
        k_tortional_spring,
        damping_tortional_spring,
        spring_original_length=segment_length,
    )

    simulator_beam = MutiRobotSimulator3D(
        time_step=time_step,
        duration=duration,
        stepper='explicit_euler',
        control_logic=None)

    simulator_beam.attach(cantilever_beam)

    for i in range(n_elements):
        simulator_beam.connected_robot.robots[i].control_input = np.array([0.0, load, 0.0, 0.0, 0.0, 0.0])

    while simulator_beam.run():
        if simulator_beam.current_time >= persistence_time:
            for i in range(n_elements):
                simulator_beam.connected_robot.robots[i].control_input = np.zeros(6)
        simulator_beam.multi_robots_step()
        simulator_beam.multi_robot_record()

    return np.array(simulator_beam.posture_collection)


def extract_z_profile_equilibrium(posture_collection):
    """Return z-positions of all disks at the final (equilibrium) frame.

    Parameters
    ----------
    posture_collection : np.ndarray, shape (T, n_disks, 4, 4)

    Returns
    -------
    z_profile : np.ndarray, shape (n_disks,)
    """
    return posture_collection[-1, :, 2, 3]


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  Convergence analysis                                                #
    #  Compares the full equilibrium z-position profile of each mesh      #
    #  against the benchmark (finest mesh) using L1, L2, and L-inf norms. #
    #  Benchmark posture is saved to disk after its simulation completes. #
    # ------------------------------------------------------------------ #

    total_length = 0.5  # m — must match run_simulation

    n_elements_list = [5,        10,      15,       20,      25     ]
    time_step_list  = [0.064816, 0.00852, 0.002568, 0.0012,  0.00066]
    duration_list   = [60.0,     60.0,    60.0,     60.0,    60.0   ]

    benchmark_n   = n_elements_list[-1]
    benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_posture.npy")

    # --- Run benchmark first and save to disk ---
    print(f"Running benchmark: n_elements={benchmark_n}, "
          f"time_step={time_step_list[-1]}, duration={duration_list[-1]}")
    posture_bench = run_simulation(benchmark_n, time_step_list[-1], duration_list[-1])
    np.save(benchmark_path, posture_bench)
    print(f"Benchmark saved to {benchmark_path}")

    z_bench = extract_z_profile_equilibrium(posture_bench)          # shape (benchmark_n,)
    s_bench = np.arange(1, benchmark_n + 1) * (total_length / benchmark_n)

    # --- Run coarser meshes and accumulate errors ---
    errors_l1   = []
    errors_l2   = []
    errors_linf = []

    for n, dt, dur in zip(n_elements_list[:-1], time_step_list[:-1], duration_list[:-1]):
        print(f"Running simulation: n_elements={n}, time_step={dt}, duration={dur}")
        posture  = run_simulation(n, dt, dur)
        z_coarse = extract_z_profile_equilibrium(posture)           # shape (n,)
        s_coarse = np.arange(1, n + 1) * (total_length / n)

        # Linearly interpolate benchmark z-profile to the coarse disk positions
        z_bench_interp = np.interp(s_coarse, s_bench, z_bench)
        diff = z_coarse - z_bench_interp

        errors_l1.append(np.linalg.norm(diff, ord=1))
        errors_l2.append(np.linalg.norm(diff, ord=2))
        errors_linf.append(np.linalg.norm(diff, ord=np.inf))

    coarser_ns = n_elements_list[:-1]

    plt.figure(figsize=(8, 5))
    plt.plot(coarser_ns, errors_l2,   'o-', color='blue',  linewidth=1.5, markersize=6, label=r'$L^2$ norm')
    plt.plot(coarser_ns, errors_l1,   's-', color='red',   linewidth=1.5, markersize=6, label=r'$L^1$ norm')
    plt.plot(coarser_ns, errors_linf, '^-', color='black', linewidth=1.5, markersize=6, label=r'$L^\infty$ norm')
    plt.xlabel("Number of Elements")
    plt.ylabel("Error")
    plt.title(f"Convergence Analysis  (benchmark: {benchmark_n} elements)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
