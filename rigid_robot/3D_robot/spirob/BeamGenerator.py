import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
from robot3d.robot.RigidRobot3D import RigidRobot3D

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


def generate_series_robot_disks_tapper(
    n_disks,
    base_length_between_disks,
    initial_position,
    initial_orientation,
    base_mass,
    base_moment_inertia,
    base_radius,
    tapper_angle
    ):

    # First pass: compute all disk positions
    positions = []
    for i in range(n_disks):
        modified_gap = i * tapper_angle * 0.55 * base_length_between_disks * i
        
        pos = initial_position + np.array([0.0, 0.0, -(i+1) * base_length_between_disks + modified_gap])
        positions.append(pos)

    # Per-disk thickness = 0.5 * gap to the next disk (last disk reuses its predecessor's gap)
    thicknesses = []
    for i in range(n_disks):
        if i < n_disks - 1:
            gap = np.linalg.norm(positions[i + 1] - positions[i])
        else:
            gap = np.linalg.norm(positions[i] - positions[i - 1]) if n_disks > 1 else base_length_between_disks
        thicknesses.append(0.5 * gap)

    print("position", positions)
    print("thickness", thicknesses)

    robot_disks = []
    for i in range(n_disks):
        modified_radius = (base_radius - i * tapper_angle * base_radius)
        modified_mass = ((modified_radius / base_radius)) * base_mass
        modified_inertia = ((modified_radius / base_radius)) * base_moment_inertia

        robot_disk = RigidRobot3D(
            position = positions[i],
            orientation = initial_orientation,
            mass = modified_mass,
            inertia = modified_inertia,
            linear_velocity = np.zeros(3),
            angular_velocity = np.zeros(3),
            radius = modified_radius,
            thickness = thicknesses[i],
        )
        robot_disks.append(robot_disk)
    return robot_disks

def generate_series_connection_map(
    cantilever_beam,
    k_spring,
    damping_spring,
    k_tortional_spring,
    damping_tortional_spring,
    spring_original_length=None):
    """
    spring_original_length : float or None
        Shared rest length for all springs.  Pass None (default) to have each
        connection's rest length measured from the robots' initial positions,
        giving zero pretension regardless of taper or spacing.
    """

    def _dist(pos_a, pos_b):
        return float(np.linalg.norm(pos_b - pos_a))

    base_pos  = np.zeros(3)
    disk0_pos = cantilever_beam.robots[0].posture[:3, 3]
    L0 = spring_original_length if spring_original_length is not None else _dist(base_pos, disk0_pos)

    cantilever_beam.add_connection(
        (0, 1),
        to_base = True,
        spring_stiffness = k_spring,
        spring_damping_coefficient = damping_spring,
        torque_spring_stiffness = k_tortional_spring,
        torque_spring_damping_coefficient = damping_tortional_spring,
        spring_original_length = L0,
    )

    for i in range(len(cantilever_beam.robots) - 1):
        if spring_original_length is not None:
            L_i = spring_original_length
        else:
            p_i   = cantilever_beam.robots[i].posture[:3, 3]
            p_ip1 = cantilever_beam.robots[i + 1].posture[:3, 3]
            L_i   = _dist(p_i, p_ip1)

        cantilever_beam.add_connection(
            (i, i + 1),
            to_base = False,
            spring_stiffness = k_spring,
            spring_damping_coefficient = damping_spring,
            torque_spring_stiffness = k_tortional_spring,
            torque_spring_damping_coefficient = damping_tortional_spring,
            spring_original_length = L_i,
        )
    return 0

