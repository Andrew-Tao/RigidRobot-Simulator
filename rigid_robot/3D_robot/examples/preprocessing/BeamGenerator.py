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
