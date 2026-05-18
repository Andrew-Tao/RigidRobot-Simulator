"""
Simulation of a serial chain of rigid disk robots connected by springs and torsional springs.

This example models a vertical series of rigid disk robots hanging under gravity.
Each disk is connected to the next disk through a linear spring and a torsional spring,
allowing both translational and rotational coupling between neighboring rigid bodies.

The chain is anchored at the base point (0, 0, 0). The first disk, robot1, is attached
to this fixed base anchor. The remaining disks are connected sequentially, forming a
serial rigid-robot chain. The final disk, robot n, is the bottom disk of the hanging
system.


The purpose of this simulation is to study the dynamic response of a spring-connected
multi-rigid-body system under gravity, including the effects of both linear spring
forces and torsional spring moments.
"""


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.RigidRobot3D import RigidRobot3D, ConnectedRigidRobots3D
from robot3d.Simulator3D import Simulator3D
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


lie3 = SE3LieAlgebra()

if __name__ == "__main__":

    robot_disk_1 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04]),
        orientation = np.array([0.0, 0.0, 1.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )

    robot_disk_2 = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04]),
        orientation = np.array([0.0, 0.0, 1.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )
    robot_collection = [robot_disk_1,robot_disk_2]
    #TODO: Start from here
    connection_map = None
    slender_robot = ConnectedRigidRobots3D(
        robots= robot_collection,
        connection_map = None
    )