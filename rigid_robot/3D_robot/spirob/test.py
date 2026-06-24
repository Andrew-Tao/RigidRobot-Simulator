
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.force.CableDrivenForce import CableDrivenForce, GravityForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra, log_SO3
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SlenderRobotVisualization import animate_slender_robot

lie3 = SE3LieAlgebra()

def rad(n):
    return (n / 180 * np.pi)

def to_degree(matrix):
    A = np.arccos(matrix[0,0]) / np.pi * 180
    #print(A)
    return A

def Qx(degree):
    c = np.cos(rad(degree))
    s = np.sin(rad(degree))
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, c,   -s],
                  [0.0, s,    c]])
    return A 



c30 = np.cos(rad(90))
s30 = np.sin(rad(90))
c50 = np.cos(rad(50))
s50 = np.sin(rad(50))


Qi = np.array([[c30,-s30,0.0],
                [s30, c30, 0.0],
                [0.0, 0.0, 1.0]])

Qj = np.array([[c50,-s50,0.0],
                [s50, c50, 0.0],
                [0.0, 0.0, 1.0]])

#print(Qi)

R = np.eye(3)
#print(R)

R = np.eye(3) @ Qi
R =np.eye(3) @ Qi @ Qx(90)
R = Qx(90)@ Qi @ np.eye(3)

print(to_degree(R))
print(R@ np.array([1.0,1.0,1.0]))


Trans = np.eye(3).T @ R
skew = log_SO3(Trans)
print(skew/ np.pi * 180)

#degree = to_degree(R)

#print(skew[2]/np.pi * 180)

#print(degree)

