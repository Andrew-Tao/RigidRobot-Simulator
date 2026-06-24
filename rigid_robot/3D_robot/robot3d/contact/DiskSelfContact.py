import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from robot3d.methods3D import SE3LieAlgebra, rpy_to_Q, log_SO3, vee
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from BeamGenerator import generate_series_connection_map
from examples.postprocessing.SlenderRobotVisualization import animate_slender_robot
import numpy as np
import matplotlib.pyplot as plt

class SelfAngleContact:
    def __init__(self):
        self.stiffness_coefficient = 10.0 
        self.damping_coefficient = 1.5 * 1e-2
    
    def compute_force_collection(self, slender_robot):

        n_robots = len(slender_robot.robots)
        #print("n_robots", n_robots)
        #TODO: Change the resistance torque to force
        contact_tor_collection = []

        for i in range(n_robots):
           
            # Get Base robot velocity and director
            if i == 0:
                base_robot_director = np.eye(3); base_robot_director[2,2] = - 1.0
                base_robot_velocity = np.zeros((3, 3))
            else: 
                base_robot_director = slender_robot.robots[i-1].posture[:3,:3]
                base_robot_velocity = slender_robot.robots[i-1].velocity_matrix[:3,:3]


            current_director = slender_robot.robots[i].posture[:3,:3]
            current_velocity = slender_robot.robots[i].velocity_matrix[:3,:3]

            # Compute Rotation
            trans = base_robot_director.T @ current_director
            skew = log_SO3(trans)

            norm_skew = np.linalg.norm(skew)
            if norm_skew < 1e-8:
                contact_tor_collection.append(np.zeros(3))
                continue
            unit_skew = skew / norm_skew

            omega = vee(current_velocity) - vee(base_robot_velocity)   # extract angular velocity from hat(ω) # TODO: Is this correct?

            xy_rotation_length = np.sqrt(skew[0]**2 + skew[1]**2)
            velocity_magnitude = np.dot(omega, unit_skew)

            penetrate_distance = xy_rotation_length - (np.pi / 6)

            tau_ct = np.heaviside(penetrate_distance, 1) * (-self.stiffness_coefficient * penetrate_distance - self.damping_coefficient * velocity_magnitude) * unit_skew
            contact_tor_collection.append(tau_ct)

            #print("kkk",contact_tor_collection)


            #tau_ct = np.heaviside(penetrate_distance, 1) * (- self.damping_coefficient * velocity_magnitude) * unit_skew
            #tau_ct = np.heaviside(penetrate_distance, 1) * (-self.stiffness_coefficient * penetrate_distance) * unit_skew

            #print("pentra",(np.pi / 6) - xy_rotation_length )
            #tau_ct = np.heaviside(penetrate_distance, 1) * np.array([1,1])
            #print("tau_ct", tau_ct)
            #print("unit_skew",unit_skew)

        return np.array(contact_tor_collection)
        
    
class SelfContact:
    def __init__(self):
        pass















    



