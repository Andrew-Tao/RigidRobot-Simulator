from .methods3D import SE3LieAlgebra, rpy_to_Q
import numpy as np
import matplotlib.pyplot as plt
lie3 = SE3LieAlgebra()
from scipy.spatial.transform import Rotation as R



class RigidRobot3D:
    def __init__(
            self, 
            position: np.ndarray, 
            orientation: np.ndarray, 
            mass: float, 
            inertia: np.ndarray, 
            linear_velocity=np.zeros(3), 
            angular_velocity= np.zeros(3),
            radius = 0.2,
            thickness = 0.2, 
            track_width_between_wheels = 0.15,
        ):

        """
        position: (x, y, z) in R^3
        orientation: (roll, pitch, yaw) in radians
        mass: scalar mass of the robot
        inertia: (Jx, Jy, Jz) moment of inertia around the center of mass
        linear_velocity: (v_x, v_y, v_z) in R^3
        angular_velocity: (omega_x, omega_y, omega_z) in radians per second
        environment_resitriction: for environment interaction logic
        thickness: thickness of the robot in z direction
        track_width_between_wheels: distance between the left and right wheels
        """

        self.radius = radius
        self.thickness = thickness
        self.posture = self.compute_posture(position, orientation)
        self.velocity_matrix = self.compute_velocity_matrix(linear_velocity, angular_velocity)
        self.mass_matrix = np.diag([mass,mass,mass, inertia[0], inertia[1], inertia[2]])  # Mass matrix in SE(3)
        self.momentum = self.mass_matrix @ lie3.vee(self.velocity_matrix)  # 3-vector μ = M·ξ
        self.environment_resitriction = None
        self.force = None  # Generalized force in R^3 vector form [f_x, f_y, tau]
        self.control_input = np.zeros(2)  # Control input (force_l, force_r)
        self.friction_coefficient = 0.6  # Simple friction model

    def compute_posture(self, position, orientation):
        Q = rpy_to_Q(orientation)
        T = np.eye(4)
        T[:3, :3] = Q
        T[:3, 3] = position
        return T

    def compute_velocity_matrix(self, linear_velocity, angular_velocity):
        velocity_vector = np.array([
            linear_velocity[0], 
            linear_velocity[1], 
            linear_velocity[2], 
            angular_velocity[0], 
            angular_velocity[1], 
            angular_velocity[2]])
        velocity_matrix = lie3.hat(velocity_vector)
        return velocity_matrix

    def compute_force_local(
        self, 
        control_input: np.ndarray, 
        spring_anchor_point=np.array([0.0, 0.0, 0.0]), 
        spring_stiffness=1,
        torque_spring_anchor_orientation=np.diag([1.0, 1.0, 1.0]),
        torque_spring_stiffness=1,
        spring_original_length=0.04
        ):

        damping_coefficient = 0.1
        position = self.posture[:3, 3]
        orientation_Q = self.posture[:3, :3]
       
        Q_error = torque_spring_anchor_orientation @ orientation_Q.T

        #print(f"orentation_Q: {orientation_Q}")


        v1, v2, v3 = self.velocity_matrix[:3, 3]   # body-frame linear velocity
        omega_x, omega_y, omega_z   = self.velocity_matrix[2, 1], self.velocity_matrix[0, 2], self.velocity_matrix[1, 0]    # angular velocity
        omega = np.array([omega_x, omega_y, omega_z])

        spring_anchor_point_local = np.linalg.inv(orientation_Q) @ (spring_anchor_point - position)
        
        spring_current_length = np.linalg.norm(spring_anchor_point_local)
        delta_length = spring_current_length - spring_original_length
        if spring_current_length > 1e-6:
            unit_anchor_vector_local = spring_anchor_point_local / spring_current_length
        else:
            unit_anchor_vector_local = np.zeros(3)
        


        linear_spring_force_local = (delta_length * unit_anchor_vector_local) * spring_stiffness
       
        f_x, f_y, f_z = linear_spring_force_local - damping_coefficient * np.array([v1, v2, v3])
       
        #TODO: Add torque spring
        # Torque spring will generate a resistanct torque in all three axes to resist changes in orientation
        theta = R.from_matrix(Q_error).as_rotvec()
        print(f"theta: {theta}")
        print(f"omega: {omega}")
        print(f"momentum: {self.momentum}")
        tau_x, tau_y, tau_z = - torque_spring_stiffness * theta - damping_coefficient * omega
 
        # Add control Inputs (all control inputs are applied as the force or toque in local framework)
        total_force_local = np.array([f_x, f_y, f_z, tau_x, tau_y, tau_z]) + control_input
         

        return total_force_local
    

    def contact(self, environment):
        """
        environment: func (position) -> True if within boundary, False otherwise)
        position in R^2
        """
        self.environment_resitriction = environment

    def is_in_contact(self):
        robot_current_center_position = self.posture[:2, 2]
        radius_vector = np.array([0, self.radius])  
        for theta in np.linspace(0, 2 * np.pi, num=8):  # Check 8 points around the circumference
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[c, -s], [s, c]])
            radius_vector_rotated = rotation_matrix @ radius_vector
            contact_point = robot_current_center_position + radius_vector_rotated
            if not self.environment_resitriction(contact_point):
                return True  # Contact detected
            
        return False  # No contact detected
    


class ConnectedRigidRobots3D:
    def __init__(self, robot1: RigidRobot3D, robot2: RigidRobot3D):
        self.robot1 = robot1
        self.robot2 = robot2