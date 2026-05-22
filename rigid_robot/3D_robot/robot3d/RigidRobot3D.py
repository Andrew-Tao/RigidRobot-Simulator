from .methods3D import SE3LieAlgebra, rpy_to_Q
import numpy as np
import matplotlib.pyplot as plt
lie3 = SE3LieAlgebra()
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

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
        self.orientation = orientation
        # Since the posture matrix can no longer correctly track the constitution relationship
        # after the robot rotate 360 degrees. So wee need a true omega tracking twist over time
        self.thickness = thickness
        self.posture = self.compute_posture(position, orientation)
        self.velocity_matrix = self.compute_velocity_matrix(linear_velocity, angular_velocity)
        self.mass_matrix = np.diag([mass,mass,mass, inertia[0], inertia[1], inertia[2]])  # Mass matrix in SE(3)
        self.momentum = self.mass_matrix @ lie3.vee(self.velocity_matrix)  # 3-vector μ = M·ξ
        self.environment_resitriction = None
        self.force = None  # Generalized force in R^3 vector form [f_x, f_y, tau]
        self.control_input = np.zeros(6)  # Control input (force_l, force_r)
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
    
    #TODO: The Constitutive relationship is not updated. So it's generally not correct !!!!! The most up-to-date one is the one for the ConnectdRigidRobot

    def compute_force_local(
        self, 
        control_input: np.ndarray, 
        spring_anchor_point=np.array([0.0, 0.0, 0.0]), 
        spring_stiffness=1,
        torque_spring_anchor_orientation=np.diag([1.0, 1.0, 1.0]),
        torque_spring_stiffness=0.01,
        spring_original_length=0.04
        ):

        

        damping_coefficient = 0.1
        position = self.posture[:3, 3]
        orientation_Q = self.posture[:3, :3]
       
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

        theta = self.orientation.copy()
        tau_x, tau_y, tau_z = - torque_spring_stiffness * theta - 0.00002 * omega

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
    

@dataclass
class Connection:
    to: int
    to_base: bool
    spring_anchor_point_local_self: np.ndarray
    spring_anchor_point_to: np.ndarray
    spring_original_length: float
    spring_stiffness: float
    torque_spring_anchor_orientation: np.ndarray
    torque_spring_stiffness: float
    spring_damping_coefficient: np.ndarray
    torque_spring_damping_coefficient: np.ndarray

@dataclass
class ExternalForce:
    external_force: np.ndarray
    external_torque: np.ndarray

class ConnectedRigidRobots3D:
    def __init__(self, robots:list[RigidRobot3D]):
        self.robots = robots
        self.connection_map = {i: list() for i in range(len(self.robots))}
        #TODO: this is now just a place holder for empty external force
        self.external_force = np.zeros(6)

        self.force = np.zeros((len(robots), 6)) # Collection of the force for each individual robot #TODO: Use this

    def compute_external_force(self):
        pass

    def compute_force_local_total_individual_robot(self,robot_index:int, external_force = np.zeros(6)):
        connection = self.connection_map[robot_index]
        number_of_connection = len(connection)
        total_force = np.zeros(6)

        

        for i in range(number_of_connection):
            if not connection[i].to_base:
                anchor_robot = self.robots[connection[i].to]
                spring_anchor_point = anchor_robot.posture[:3, 3]
                torque_spring_anchor_orientation = anchor_robot.orientation
                # Convert anchor body velocity to world frame: v_world = R @ v_body
                anchor_Q = anchor_robot.posture[:3, :3]
                anchor_velocity_world = anchor_Q @ anchor_robot.velocity_matrix[:3, 3]
                aw_x = anchor_robot.velocity_matrix[2, 1]
                aw_y = anchor_robot.velocity_matrix[0, 2]
                aw_z = anchor_robot.velocity_matrix[1, 0]
                anchor_angular_velocity_world = anchor_Q @ np.array([aw_x, aw_y, aw_z])

                #TODO: If you get direction like this, when you model the slender robot seq have to be [base, robot0, robot1 ...]
                is_upon_sequence_flag = True if connection[i].to > robot_index else False 

            else:
                spring_anchor_point = np.zeros(3)
                torque_spring_anchor_orientation = np.zeros(3)
                anchor_velocity_world = np.zeros(3)
                anchor_angular_velocity_world = np.zeros(3)
                is_upon_sequence_flag = False

            spring_stiffness = connection[i].spring_stiffness
            torque_spring_stiffness = connection[i].torque_spring_stiffness
            spring_original_length = connection[i].spring_original_length
            spring_damping_coefficient = connection[i].spring_damping_coefficient
            torque_spring_damping_coefficient = connection[i].torque_spring_damping_coefficient

            test_flag = robot_index
            total_force += self.compute_single_spring_force(
                robot = self.robots[robot_index],
                is_upon_anchor_disk= is_upon_sequence_flag,
                spring_anchor_point = spring_anchor_point,
                torque_spring_anchor_orientation= torque_spring_anchor_orientation,
                spring_stiffness= spring_stiffness,
                torque_spring_stiffness= torque_spring_stiffness,
                spring_original_length = spring_original_length,
                anchor_velocity_world=anchor_velocity_world,
                anchor_angular_velocity_world=anchor_angular_velocity_world,
                test_flag = test_flag,
                spring_damping_coefficient= spring_damping_coefficient,
                torque_spring_damping_coefficient= torque_spring_damping_coefficient
            )

        total_force += self.robots[robot_index].control_input
        total_force += external_force
        
        """
        if robot_index == 1:
            from tqdm import tqdm
            print(f"external_force {external_force}")
            print(f"control_input {self.robots[robot_index].control_input}")
            print(f"total_force {total_force}")
        """
        return total_force

    def compute_single_spring_force(
        self,
        robot: RigidRobot3D,
        is_upon_anchor_disk: bool, 
        spring_anchor_point=np.array([0.0, 0.0, 0.0]),
        torque_spring_anchor_orientation = np.array([0.0, 0.0, 0.0]),
        spring_stiffness=np.array([1.0,1.0,1.0]),
        torque_spring_stiffness=np.array([0.01,0.01,0.01]),
        spring_original_length=0.04,
        anchor_velocity_world=np.zeros(3),
        anchor_angular_velocity_world=np.zeros(3),
        test_flag = 3,
        shear_stiffness = 5.0,
        spring_damping_coefficient = np.array([1.0,1.0,1.0]),
        torque_spring_damping_coefficient = np.array([1e-3,1e-3,1e-3])
        ):

        # TODO: Write a add_damping method


        position = robot.posture[:3, 3]
        orientation_Q = robot.posture[:3, :3]

        v_body = robot.velocity_matrix[:3, 3]   # body-frame linear velocity = R^T * v_world
        omega_x, omega_y, omega_z   = robot.velocity_matrix[2, 1], robot.velocity_matrix[0, 2], robot.velocity_matrix[1, 0]    # angular velocity
        omega = np.array([omega_x, omega_y, omega_z])

        #---------------------------------Elongation & Shear -------------------------------------
        relative_spring_anchor_point_global = (spring_anchor_point - position)
        original_front_direction_vector = spring_original_length * orientation_Q[:3,2] 
        if is_upon_anchor_disk: original_front_direction_vector = - original_front_direction_vector 
        strain_local = orientation_Q.T @ (relative_spring_anchor_point_global - original_front_direction_vector) # strain_L = Q.T (et - d3)

         # Damping uses relative velocity: v_self_world - v_anchor_world, expressed in body frame
        relative_velocity_body = v_body - orientation_Q.T @ anchor_velocity_world

        linear_spring_force_local = strain_local * spring_stiffness
        f_x, f_y, f_z = linear_spring_force_local - spring_damping_coefficient * relative_velocity_body

        #---------------------------------Bending & Twisting ------------------------------
        theta = robot.orientation.copy() - torque_spring_anchor_orientation
        relative_omega = omega - orientation_Q.T @ anchor_angular_velocity_world
        bend_twist_internal_couple = - torque_spring_stiffness * theta
        #TODO: Is this correct ???????
        front_direction_unit_vector = - np.array([0.0, 0.0, 1.0])
        
        shear_stretch_internal_couple =  spring_original_length * np.cross(front_direction_unit_vector, linear_spring_force_local)
        if not is_upon_anchor_disk: shear_stretch_internal_couple = np.zeros(3)
        tau_x, tau_y, tau_z = bend_twist_internal_couple + shear_stretch_internal_couple - torque_spring_damping_coefficient * relative_omega

        total_force_local = np.array([f_x, f_y, f_z, tau_x, tau_y, tau_z])
        if test_flag != 3: 
            pass 
            print("robot_index", test_flag)
            print("spring_anchor_point",  spring_anchor_point)
            print("is_upon",is_upon_anchor_disk)
           
            #print("position",position)
            print("strain_local",strain_local)
            #print("original_front", original_front_direction_vector )
            #print("Spring_anchor_point_global_relative", relative_spring_anchor_point_global)
            print("total_force", total_force_local)
            print("shear_induced_couple", shear_stretch_internal_couple)
        print("\n")

        return total_force_local


    def compute_internal_force(self, robot_index: int):
        """Spring/damping forces only — excludes control_input and external_force."""
        connection = self.connection_map[robot_index]
        total_force = np.zeros(6)
        for i in range(len(connection)):
            if not connection[i].to_base:
                anchor_robot = self.robots[connection[i].to]
                spring_anchor_point = anchor_robot.posture[:3, 3]
                torque_spring_anchor_orientation = anchor_robot.orientation
                anchor_Q = anchor_robot.posture[:3, :3]
                anchor_velocity_world = anchor_Q @ anchor_robot.velocity_matrix[:3, 3]
                aw_x = anchor_robot.velocity_matrix[2, 1]
                aw_y = anchor_robot.velocity_matrix[0, 2]
                aw_z = anchor_robot.velocity_matrix[1, 0]
                anchor_angular_velocity_world = anchor_Q @ np.array([aw_x, aw_y, aw_z])
                is_upon_sequence_flag = connection[i].to > robot_index
            else:
                spring_anchor_point = np.zeros(3)
                torque_spring_anchor_orientation = np.zeros(3)
                anchor_velocity_world = np.zeros(3)
                anchor_angular_velocity_world = np.zeros(3)
                is_upon_sequence_flag = False

            total_force += self.compute_single_spring_force(
                robot=self.robots[robot_index],
                is_upon_anchor_disk=is_upon_sequence_flag,
                spring_anchor_point=spring_anchor_point,
                torque_spring_anchor_orientation=torque_spring_anchor_orientation,
                spring_stiffness=connection[i].spring_stiffness,
                torque_spring_stiffness=connection[i].torque_spring_stiffness,
                spring_original_length=connection[i].spring_original_length,
                anchor_velocity_world=anchor_velocity_world,
                anchor_angular_velocity_world=anchor_angular_velocity_world,
                test_flag=3,
                spring_damping_coefficient=connection[i].spring_damping_coefficient,
                torque_spring_damping_coefficient=connection[i].torque_spring_damping_coefficient,
            )
        return total_force

   

    def add_connection(self,
        index_pairs:tuple, # If to_base is true, the second elment is ignored
        to_base: bool,
        # So far this will always be zero, No offset allowed
        spring_anchor_point_local_1=np.array([0.0, 0.0, 0.0]), # spring_anchor_point at the robot's local frame. Offset to center of robot1
        spring_anchor_point_local_2 = np.array([0.0,0.0,0.0]),  # spring_anchor_point at the robot's local frame. Offset to center of robot2
        spring_stiffness=np.array([1.0,1.0,1.0]),
        torque_spring_anchor_orientation=np.zeros(3),
        torque_spring_stiffness=np.array([1.0,1.0,1.0]),
        spring_original_length=0.04,
        spring_damping_coefficient = np.array([1.0,1.0,1.0]),
        torque_spring_damping_coefficient = np.array([1e-3,1e-3,1e-3])
        ):

        "index_pair: (first robot index, second robot index)"
        i, j = index_pairs

        self.connection_map[i].append(Connection(
            to=j,
            to_base= to_base,
            spring_anchor_point_local_self=spring_anchor_point_local_1,
            spring_anchor_point_to=spring_anchor_point_local_2,
            spring_original_length=spring_original_length,
            spring_stiffness=spring_stiffness,
            torque_spring_anchor_orientation=torque_spring_anchor_orientation,
            torque_spring_stiffness=torque_spring_stiffness,
            spring_damping_coefficient = spring_damping_coefficient,
            torque_spring_damping_coefficient = torque_spring_damping_coefficient
        ))

        if not to_base:
            self.connection_map[j].append(Connection(
                to=i,
                to_base = False,
                spring_anchor_point_local_self=spring_anchor_point_local_2,
                spring_anchor_point_to=spring_anchor_point_local_1,
                spring_original_length=spring_original_length,
                spring_stiffness=spring_stiffness,
                torque_spring_anchor_orientation=torque_spring_anchor_orientation,
                torque_spring_stiffness=torque_spring_stiffness,
                spring_damping_coefficient = spring_damping_coefficient,
                torque_spring_damping_coefficient = torque_spring_damping_coefficient
            ))
            
  