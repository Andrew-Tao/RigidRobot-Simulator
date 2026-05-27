from .methods3D import SE3LieAlgebra, rpy_to_Q
import numpy as np
import matplotlib.pyplot as plt
lie3 = SE3LieAlgebra()
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from .RigidRobot3D import RigidRobot3D


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
        self.base_robot = self.generate_base_robot() # This is an imagine robot which represent the fixed origin point

    def compute_force_local_total_individual_robot(self,robot_index:int):
        connection = self.connection_map[robot_index]
        number_of_connection = len(connection)
        total_force = np.zeros(6)

        for i in range(number_of_connection):
            if not connection[i].to_base:

                is_upon_sequence_flag = True if connection[i].to > robot_index else False 
                
                # See Gazzola Cossart Rod Model Equation 3.8
                if is_upon_sequence_flag == True:
                    anchor_robot = self.robots[connection[i].to]
                    who_i_am_robot = self.robots[robot_index]
                else:
                    anchor_robot = self.robots[robot_index]
                    who_i_am_robot =  self.robots[connection[i].to]

            else:
                who_i_am_robot = self.base_robot
                anchor_robot =  self.robots[robot_index]
                is_upon_sequence_flag = False

            spring_anchor_point = anchor_robot.posture[:3, 3]
            torque_spring_anchor_orientation = anchor_robot.orientation
            # Convert anchor body velocity to world frame: v_world = R @ v_body
            anchor_Q = anchor_robot.posture[:3, :3]
            anchor_velocity_world = anchor_Q @ anchor_robot.velocity_matrix[:3, 3]
            aw_x = anchor_robot.velocity_matrix[2, 1]
            aw_y = anchor_robot.velocity_matrix[0, 2]
            aw_z = anchor_robot.velocity_matrix[1, 0]
            anchor_angular_velocity_world = anchor_Q @ np.array([aw_x, aw_y, aw_z])


            spring_stiffness = connection[i].spring_stiffness
            torque_spring_stiffness = connection[i].torque_spring_stiffness
            spring_original_length = connection[i].spring_original_length
            spring_damping_coefficient = connection[i].spring_damping_coefficient
            torque_spring_damping_coefficient = connection[i].torque_spring_damping_coefficient

            test_flag = robot_index
            half_cross_section_force = self.compute_single_spring_force(
                robot = who_i_am_robot,
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

            if not is_upon_sequence_flag:
                R_transform = self.robots[robot_index].posture[:3, :3].T @ who_i_am_robot.posture[:3, :3]
                half_cross_section_force = np.concatenate([
                    -R_transform @ half_cross_section_force[:3],
                    -R_transform @ half_cross_section_force[3:]
                ])

            #print("half_section_force", half_cross_section_force)

            total_force += half_cross_section_force

        total_force += self.robots[robot_index].control_input
        
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
        original_front_direction_vector = - spring_original_length * orientation_Q[:3,2] 
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
        if True == False: 
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
        #print("\n")

        return total_force_local



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

    def generate_base_robot(self):
        base_robot = RigidRobot3D(
            position = np.zeros(3), 
            orientation = np.zeros(3), 
            mass = 0.0, # This doesn't matter as it won't be used
            inertia = np.zeros(3), # This doesn't matter as it won't be used
            linear_velocity=np.zeros(3), 
            angular_velocity= np.zeros(3),
            radius = 0.2,
            thickness = 0.2, 
            track_width_between_wheels = 0.15,
        )
        return base_robot
            
  