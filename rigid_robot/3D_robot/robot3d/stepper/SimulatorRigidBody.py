import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
try:
    from ..methods3D import SE3LieAlgebra, rotation_matrix_to_euler_zyx, vee, rot_x
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from robot3d.methods3D import SE3LieAlgebra, rotation_matrix_to_euler_zyx, vee, rot_x
from tqdm import tqdm

lie3 = SE3LieAlgebra()

class RigidObjectSimulator:
    def __init__(self, time_step = 0.1, duration = 10.0, control_logic: callable = None, stepper = 'explicit_euler', show_progress = True):
        self.solid_object_collection = list()  # List of robots in the simulation
        self.time_step = time_step
        self.duration = duration
        self.control_logic = control_logic
        self.stepper = stepper
        self.external_force_list = []
        self.time_collection = []  # To store time points
        self.posture_collection = []  # To store robot positions over time
        self.velocity_matrix_collection = []  # To store robot velocities over time
        self.force_collection = []  # To store forces over time
        self.momentum_collection = []  # To store momentum over time
        self.orientation_collection = []
        # Since the posture matrix can no longer correctly track the constitution relationship
        # after the robot rotate 360 degrees. So wee need a true omega tracking twist over time
        self._total_steps = int(duration / time_step)
        self._last_pct = 0
        self._pbar = tqdm(total=100, desc="Simulating", unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]") if show_progress else None


        # ---------For debug -------
        self.penetration_collection = []
        self.end_position_collection = []   # [[top_z, bottom_z], ...] per step
        self.F_wall_collection = [] 

    def run(self):
        if len(self.time_collection) * self.time_step >= self.duration:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            return False  # Simulation finished
        return True

    @property
    def robots(self):
        return self.solid_object_collection

    def attach(self, robot):
        self.solid_object_collection.append(robot)

    def add_external_force(self, force_type):
        self.external_force_list.append(force_type)

    def step(self):

        self.current_time = len(self.time_collection) * self.time_step
        n = len(self.solid_object_collection)

        # Gather colleciotn 

        # Gather all forces and states at time k before updating any robot
        states_k = []
        forces_k = []
        posture_k_half_collection = np.zeros((n, 4, 4))

        # ------------Add Internal forces, adding to forces_k---------------
        for i in range(n):
            solid_object = self.solid_object_collection[i]
            states_k.append((solid_object.momentum.copy(), solid_object.posture.copy(), solid_object.velocity_matrix.copy()))
            forces_k.append(np.zeros(6))

        #-------------Add External Forces, adding to forces_k---------------
        external_force_k = np.zeros((n, 6))
        for force_type in self.external_force_list:
            external_force_k += force_type.compute_force_collection(self, self.current_time)
        for i in range(n):
            forces_k[i] += external_force_k[i]

        self._last_forces_k = forces_k

        # -------------Position Verlet Integration --------------------
        for i in range(n):
            momentum_k, posture_k, velocity_k_matrix = states_k[i]
            posture_k_phalf = posture_k @ lie3.exp(velocity_k_matrix * self.time_step / 2)
            self.solid_object_collection[i].posture = posture_k_phalf
            posture_k_half_collection[i] = posture_k_phalf

        forces_k_half = []
        # Add external forces (gravity, cable, etc.) evaluated at k+1/2 postures
        external_force_k_half = np.zeros((n, 6))
        for force_type in self.external_force_list:
            external_force_k_half += force_type.compute_force_collection(self, self.current_time + self.time_step / 2)
        for i in range(n):
            forces_k_half.append(external_force_k_half[i])

        # Compute Contact Force
        for i in range(n): 
            forces_k_half[i] +=(self.compute_contact_force(self.solid_object_collection[i], forces_k_half[i]))
        #Compute Contact Friction
        for i in range(n):
            forces_k_half[i] += (self.compute_contact_friction(self.solid_object_collection[i], forces_k_half[i]))

        # Phase 3: integrate ALL robots to k+1
        for i in range(n):
            solid_object = self.solid_object_collection[i]
            momentum_k, posture_k, velocity_k_matrix = states_k[i]
            posture_k_phalf = posture_k_half_collection[i]
            force_k_phalf = forces_k_half[i]

            momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix * self.time_step) @ momentum_k) + (force_k_phalf * self.time_step)
            velocity_kp1_matrix = lie3.hat(solid_object.mass_matrix_inv @ momentum_kp1)
            posture_kp1 = posture_k_phalf @ lie3.exp(velocity_kp1_matrix * (self.time_step / 2))
            xi_kp1 = solid_object.mass_matrix_inv @ momentum_kp1

            solid_object.posture = posture_kp1
            solid_object.momentum = momentum_kp1
            solid_object.velocity_matrix = lie3.hat(xi_kp1)
            solid_object.orientation = rotation_matrix_to_euler_zyx(posture_kp1[:3, :3])

        self.time_collection.append(self.current_time)
        if self._pbar is not None:
            current_pct = int(len(self.time_collection) / self._total_steps * 100)
            if current_pct > self._last_pct:
                self._pbar.update(current_pct - self._last_pct)
                self._last_pct = current_pct

    def compute_contact_friction(self, object, external_force):

        plane_posture = np.zeros((4,4))
        plane_posture[:3, :3] = rot_x(30)

        object_posture = plane_posture.T @ object.posture.copy()
        # Apply a boundary at the z = 0 plan 
        director = object_posture[:3, :3].copy()
        position = object_posture[:3, 3].copy()
        
        unit_parallel = director.T * np.array([1.0, 0.0, 0.0])
        unit_vertical = director.T * np.array([0.0, 0.0, 1.0])


        # ---------- Longitudal Friction -------------
        transitional_static_friction_coefficient = 0.2
        transitional_dynamic_friction_coefficient = 0.18





        # ---------- Rolling Friction ------------------
        rolling_static_friction_coefficient = 0.2 
        rolling_static_friction_coefficient = 0.18




        return np.zeros(6)

    def compute_contact_force(self, object, external_force):

        plan_posture = np.zeros((4,4))
        plan_posture[:3, :3] = rot_x(30)

        height = object.thickness
        end_position_collection = [np.array([0.0, 0.0, height / 2]), np.array([0.0, 0.0, -height / 2])]
        F_wall1 = self.compute_contact_force_one_end(object=object,end_position_vector = end_position_collection[0], plane_posture= plan_posture)
        F_wall2 = self.compute_contact_force_one_end(object=object,end_position_vector = end_position_collection[1], plane_posture= plan_posture) 
        F_wall = F_wall1 + F_wall2

        return F_wall


    def compute_contact_force_one_end(self, object, end_position_vector, plane_posture):
        """
        Plan_positon SE(3) (4,4)

        """
        object_posture = plane_posture.T @ object.posture.copy()
    
        # Apply a boundary at the z = 0 plan 
        director = object_posture[:3, :3].copy()
        position = object_posture[:3, 3].copy()
    
        end_position = (director @ end_position_vector + position)[2]

        # Calulate the minimal distance
        d_min = end_position.min()
        penetration = object.radius - d_min
    
        damping_coefficient = 1e4
        stiffness_wall = 1e5
        
        unit_wall_up = np.array([0.0, 0.0, 1.0])
        penetration_velocity = np.dot(director @ object.velocity_matrix[:3, 3], unit_wall_up)

        # TODO: replace 0 with i to compute Muti bodies
        F_wall = np.heaviside(penetration, 0) * ( stiffness_wall * penetration - damping_coefficient * penetration_velocity ) * unit_wall_up 
        F_wall = director.T @ F_wall
       
        # Compute Contact force 
        self.penetration_collection.append(penetration)
        self.F_wall_collection.append(F_wall)

        # Contact Torque 
        contact_force_arm_local =  end_position_vector - director.T @ (unit_wall_up * object.radius) 
        contact_torque = np.cross(contact_force_arm_local, F_wall) 

        contact_general_force = np.concatenate((F_wall, contact_torque), axis= 0)

        return contact_general_force
    

    def record(self):
        self.posture_collection.append([obj.posture.copy() for obj in self.solid_object_collection])
        self.velocity_matrix_collection.append([obj.velocity_matrix.copy() for obj in self.solid_object_collection])
        self.momentum_collection.append([obj.momentum.copy() for obj in self.solid_object_collection])
        self.orientation_collection.append([obj.orientation.copy() for obj in self.solid_object_collection])
        if hasattr(self, '_last_forces_k'):
            self.force_collection.append([f.copy() for f in self._last_forces_k])

def control_logic(time):
    input_force_left = 1.0
    input_force_right = 1.0
    return np.array([input_force_left, input_force_right])


