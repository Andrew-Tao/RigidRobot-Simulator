import numpy as np
from .methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from .RigidRobot3D import ConnectedRigidRobots3D
from tqdm import tqdm

lie3 = SE3LieAlgebra()

class Simulator3D:
    def __init__(self, time_step = 0.1, duration = 10.0, control_logic: callable = None, stepper = 'explicit_euler', show_progress = True):
        self.robot = list()  # List of robots in the simulation
        self.time_step = time_step
        self.duration = duration
        self.control_logic = control_logic
        self.stepper = stepper
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

    def run(self):
        if len(self.time_collection) * self.time_step >= self.duration:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            return False  # Simulation finished
        return True

    def attach(self, robot):
        self.robot.append(robot)

    def step(self):
        # Explicit Euler integration Method
        self.robot[0].control_input = self.control_logic(len(self.time_collection) * self.time_step)
        momentum_k = self.robot[0].momentum
        posture_k = self.robot[0].posture
        velocity_k_matrix = self.robot[0].velocity_matrix
        force_k = self.robot[0].compute_force_local(self.robot[0].control_input)


        # -------------------Explicit Euler Integration -------------------

        if self.stepper == 'explicit_euler':
            # Kinematics: T_{k+1} = T_k @ exp(ξ_k · dt)
            posture_kp1 = posture_k @ lie3.exp(velocity_k_matrix * self.time_step)
            # Euler-Poincaré: μ_{k+1} = μ_k + (coad(V_k) · μ_k + F_k) · dt
            momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k * self.time_step)

        # -------------------Symplectic Euler Integration -------------------
        if self.stepper == 'symplectic_euler':

            momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k * self.time_step)
            velocity_kp1_matrix = lie3.hat(np.linalg.inv(self.robot[0].mass_matrix) @ momentum_kp1)
            posture_kp1 = posture_k @ lie3.exp(velocity_kp1_matrix * self.time_step)
            
        #---------------------Position Verlet Integration----------------------
        if self.stepper == 'position_verlet':
           
           posture_k_phalf = posture_k @ lie3.exp(velocity_k_matrix * (self.time_step / 2))
           self.robot[0].posture = posture_k_phalf
           force_k_phalf = self.robot[0].compute_force_local(self.robot[0].control_input)
           momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k_phalf * self.time_step)
           posture_kp1 = posture_k_phalf @ lie3.exp(velocity_k_matrix * (self.time_step / 2))

            # Recover velocity matrix: ξ_{k+1} = M⁻¹ · μ_{k+1}
        
        xi_kp1 = np.linalg.solve(self.robot[0].mass_matrix, momentum_kp1)

        self.robot[0].posture = posture_kp1
        self.robot[0].momentum = momentum_kp1
        self.robot[0].velocity_matrix = lie3.hat(xi_kp1)
        velocity_temp = np.array([self.robot[0].velocity_matrix[2, 1], self.robot[0].velocity_matrix[0, 2], self.robot[0].velocity_matrix[1, 0]]) # Angular Velocity
        delta_orientation = velocity_temp * self.time_step
        self.robot[0].orientation = self.robot[0].orientation + delta_orientation

    def record(self):
        self.time_collection.append(len(self.time_collection) * self.time_step)
        self.posture_collection.append(self.robot[0].posture.copy())
        self.velocity_matrix_collection.append(self.robot[0].velocity_matrix.copy())
        self.force_collection.append(self.robot[0].compute_force_local(self.robot[0].control_input).copy())
        self.momentum_collection.append(self.robot[0].momentum.copy())
        self.orientation_collection.append(self.robot[0].orientation.copy())
        if self._pbar is not None:
            current_pct = int(len(self.time_collection) / self._total_steps * 100)
            if current_pct > self._last_pct:
                self._pbar.update(current_pct - self._last_pct)
                self._last_pct = current_pct

def boundary(position: np.ndarray) -> bool:
    x, y = position
    in_bottom = (0 <= x <= 3)   and (0 <= y <= 1)
    in_middle = (0.6 <= x <= 2.4) and (1 <= y <= 3)
    in_top    = (0 <= x <= 3)   and (3 <= y <= 4)
    return bool(in_bottom or in_middle or in_top)

def control_logic(time):
    input_force_left = 1.0
    input_force_right = 1.0
    return np.array([input_force_left, input_force_right])


class MutiRobotSimulator3D():
    def __init__(
        self,
        time_step: float,
        duration: float,
        stepper,
        control_logic = None,
        show_progress = True,
        ):

        self.connected_robot = None
        self.time_step = time_step
        self.duration = duration
        self.control_logic = control_logic
        self.stepper = stepper
        self.external_force = [] # Collection of external force funciton

        self.time_collection = []  # To store time points
        self.posture_collection = []
        self.velocity_matrix_collection = []
        self.force_collection = []
        self.momentum_collection = []
        self.orientation_collection = []

        self.external_force_list = [] # Collection of External Forces
        self.current_time = 0.0
        self._total_steps = int(duration / time_step)
        self._last_pct = 0
        self._pbar = tqdm(total=100, desc="Simulating", unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]") if show_progress else None
        
    
    def attach(self,robot: ConnectedRigidRobots3D):
        self.connected_robot = robot
    
    def run(self):
        if len(self.time_collection) * self.time_step >= self.duration:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            return False  # Simulation finished
        return True

    def add_external_force(self,force_type):
        self.external_force_list.append(force_type)


    def multi_robots_step(self):
        self.current_time = len(self.time_collection) * self.time_step
        n = len(self.connected_robot.robots)

        # Gather all forces and states at time k before updating any robot
        states_k = []
        forces_k = []

        for i in range(n):
            robot = self.connected_robot.robots[i]
            states_k.append((robot.momentum.copy(), robot.posture.copy(), robot.velocity_matrix.copy()))
            forces_k.append(self.connected_robot.compute_force_local_total_individual_robot(robot_index=i, external_force=np.zeros(6)))

        # Add External Forces, adding to forces_k
        external_force_k = np.zeros((n, 6))  # Bug 9 fixed: np.zeros needs a tuple shape
        for force_type in self.external_force_list:  # Bug 10 fixed: iterate over the correct list
            external_force_k += force_type.compute_force_collection(self.connected_robot, self.current_time)  # Bug 11 fixed: pass slender_robot argument
        for i in range(n):
            forces_k[i] += external_force_k[i]
            

        # Now integrate all robots using only time-k states
        for i in range(n):
            robot = self.connected_robot.robots[i]
            momentum_k, posture_k, velocity_k_matrix = states_k[i]
            force_k = forces_k[i]

            # -------------------Explicit Euler Integration -------------------
            if self.stepper == 'explicit_euler':
                posture_kp1 = posture_k @ lie3.exp(velocity_k_matrix * self.time_step)
                momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k * self.time_step)

            # -------------------Symplectic Euler Integration -------------------
            if self.stepper == 'symplectic_euler':
                momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k * self.time_step)
                velocity_kp1_matrix = lie3.hat(np.linalg.inv(robot.mass_matrix) @ momentum_kp1)
                posture_kp1 = posture_k @ lie3.exp(velocity_kp1_matrix * self.time_step)

            #---------------------Position Verlet Integration----------------------
            if self.stepper == 'position_verlet':
                posture_k_phalf = posture_k @ lie3.exp(velocity_k_matrix * (self.time_step / 2))
                robot.posture = posture_k_phalf
                force_k_phalf = robot.compute_force_local_total_individual_robot(robot.control_input)
                momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k_phalf * self.time_step)
                posture_kp1 = posture_k_phalf @ lie3.exp(velocity_k_matrix * (self.time_step / 2))

            xi_kp1 = np.linalg.solve(robot.mass_matrix, momentum_kp1)

            robot.posture = posture_kp1
            robot.momentum = momentum_kp1
            robot.velocity_matrix = lie3.hat(xi_kp1)
            velocity_temp = np.array([robot.velocity_matrix[2, 1], robot.velocity_matrix[0, 2], robot.velocity_matrix[1, 0]])
            delta_orientation = velocity_temp * self.time_step
            robot.orientation = robot.orientation + delta_orientation

        self.time_collection.append(self.current_time)

    def multi_robot_record(self):

        posture_frame = []
        velocity_matrix_frame = []
        force_frame = []
        momentum_frame = []
        orientation_frame = []

        for i in range(len(self.connected_robot.robots)):
           
            posture_frame.append(self.connected_robot.robots[i].posture.copy())
            velocity_matrix_frame.append(self.connected_robot.robots[i].velocity_matrix.copy())
            force_frame.append(self.connected_robot.compute_force_local_total_individual_robot(robot_index=i, external_force=np.zeros(6)).copy())
            momentum_frame.append(self.connected_robot.robots[i].momentum.copy())
            orientation_frame.append(self.connected_robot.robots[i].orientation.copy())


        posture_frame = np.array(posture_frame)
        velocity_matrix_frame = np.array(velocity_matrix_frame)
        force_frame = np.array(force_frame)
        momentum_frame = np.array(momentum_frame)
        orientation_frame = np.array(orientation_frame)


        self.posture_collection.append(posture_frame)
        self.velocity_matrix_collection.append(velocity_matrix_frame)
        self.force_collection.append(force_frame)
        self.momentum_collection.append(momentum_frame)
        self.orientation_collection.append(orientation_frame)
        if self._pbar is not None:
            current_pct = int(len(self.posture_collection) / self._total_steps * 100)
            if current_pct > self._last_pct:
                self._pbar.update(current_pct - self._last_pct)
                self._last_pct = current_pct
        
        




