import sys, os

# Allow running this file directly: `python SimulatorConnectedRobot3D.py`
# When imported as part of the package the relative imports work as-is.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "robot3d"

import numpy as np
from .methods3D import SE3LieAlgebra, rotation_matrix_to_euler_zyx
from .ConnectedRigidRobot3D import ConnectedRigidRobots3D
from tqdm import tqdm

lie3 = SE3LieAlgebra()

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
        self.internal_force_collection = []
        self.momentum_collection = []
        self.orientation_collection = []

        # Add for Debugging
        self.bending_internal_couple_collection = []
        self.shear_internal_couple_collection = []
        self.tau_x_base_collection = []
        self.strain_local_collection = []
        self.damping_couple_collection = []


        self.external_force_list = [] # Collection of External Forces
        self._last_forces_k = None
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
        posture_k_half_collection = np.zeros((n, 4, 4))

        # ------------Add Internal forces, adding to forces_k---------------

        for i in range(n):
            robot = self.connected_robot.robots[i]
            states_k.append((robot.momentum.copy(), robot.posture.copy(), robot.velocity_matrix.copy()))
            forces_k.append(self.connected_robot.compute_force_local_total_individual_robot(robot_index=i))

        #-------------Add External Forces, adding to forces_k---------------
        external_force_k = np.zeros((n, 6))
        for force_type in self.external_force_list:
            external_force_k += force_type.compute_force_collection(self.connected_robot, self.current_time)
        for i in range(n):
            forces_k[i] += external_force_k[i]

        self._last_forces_k = forces_k


        #---------------------Position Verlet Integration----------------------
        if self.stepper == 'position_verlet':
            # Phase 1: advance ALL robots to k+1/2 simultaneously
            for i in range(n):
                momentum_k, posture_k, velocity_k_matrix = states_k[i]
                posture_k_phalf = posture_k @ lie3.exp(velocity_k_matrix * (self.time_step / 2))
                self.connected_robot.robots[i].posture = posture_k_phalf
                posture_k_half_collection[i] = posture_k_phalf

            # Phase 2: compute ALL half-step forces with consistent k+1/2 postures
            forces_k_half = []
            for i in range(n):
                forces_k_half.append(self.connected_robot.compute_force_local_total_individual_robot(robot_index=i))

            # Phase 3: integrate ALL robots to k+1
            for i in range(n):
                robot = self.connected_robot.robots[i]
                momentum_k, posture_k, velocity_k_matrix = states_k[i]
                posture_k_phalf = posture_k_half_collection[i]
                force_k_phalf = forces_k_half[i]

                momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix * self.time_step) @ momentum_k) + (force_k_phalf * self.time_step)
                velocity_kp1_matrix = lie3.hat(robot.mass_matrix_inv @ momentum_kp1)
                posture_kp1 = posture_k_phalf @ lie3.exp(velocity_kp1_matrix * (self.time_step / 2))
                xi_kp1 = robot.mass_matrix_inv @ momentum_kp1

                robot.posture = posture_kp1
                robot.momentum = momentum_kp1
                robot.velocity_matrix = lie3.hat(xi_kp1)
                robot.orientation = rotation_matrix_to_euler_zyx(posture_kp1[:3, :3])

        else:
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
                elif self.stepper == 'symplectic_euler':
                    momentum_kp1 = (lie3.exp_adjoint(-velocity_k_matrix*self.time_step) @ momentum_k) + (force_k * self.time_step)
                    velocity_kp1_matrix = lie3.hat(robot.mass_matrix_inv @ momentum_kp1)
                    posture_kp1 = posture_k @ lie3.exp(velocity_kp1_matrix * self.time_step)

            
                xi_kp1 = robot.mass_matrix_inv @ momentum_kp1

                robot.posture = posture_kp1
                robot.momentum = momentum_kp1
                robot.velocity_matrix = lie3.hat(xi_kp1)
                robot.orientation = rotation_matrix_to_euler_zyx(posture_kp1[:3, :3])

        # TODO: This is just a test for a boundary condition
        # Apply boundary condition that the last robot always has the same orientation as the robot right before it for slender robot
        if n > 1:
            self.connected_robot.robots[-1].orientation = self.connected_robot.robots[-2].orientation.copy()
            print("I excuted the boundary condition")
    
        self.time_collection.append(self.current_time)

    def multi_robot_record(self):

        posture_frame = []
        velocity_matrix_frame = []
        force_frame = []
        momentum_frame = []
        orientation_frame = []

        bending_internal_couple_frame = []
        shear_internal_couple_frame = []
        tau_x_base_frame = []
        strain_local_frame = []
        damping_couple_frame = []

       
        for i in range(len(self.connected_robot.robots)):

            posture_frame.append(self.connected_robot.robots[i].posture.copy())
            velocity_matrix_frame.append(self.connected_robot.robots[i].velocity_matrix.copy())
            force_frame.append(self._last_forces_k[i].copy())
            momentum_frame.append(self.connected_robot.robots[i].momentum.copy())
            orientation_frame.append(self.connected_robot.robots[i].orientation.copy())
            bending_internal_couple_frame.append(self.connected_robot.bending_internal_couple.copy())
            shear_internal_couple_frame.append(self.connected_robot.shear_internal_couple.copy())
            tau_x_base_frame.append(self.connected_robot.tau_x_base)
            strain_local_frame.append(self.connected_robot.strain_local.copy())
            damping_couple_frame.append(self.connected_robot.damping_couple.copy())


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

        self.bending_internal_couple_collection.append(bending_internal_couple_frame)
        self.shear_internal_couple_collection.append(shear_internal_couple_frame)
        self.tau_x_base_collection.append(tau_x_base_frame)
        self.strain_local_collection.append(strain_local_frame)
        self.damping_couple_collection.append(damping_couple_frame)

        if self._pbar is not None:
            current_pct = int(len(self.posture_collection) / self._total_steps * 100)
            if current_pct > self._last_pct:
                self._pbar.update(current_pct - self._last_pct)
                self._last_pct = current_pct
        
        




