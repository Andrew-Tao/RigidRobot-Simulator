import numpy as np 
from methods2D import SE2LieAlgebra
import matplotlib.pyplot as plt
lie2 = SE2LieAlgebra()

class Simulator2D:
    def __init__(self, time_step = 0.1, duration = 10.0, control_logic: callable = None,stepper = 'explicit_euler'):
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
        


    def run(self):
        if len(self.time_collection) * self.time_step >= self.duration:
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
            posture_kp1 = posture_k @ lie2.exp(velocity_k_matrix * self.time_step)

            # Euler-Poincaré: μ_{k+1} = μ_k + (coad(V_k) · μ_k + F_k) · dt
            #TODO: Is this correct or not?
            momentum_kp1 = (lie2.exp(lie2.coadjoint(-velocity_k_matrix*self.time_step)) @ momentum_k) + (force_k * self.time_step)

          
        # -------------------Symplectic Euler Integration -------------------
        if self.stepper == 'symplectic_euler':

            momentum_kp1 = (lie2.exp(lie2.coadjoint(-velocity_k_matrix*self.time_step)) @ momentum_k) + (force_k * self.time_step)
            velocity_kp1_matrix = lie2.hat(np.linalg.inv(self.robot[0].mass_matrix) @ momentum_kp1)
            posture_kp1 = posture_k @ lie2.exp(velocity_kp1_matrix * self.time_step)
            
           

        #---------------------Position Verlet Integration----------------------
        if self.stepper == 'position_verlet':
           
           posture_k_phalf = posture_k @ lie2.exp(velocity_k_matrix * (self.time_step / 2))
           self.robot[0].posture = posture_k_phalf
           force_k_phalf = self.robot[0].compute_force_local(self.robot[0].control_input)
           momentum_kp1 = (lie2.exp(lie2.coadjoint(-velocity_k_matrix*self.time_step)) @ momentum_k) + (force_k_phalf * self.time_step)
           posture_kp1 = posture_k_phalf @ lie2.exp(velocity_k_matrix * (self.time_step / 2))

            # Recover velocity matrix: ξ_{k+1} = M⁻¹ · μ_{k+1}
        
        xi_kp1 = np.linalg.solve(self.robot[0].mass_matrix, momentum_kp1)

        self.robot[0].posture = posture_kp1
        self.robot[0].momentum = momentum_kp1
        self.robot[0].velocity_matrix = lie2.hat(xi_kp1)

    def record(self):
        self.time_collection.append(len(self.time_collection) * self.time_step)
        self.posture_collection.append(self.robot[0].posture.copy())
        self.velocity_matrix_collection.append(self.robot[0].velocity_matrix.copy())
        self.force_collection.append(self.robot[0].compute_force_local(self.robot[0].control_input).copy())
        self.momentum_collection.append(self.robot[0].momentum.copy())
        
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