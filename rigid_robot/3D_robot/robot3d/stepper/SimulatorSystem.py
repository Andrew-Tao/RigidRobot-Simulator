import sys, os

# Allow running this file directly: `python SimulatorConnectedRobot3D.py`
# When imported as part of the package the relative imports work as-is.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "robot3d.stepper"

import numpy as np
from ..methods3D import SE3LieAlgebra, rotation_matrix_to_euler_zyx
from ..robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from tqdm import tqdm
from ..contact.solidcontact import SolidObject 
from SimulatorConnectedRobot3D import MutiRobotSimulator3D
from Simulator3D import Simulator3D
lie3 = SE3LieAlgebra()
#TODO: Modify this to make it a simualtion two kind of asset slender robot and solid object 



class SystemSimulator3D():
    def __init__(
        self,
        time_step: float,
        duration: float,
        stepper,
        control_logic = None,
        show_progress = True,
        ):

        # Asset Collection
        self.connected_robot = None
        self.solid_object = []

        # Simualator
        slender_robot_simulator = MutiRobotSimulator3D(time_step, duration, stepper, control_logic= None, show_progress= False)
        solid_object_simulator = Simulator3D(time_step, duration, stepper, control_logic= None, show_progress= False)
        self.simulator_colletion = {"SlenderRobotSimulator": slender_robot_simulator, "SolidObejctSimulator": solid_object_simulator }

        # Simulation Prameter
        self.time_step = time_step
        self.duration = duration
        self.control_logic = control_logic
        self.stepper = stepper
        self.external_force = [] # Collection of external force funciton

        # Simulation CallBack Collection
        self.time_collection = []  # To store time points
        self._pbar = tqdm(total=100, desc="Simulating", unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]") if show_progress else None
        
    
    def attach(self, asset: ConnectedRigidRobots3D | SolidObject) -> None:
        if isinstance(asset, ConnectedRigidRobots3D): self.connected_robot = asset
        if isinstance(asset, SolidObject): self.solid_object.append(asset)

    def run(self):
        if len(self.time_collection) * self.time_step >= self.duration:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            return False  # Simulation finished
        return True

    def add_external_force(self,force_type):
        self.simulator_colletion["SlenderRobotSimulator"].add_external_force(force_type)
        self.simulator_colletion["SolidObjectSimulator"].add_external_force(force_type)
        # TODO: Some force type shoould only be appied to some robot like cable driven force 

    def step(self):

        # TODO: DON'T update slnder robot for now later should have state variable that esure parallel update
 
        # Update

        self.simulator_colletion["SolidObjectSimulator"].step()

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
        
        




