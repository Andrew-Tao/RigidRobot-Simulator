import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.RigidRobot3D import RigidRobot3D
from robot3d.Simulator3D import Simulator3D
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

lie3 = SE3LieAlgebra()

if __name__ == "__main__":

    # TPU material parameters density = 1210 kg/m^3, Young's modulus = 25 MPa, Poisson's ratio = 0.45
    # r = 0.03 m, h = 0.025 m Disk

    robot = RigidRobot3D(
        position = np.array([0.0, 0.0, -0.04]),
        orientation = np.array([0.0, 0.0, 1.0]),
        mass = 0.0855,
        inertia = np.array([2.37e-5, 2.37e-5, 3.84e-5]), # Ixx = Iyy = (1/12) * m *( 3r^2 + h^2), Izz = (1/2) * m * r^2
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.03,
        thickness = 0.025,
        track_width_between_wheels = 0.15,# This is useless 
    )

    def control_logic(time):
        # Apply an upward force for the first 5 seconds, then no control input
        if time >0.0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        else:
            return np.zeros(6)  # No control input
        
    simulator = Simulator3D(
        time_step = 0.1,
        duration = 3.0,
        control_logic = control_logic,  # No control input
        stepper = 'explicit_euler',  
    )

    simulator.attach(robot)
    while simulator.run():
        simulator.step()
        simulator.record()

    # Data collection (place holder for now)
    time_collection = np.array(simulator.time_collection)
    posture_collection = np.array(simulator.posture_collection)  # Shape: (num_steps, 4, 4)
    position_over_time = posture_collection[:, :3, 3]  # Extract position (x, y, z) over time
    position_colleiton_x = position_over_time[:, 0]
    position_colleiton_y = position_over_time[:, 1]
    position_colleiton_z = position_over_time[:, 2]
    theta = np.array([R.from_matrix(posture_collection[i, :3, :3]).as_euler('xyz') for i in range(len(posture_collection))])  # Extract orientation (roll, pitch, yaw) over time
    theta_x = theta[:, 0]
    theta_y = theta[:, 1]
    theta_z = theta[:, 2]

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(time_collection,theta, label='Z Rotation (Yaw)')
    plt.title('Z Rotation of the Robot Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Rotation (rad)')
    plt.grid()
    plt.legend()
    plt.show()
