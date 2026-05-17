import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot3d.RigidRobot3D import RigidRobot3D
from robot3d.Simulator3D import Simulator3D
import numpy as np
from robot3d.methods3D import SE3LieAlgebra
import matplotlib.pyplot as plt
lie3 = SE3LieAlgebra()

if __name__ == "__main__":

    robot = RigidRobot3D(
        position = np.array([0.0, 0.0, -1.0]),
        orientation = np.array([0.0, 0.0, 0.0]),
        mass = 1.0,
        inertia = np.array([0.1, 0.1, 0.1]),
        linear_velocity = np.zeros(3),
        angular_velocity = np.zeros(3),
        radius = 0.2,
        thickness = 0.2,
        track_width_between_wheels = 0.15,
    )


    simulator = Simulator3D(
        time_step = 0.1,
        duration = 100.0,
        control_logic = lambda t: -10,  # No control input
        stepper = 'position_verlet'
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
    

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(time_collection,position_colleiton_z, label='Z Position')
    plt.title('Z Position of the Robot Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.grid()
    plt.legend()
    plt.show()
