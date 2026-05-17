import numpy as np
import matplotlib.pyplot as plt

class RigidRobot:
    def __init__(self, initial_position, initial_velocity, mass):
        self.position = np.array(initial_position, dtype=float).reshape(3, 1)
        self.velocity = np.array(initial_velocity, dtype=float).reshape(3, 1)
        self.mass = mass
        self.rotation = np.eye(3)

    @property
    def state(self):
        """4x4 homogeneous transformation matrix [R | p; 0 | 1]."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3:] = self.position
        return T

    def apply_force(self, force, time_step):
        acceleration = np.array(force, dtype=float).reshape(3, 1) / self.mass
        self.velocity += acceleration * time_step

    def update_position(self, time_step):
        self.position += self.velocity * time_step

    def simulate(self, forces, time_step, total_time):
        positions = []
        for t in np.arange(0, total_time, time_step):
            if t < len(forces) * time_step:
                self.apply_force(forces[int(t / time_step)], time_step)
            self.update_position(time_step)
            positions.append(self.position.copy())
        return positions
    

