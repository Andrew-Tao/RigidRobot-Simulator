import sys, os

# Allow running this file directly: `python SimulatorConnectedRobot3D.py`
# When imported as part of the package the relative imports work as-is.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "robot3d"
    
from .methods3D import SE3LieAlgebra, rpy_to_Q, rotation_matrix_to_euler_zyx
import numpy as np
orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]) # This orientation is just a place holder and won't be used

this_orientation = rotation_matrix_to_euler_zyx(orientation)
print("Orientation in RPY:\n", this_orientation)
