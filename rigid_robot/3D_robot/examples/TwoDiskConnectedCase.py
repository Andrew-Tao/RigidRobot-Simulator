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
    pass
   


