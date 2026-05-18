import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

posture_i = np.array([
            [-0.73507489 ,-0.67798592,  0.,          0.        ],
            [ 0.67798592 , -0.73507489,  0. ,         0.        ],
            [ 0.         , 0.    ,      1.  ,       -0.04      ],
            [ 0.        , 0.       ,   0.    ,      1.        ]])
A = np.array([[ 0.00829623, -0.99996559,  0.      ,    0.        ],
                [ 0.99996559,  0.00829623,  0.       ,   0.        ],
                [ 0.         , 0.         , 1.     ,     0.        ],
                [ 0.         , 0.         , 0.       ,   1.        ]])

posture_collection = []
A_collection = []
for i in range(80):
    posture_i = posture_i @ A
    posture_collection.append(posture_i)
    A_collection.append(posture_i[0,1])

posture_collection = np.array(posture_collection)
A_collection = np.array(A_collection)
print(A_collection)

theta = np.array([R.from_matrix(posture_collection[i, :3, :3]).as_euler('xyz') for i in range(len(posture_collection))])  # Extract orientation (roll, pitch, yaw) over time
theta_x = theta[:, 0]
theta_y = theta[:, 1]
theta_z = theta[:, 2]

plt.plot(A_collection, label='roll')
plt.show()

