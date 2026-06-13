import numpy as np
from dataclasses import dataclass
from sympy import diagonalize_vector, print_tree
from cossart_rod import CossartRod


Q_history = [np.array([[1,0,0],[0,1,0],[0,0,1]]),
             np.array([[-0.866,0.5,0],[0.5,0.866,0],[0,0,1]]),]

def Q_matrix(d_vectors: list[np.ndarray]) -> np.ndarray:

    if len(d_vectors) != 3:
        raise ValueError("Expected exactly 3 arrays")
    vector_length = [np.linalg.norm(v) for v in d_vectors]
    vector_orthogonal = [np.dot(d_vectors[i%3], d_vectors[(i+2)%3]) for i in range(3)]

    assert all(x ==1 for x in vector_length)
    assert all(x == 0 for x in vector_orthogonal)

    Q = np.array([d_vectors[0], d_vectors[1], d_vectors[2]])

    return Q

def hat_map(skew_matrix: np.ndarray) -> np.ndarray:
    assert skew_matrix.shape == (3,3)
    assert np.allclose(skew_matrix.T, -skew_matrix, atol=1e-5)
    omega = np.array([skew_matrix[2,1], skew_matrix[0,2], skew_matrix[1,0]]).reshape(3,1)
    return omega

def vee_map(omega: np.ndarray) -> np.ndarray:
    assert omega.shape == (3,1)
    skew_matrix = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ])
    return skew_matrix

def omega(rod,time_step = 1) :
    dQ_dt = ((rod.Q_collection[2] - rod.Q_collection_last[2])/1)

    skew_temp = dQ_dt.T@rod.Q_collection[2]
    skew = 0.5 * (skew_temp - skew_temp.T)
    omega = hat_map(skew)
    return omega

def kappa(rod) :
    dQ_ds = ((rod.Q_collection[2] - rod.Q_collection[1])/(rod.lenth/rod.n_elem))

    skew_temp = dQ_ds.T@rod.Q_collection[2]
    skew = 0.5 * (skew_temp - skew_temp.T)
    kappa = hat_map(skew)
    return kappa




if __name__ == "__main__":

    #===============Initialization: Generate a twisted Q collection =======================
    rod = CossartRod(3)


    print("Bend matrix\n", rod.B,"\n", "Stiffness matrix\n", rod.S)
    print("R_collection\n",rod.r_collection)

    #===================Test: DO NOT RUN this section when not testing=======================
    """
    d1 = np.array([1, 0, 0])
    d2 = np.array([0, 1, 0])
    d3 = np.array([0, 0, 1])
    d_list = [d1, d2, d3]
    Q = Q_matrix(d_list)
    # print(Q)
    skew = np.array([
        [0, -2, 1],
        [2 , 0, -3],
        [-1, 3, 0]
    ])
    omega = hat_map(skew)
    print(omega.shape)
    vee_map(omega)
    """

    """
    def Q_after_rotation(theta):
        Q_after_xy_rotation = np.array([
            [np.cos(theta),np.sin(theta),0],
            [-np.sin(theta),np.cos(theta),0],
            [0,0,1]
        ])
        return Q_after_xy_rotation
        
    #print("The initial Q collection", rod.Q_collection)

    Q_collection_new = np.array([Q_after_rotation(0.25*i) for i in range(3)])

    rod.update_Q_collection(Q_collection_new)
    
    rod.update_Q_collection(Q_collection_new)
    
    #print("The updated Q collection", rod.Q_collection)
    #print("The last Q collection", rod.Q_collection_last)
    omega = omega(rod)
    #print(omega)
    """