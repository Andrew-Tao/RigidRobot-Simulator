import numpy as np
from dataclasses import dataclass
from sympy import diagonalize_vector, print_tree


class CossartRod:
    def __init__(self, n_elem: int):
        self.length = 0.5
        self.n_elem = n_elem
        self.Q_collection = self.compute_Q_collection()
        self.Q_collection_last = self.Q_collection.copy()
        self.r_collection = self.compute_r_collection()
        self.r_collection_last = self.r_collection.copy()
        #TODO: Move the time_step out of Rod Class to the stepper
        self.time_step = 0.1
        self.segment_length = self.length / self.n_elem

        # =============== Matrial and geometry properties =================
        #Assume Steel
        self.rho = 8000
        self.E = 2 * 1e9
        self.G = 80 * 10e6
        self.A = 1e-4

        #TODO: Know all the elements shared the same S and B matrix, extent to a list for each elements
        self.S =  self.compute_S_matrix()
        self.B = self.compute_B_matrix()
        self.J = self.compute_J_matrix()


    #TODO: BUGGGGGGGGGG! Q elements not nomalized

    def compute_Q_collection(self) -> np.ndarray:
        Q_0 = np.array([
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,-1.0],
        ])
        Q_collection = np.array([ Q_0 for i in range(self.n_elem)])
        return Q_collection

    def update_Q_collection(self, Q_collection_new: np.ndarray) -> None:
        self.Q_collection_last =self.Q_collection.copy()
        self.Q_collection = Q_collection_new.copy()


    def compute_S_matrix(self) -> np.ndarray:
        return self.A * np.array([
            [(4/3)*self.G, 0.0, 0.0],
            [0.0, (4/3)*self.G, 0.0],
            [0.0, 0.0, (4/3)*self.E]
        ])

    def compute_B_matrix(self) -> np.ndarray:
        property_matrix = np.array([
            [ self.E, 0.0, 0.0],
            [0.0,  self.E, 0.0],
            [0.0, 0.0,  self.G]
        ])
        Ix = (np.pi / 4) * ((self.A / np.pi)**2)
        Iy = (np.pi / 4) * ((self.A / np.pi)**2)
        Iz = (np.pi / 2) * ((self.A / np.pi)**2)
        I = np.diag([Ix, Iy, Iz])
        #print(I)
        B = property_matrix @ I
        return B
    def compute_J_matrix(self) -> np.ndarray:
        Ix = (np.pi / 4) * ((self.A / np.pi) ** 2)
        Iy = (np.pi / 4) * ((self.A / np.pi) ** 2)
        Iz = (np.pi / 2) * ((self.A / np.pi) ** 2)
        I = np.diag([Ix, Iy, Iz])
        J = I * self.rho
        return J

    def compute_r_collection(self) -> np.ndarray:
        # A straight rod anchored at origin and extend in negative z direction
        segment_length = self.length / (self.n_elem)
        r_collection = np.array([ np.array([0.0,0.0,-i*segment_length]) for i in range(self.n_elem+1)])
        return r_collection

    def compute_e_collection(self):
        # Initial e values are all 1 (no elongation)
        e_collection = np.array([ 1 for i in range(self.n_elem)])

    def kappa(self,i):
        dQ_ds = ((self.Q_collection[i] - self.Q_collection[i-1]) / (self.length / self.n_elem))

        skew_temp = dQ_ds.T @ self.Q_collection[i]
        skew = 0.5 * (skew_temp - skew_temp.T)
        kappa = self.hat_map(skew)
        return kappa

    def hat_map(self,skew_matrix: np.ndarray) -> np.ndarray:
        assert skew_matrix.shape == (3, 3)
        assert np.allclose(skew_matrix.T, -skew_matrix, atol=1e-5)
        omega = np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]]).reshape(3, 1)
        return omega

    def vee_map(self, omega: np.ndarray) -> np.ndarray:
        assert omega.shape == (3, 1)
        skew_matrix = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        return skew_matrix

    def discrete_delta(self, i, quantity):
        #print(i)
        if i == 0:
            return quantity(i)
        if i > 0 and i <= self.n_elem-1:
            return quantity(i) - quantity(i-1)
        if i == self.n_elem:
            return -quantity(i-1)

    def discrete_A(self, i, quantity):
        if i == 0:
            return quantity(i)/2
        if i > 0 and i <= self.n_elem-1:
            return (quantity(i) - quantity(i-1))/2
        if i == self.n_elem:
            return quantity(i-1)/2

