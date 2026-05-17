import numpy as np
from dataclasses import dataclass
from sympy import diagonalize_vector, print_tree
from sympy.codegen.ast import continue_

from cossart_rod import CossartRod
from methods import hat_map, vee_map, omega, kappa


class Rod(CossartRod):
    def __init__(self, n_elem: int):
        super().__init__(n_elem)
        self.sigma_collection = np.zeros((n_elem, 3))
        self.e_collection = np.array([1.0 for i in range(n_elem)])
        self.F_collection = np.array([np.array([0.0,0.0,0.0]) for i in range(n_elem+1)])
        self.a_collection = np.array([np.zeros(3) for i in range(n_elem+1)])
        self.l_collection = np.array([self.segment_length for i in range(n_elem)])
        self.D_collection = np.array([self.segment_length for i in range(n_elem)])
        self.dilatation_factor = np.array([1.0 for i in range(n_elem)])
        self.kappa_collection = np.array([np.array([0.0,0.0,0.0]) for i in range(n_elem)])
        self.t_collection = np.array([np.array([0.0,0.0,-1.0]) for i in range(n_elem)])

        self.omega_collection = np.array([np.array([0.0,0.0,0.0]) for i in range(n_elem)])
        self.de_dt_collection = np.array([0.0 for i in range(n_elem)])
        self.e_collection_last = self.e_collection.copy()
        self.C_collection = np.array([np.zeros(3) for i in range(n_elem)])
        self.alpha_collection = np.array([np.zeros(3) for i in range(n_elem)])

        self.start_time = 0
        self.end_time = 10
        self.step = int(10 / (0.1 * (0.5/self.n_elem) / 50)/100)
        self.dt = (self.end_time - self.start_time) / self.step
        self.current_time = self.start_time

        #======================History Collection===============
        self.r_history = np.array([])
        self.a_history = np.array([])

    def Verlet_stepper(self):
        # ==============Calculate acceleration from Eq(2.14) =========================
        for i in range(self.step):
            if i == -1:
                self.F_collection = np.array([np.array([0.0,0.0,0.0]) for i in range(self.n_elem+1)])
            if i == 500:
                break
            print(f"===================={i}===================")
            print(f"time ={i * self.dt}s")
            print("r",self.r_collection)

            #print("Q",self.Q_collection)
            #print("sigma",self.sigma_collection)
            #print("e",self.e_collection)

            #print("F",self.F_collection)

            #print("t",self.t_collection)
            #print("e_collection", self.e_collection, "t", self.t_collection)
            r_collection_next = np.array([np.array([0.0, 0.0, 0.0]) for x in range(self.n_elem + 1)])
            # Compute Linear and Angular Acceleration
            self.linear_acceleration_3dot8()
            print("a",self.a_collection)
            self.angular_acceleration_3dot9()
            #print("alpha",self.alpha_collection)

            r_collection_next = self.r_collection.copy()

            for j in range(self.n_elem+1):
                # ============== Update Position ================
                a = self.a_collection[j]
                r_collection_next[j] = 2 * self.r_collection[j] - self.r_collection_last[j] + (a * (self.dt ** 2))
                print(j,"r_collection",self.r_collection[j],"a",a,"dt",self.dt)
                print("r_collection_last",self.r_collection_last[j])
                print("r_collection_next",r_collection_next[j])
                print(" ")

                # Fix the End
                if j == 0:
                    r_collection_next[j] = np.array([0.0, 0.0, 0.0])

                if j == self.n_elem:
                    break
                # =============== Update Omega ====================
                alpha = self.alpha_collection[j]
                self.omega_collection[j] = self.omega_collection[j] + (self.dt * alpha)

            self.update_Q_matrix_3dot7()

            self.r_collection_last = self.r_collection.copy()
            self.r_collection = r_collection_next

            self.compute_l_and_D_collection()

            self.e_collection_last = self.e_collection.copy()
            self.compute_de_dt_collection()
            self.compute_e_collection()

            self.compute_kappa_collection()
            self.compute_dilatation_factor()
            self.compute_t_collection()
            self.compute_sigma_collection()

    def linear_acceleration_3dot8(self):
        #TODO: How you deal with the first element? See Appendix D
        #print("S",self.S)
        for i in range(self.n_elem+1):
            def internal_force(x):
                internal_force_i = (self.Q_collection[x].T @ self.S @ self.sigma_collection[x]) / self.e_collection[x]
                return internal_force_i

            #if i < self.n_elem: print("internal_force",internal_force(i))
            internal_term = self.discrete_delta(i, internal_force)
            #print("internal_term",internal_term)


            RHS = internal_term + self.F_collection[i]
            #print("RHS",RHS)
            #print(" ")
            self.a_collection[i] = RHS/ (self.rho * self.A * self.segment_length)

    def angular_acceleration_3dot9(self):
        #print(self.kappa_collection.shape)
        #print(self.dilatation_factor.shape)
        for i in range(self.n_elem):
            #===================bend-twist internal coupling ========================
            #print("dilatation",self.dilatation_factor)

            def bend_i(x):
                return ((self.B @ self.kappa_collection[x]) / (self.dilatation_factor[x]**3))
            def twist_i(x):
                return (np.cross((self.kappa_collection[x]),(self.B @ self.kappa_collection[x]))/(self.dilatation_factor[x]**3)) * self.D_collection[x]

            # TODO: now we assume each segment has the same length so we can use self.segment_legnth
            # TODO: for the D_hat, modified this later

            bend_term = self.discrete_delta(i,bend_i)
            twist_term = self.discrete_A(i,twist_i)
            #print("Bend", bend_term, "twist", twist_term)

            bend_twist_term = bend_term + twist_term
            #==================shear/stretch internal couple==========================
            shear_stretch_term = np.cross(self.Q_collection[i] @ self.t_collection[i], self.S @ self.sigma_collection[i]) * self.l_collection[i]
            #==========================Lagrangian transport============================
            Lag_temp = self.J @ (self.omega_collection[i] / self.e_collection[i])
            Lagragian_term = np.cross(Lag_temp, self.omega_collection[i])
            #==========================Unsteady dilatation=============================
            Unsteady_term = (self.J @ self.omega_collection[i]) / (self.e_collection[i]**2) * self.de_dt_collection[i]
            #=========================== Angular Acceleration===============================
            RHS = bend_twist_term + shear_stretch_term + Lagragian_term + Unsteady_term
            #print(Lagragian_term)
            self.alpha_collection[i] = np.linalg.inv(self.J) @ (RHS * self.e_collection[i])

    def update_Q_matrix_3dot7(self):
        for i in range(self.n_elem):
            Q_old = self.Q_collection[i].copy()
            for j in range(3):
                d_ij = Q_old[j]
                RHS = np.cross(Q_old.T @ self.omega_collection[i], d_ij)
                d_ij_new = d_ij + (self.dt * RHS)
                d_ij_new = d_ij_new / np.linalg.norm(d_ij_new)
                self.Q_collection[i,j] = d_ij_new

    def compute_strain(self):
        for i in range(self.n_elem):
            #TODO: Is this correct?
            dr_ds = (rod.r_collection[i+1] - rod.r_collection[i]) / self.segment_length
            d3 = rod.Q_collection[i,2]
            #print("dr_ds", dr_ds)
            sigma = rod.Q_collection[i] @ (dr_ds - d3)
            self.sigma_collection[i] = sigma

    def compute_t_collection(self):
        for i in range(self.n_elem):
            t_temp = rod.r_collection[i+1] - rod.r_collection[i]
            #print("temp",t_temp)
            t_temp = t_temp / np.linalg.norm(t_temp)
            self.t_collection[i] = t_temp

    def compute_e_collection(self):
        for i in range(self.n_elem):
            dr_ds = (rod.r_collection[i + 1] - rod.r_collection[i]) / self.segment_length
            #print(rod.r_collection)
            #print("drds",dr_ds)

            e = np.sum(dr_ds * self.t_collection[i]) / np.sum(self.t_collection[i] * self.t_collection[i])
            self.e_collection[i] = e
            #print("dr_ds",dr_ds)
            #print("t",self.t_collection[i])
            self.e_collection[i] = e
            #print("e",e)

    def compute_l_and_D_collection (self):
        #print("D",self.D_collection)
        for i in range(self.n_elem):
            #TODO: Is this the correct way to deal with the last element
            if i == self.n_elem-1:
                self.D_collection[i] = self.D_collection[i-1]
                self.l_collection[i] = self.l_collection[i-1]
            else:
                l_i = np.linalg.norm(self.r_collection[i+1] - self.r_collection[i])
                l_i_plus_1 = np.linalg.norm(self.r_collection[i+2] - self.r_collection[i+1])
                D_i = (l_i_plus_1 + l_i)/2
                self.D_collection[i] = D_i
                self.l_collection[i] = l_i

    def compute_dilatation_factor(self):
        for i in range(self.n_elem):
            #TODO: now we assume each segment has the same length so we can use self.segment_legnth
            #TODO: for the D_hat, modified this later

            dilatation_factor_i = self.D_collection[i] / self.segment_length
            #print("dilatation_factor",dilatation_factor_i)
            self.dilatation_factor[i] = dilatation_factor_i

    def compute_kappa_collection(self):
        for i in range(self.n_elem):
            ka = self.kappa(i).reshape(1,3)[0]
            #print(ka)
            self.kappa_collection[i] = ka

    def compute_sigma_collection(self):
        for i in range(self.n_elem):
            # Equation 2.7
            sigma_l = self.Q_collection[i] @ (self.e_collection[i] * self.t_collection[i] - self.Q_collection[i,2])
            self.sigma_collection[i] = sigma_l

    def compute_de_dt_collection(self):
        self.de_dt_collection = (self.e_collection- self.e_collection_last)/self.dt





if __name__ == "__main__":
    # ===============Initialization: Generate a twisted Q collection =======================
    rod = Rod(10)

    #print("Bend matrix\n", rod.B, "\n", "Stiffness matrix\n", rod.S)
    #print("R_collection\n", rod.r_collection)

    #print(rod.sigma_collection)
    #print(rod.e_collection)
    #rod.linear_acceleration_3dot8()
    #print(rod.a_collection)
    #rod.compute_e_collection()

    # ==================Initialization ==============

    rod.rho = 1000
    rod.E = 1.2e7
    rod.G = 1.2e7 / 2
    rod.A = 0.01 * 0.01
    rod.S = rod.compute_S_matrix()
    rod.J = rod.compute_J_matrix()
    rod.B = rod.compute_B_matrix()
    #print(rod.G)
    #print(rod.S)

    rod.e_collection_last = rod.e_collection.copy()
    rod.compute_de_dt_collection()
    rod.compute_e_collection()

    rod.compute_kappa_collection()
    rod.compute_dilatation_factor()
    rod.compute_t_collection()
    rod.compute_sigma_collection()

    rod.F_collection[-1] = np.array([0.0,0.0,-10.0])
    #print("F",rod.F_collection)


    rod.Verlet_stepper()

    from visualizor import visualize_rod
    visualize_rod(rod.r_collection)

