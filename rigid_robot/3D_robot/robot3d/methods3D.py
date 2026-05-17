import numpy as np


class SE3LieAlgebra:
    """
    SE(3) Lie group / se(3) Lie algebra utilities.

    Convention: xi = [v1, v2, v3, w1, w2, w3] in R^6  (linear first, angular second)
    T in SE(3) is a 4x4 homogeneous matrix:
        [ R  t ]
        [ 0  1 ]
    where R in SO(3), t in R^3.

    se(3) matrix (hat form):
        Xi = [[ w×,  v ],
              [ 0^T, 0 ]]
    where w× is the 3x3 skew-symmetric matrix of w.
    """

    _EPS = 1e-8

    # ------------------------------------------------------------------ #
    # Internal helper                                                      #
    # ------------------------------------------------------------------ #

    def _skew(self, w: np.ndarray) -> np.ndarray:
        """3-vector -> 3x3 skew-symmetric matrix."""
        return np.array([[    0, -w[2],  w[1]],
                         [ w[2],     0, -w[0]],
                         [-w[1],  w[0],     0]], dtype=float)

    # ------------------------------------------------------------------ #
    # Algebra operators                                                    #
    # ------------------------------------------------------------------ #

    def hat(self, xi: np.ndarray) -> np.ndarray:
        """Map R^6 -> se(3): vector to 4x4 skew matrix."""
        v1, v2, v3, w1, w2, w3 = xi
        return np.array([[  0, -w3,  w2, v1],
                         [ w3,   0, -w1, v2],
                         [-w2,  w1,   0, v3],
                         [  0,   0,   0,  0]], dtype=float)

    def vee(self, Xi: np.ndarray) -> np.ndarray:
        """Map se(3) -> R^6: 4x4 matrix to vector (inverse of hat)."""
        return np.array([Xi[0, 3], Xi[1, 3], Xi[2, 3],   # v1, v2, v3
                         Xi[2, 1], Xi[0, 2], Xi[1, 0]])  # w1, w2, w3

    def bracket(self, xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
        """Lie bracket [xi1, xi2] = vee(hat(xi1) @ hat(xi2) - hat(xi2) @ hat(xi1))."""
        return self.vee(self.hat(xi1) @ self.hat(xi2) - self.hat(xi2) @ self.hat(xi1))

    # ------------------------------------------------------------------ #
    # Group operators                                                      #
    # ------------------------------------------------------------------ #

    def inverse(self, T: np.ndarray) -> np.ndarray:
        """Inverse of T in SE(3): T^{-1} = [R^T, -R^T t; 0, 1]."""
        R = T[:3, :3]
        t = T[:3,  3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3,  3] = -R.T @ t
        return T_inv

    # ------------------------------------------------------------------ #
    # Exponential and logarithmic maps                                     #
    # ------------------------------------------------------------------ #

    def exp(self, Xi: np.ndarray) -> np.ndarray:
        """
        Exponential map se(3) matrix -> SE(3).

        Xi = [[ w×,  v ],
              [ 0^T, 0 ]]
        """
        v     = Xi[:3, 3]
        W     = Xi[:3, :3]                                          # skew(w)
        theta = np.sqrt(W[2, 1]**2 + W[0, 2]**2 + W[1, 0]**2)    # |w|

        T = np.eye(4)
        if theta < self._EPS:
            T[:3, :3] = np.eye(3) + W
            T[:3,  3] = (np.eye(3) + 0.5 * W) @ v
        else:
            W2   = W @ W
            s, c = np.sin(theta), np.cos(theta)
            R    = np.eye(3) + (s / theta) * W + ((1 - c) / theta**2) * W2
            V    = np.eye(3) + ((1 - c) / theta**2) * W + ((theta - s) / theta**3) * W2
            T[:3, :3] = R
            T[:3,  3] = V @ v

        return T
    
    def exp_SO3(self, W: np.ndarray) -> np.ndarray:
        """Exponential map for SO(3) rotation matrix."""
        theta = np.sqrt(W[2, 1]**2 + W[0, 2]**2 + W[1, 0]**2)
        if theta < self._EPS:
            return np.eye(3) + W
        else:
            W2   = W @ W
            s, c = np.sin(theta), np.cos(theta)
            R    = np.eye(3) + (s / theta) * W + ((1 - c) / theta**2) * W2
            return R

    def log(self, T: np.ndarray) -> np.ndarray:
        """
        Logarithmic map SE(3) -> se(3) matrix.

        Returns Xi = [[ w×,  v ],
                      [ 0^T, 0 ]]
        """
        R = T[:3, :3]
        t = T[:3,  3]

        cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        theta     = np.arccos(cos_theta)

        Xi = np.zeros((4, 4))
        if theta < self._EPS:
            W     = (R - R.T) / 2
            V_inv = np.eye(3) - W / 2
        else:
            W     = (theta / (2 * np.sin(theta))) * (R - R.T)
            W2    = W @ W
            coeff = (2 * np.sin(theta) - theta * (1 + np.cos(theta))) \
                    / (2 * theta**2 * np.sin(theta))
            V_inv = np.eye(3) - W / 2 + coeff * W2

        Xi[:3, :3] = W
        Xi[:3,  3] = V_inv @ t
        return Xi

    # ------------------------------------------------------------------ #
    # Adjoint representations                                              #
    # ------------------------------------------------------------------ #

    def ad(self, Xi: np.ndarray) -> np.ndarray:
        """
        Lie algebra adjoint operator ad_xi from a 4x4 se(3) matrix Xi = hat(xi).

        Xi = [[hat(w), v],
            [0^T,    0]]

        Uses twist ordering xi = [w, v].
        """
        v = Xi[:3, 3]
        W = Xi[:3, :3]  # hat(w)

        ad = np.zeros((6, 6))
        ad[:3, :3] = W
        ad[3:, :3] = self._skew(v)
        ad[3:, 3:] = W

        return ad

    def coadjoint(self, Xi: np.ndarray) -> np.ndarray:
        """
        Co-adjoint ad_xi^* = ad_xi^T from a 4x4 se(3) matrix Xi = hat(xi).
        Used in the Euler-Poincaré equation: dμ/dt = coad(Xi) μ + F

        With twist ordering xi = [v, w] (linear first, angular second):

            ad  = [[ W,      0   ],      coad = ad^T = [[ W^T,   hat(v)^T ],
                   [ hat(v), W   ]]                     [ 0,      W^T      ]]

        where W = hat(w), so W^T = -W.
        """
        v = Xi[:3, 3]
        W = Xi[:3, :3]  # hat(w), skew-symmetric
        
        coad = np.zeros((6, 6))
        coad[:3, :3] = W.T
        coad[:3, 3:] = self._skew(v).T
        coad[3:, 3:] = W.T
        return coad
    

    def exp_adjoint(self, Xi: np.ndarray) -> np.ndarray:
   
        v = Xi[:3, 3]
        W = Xi[:3, :3]  # hat(w), skew-symmetric
       
        W_negative = -W
        W2 = W_negative @ W_negative
        theta = np.sqrt(W_negative[2, 1]**2 + W_negative[0, 2]**2 + W_negative[1, 0]**2)
        s, c = np.sin(theta), np.cos(theta)

        if theta < self._EPS:
            R = np.eye(3) 
        else:
            R   = np.eye(3) + ((1 - c) / theta**2) * W + ((theta - s) / theta**3) * W2

        exp_ad = np.zeros((6, 6))
        exp_ad[:3, :3] =  self.exp_SO3(-W)

        exp_ad[3:, :3] = self.exp_SO3(-W) @ R @ (-self._skew(v))
        exp_ad[3:, 3:] = self.exp_SO3(-W)

        return exp_ad

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """SO(3) rotation matrix via Rodrigues' formula (axis need not be unit)."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        W    = self._skew(axis)
        c, s = np.cos(angle), np.sin(angle)
        return np.eye(3) + s * W + (1 - c) * (W @ W)

    def compose(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """Group composition T1 @ T2."""
        return T1 @ T2
    
    import numpy as np

# Additional utility function for converting roll-pitch-yaw to rotation matrix
def rpy_to_Q(orientation):
    phi = orientation[0]
    theta = orientation[1]
    psi = orientation[2]

    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Q = np.array([
        [
            cpsi * ctheta,
            cpsi * stheta * sphi - spsi * cphi,
            cpsi * stheta * cphi + spsi * sphi
        ],
        [
            spsi * ctheta,
            spsi * stheta * sphi + cpsi * cphi,
            spsi * stheta * cphi - cpsi * sphi
        ],
        [
            -stheta,
            ctheta * sphi,
            ctheta * cphi
        ]
    ])

    return Q
