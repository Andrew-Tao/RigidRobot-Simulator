import numpy as np
from numba import njit

_EPS = 1e-8


@njit(cache=True)
def _hat_jit(xi: np.ndarray) -> np.ndarray:
    v1, v2, v3, w1, w2, w3 = xi
    return np.array([[ 0.0, -w3,  w2, v1],
                     [  w3, 0.0, -w1, v2],
                     [ -w2,  w1, 0.0, v3],
                     [ 0.0, 0.0, 0.0, 0.0]])


@njit(cache=True)
def _exp_jit(Xi: np.ndarray) -> np.ndarray:
    v     = np.ascontiguousarray(Xi[:3, 3])
    W     = np.ascontiguousarray(Xi[:3, :3])
    theta = np.sqrt(W[2, 1]**2 + W[0, 2]**2 + W[1, 0]**2)

    T = np.eye(4)
    if theta < _EPS:
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
        return _hat_jit(xi)

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
        return _exp_jit(Xi)
    
    def exp_SO3(self, W: np.ndarray) -> np.ndarray:
        """Exponential map for SO(3) rotation matrix."""
        theta = np.sqrt(W[2, 1]**2 + W[0, 2]**2 + W[1, 0]**2)
        
        if theta < self._EPS:
            return np.eye(3) + W
        else:
            W2   = W @ W
            s, c = np.sin(theta), np.cos(theta)
            #print(f"theta in exp_SO3: {theta}")
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
        W2 = W @ W
       
 
        theta = np.sqrt(W[2, 1]**2 + W[0, 2]**2 + W[1, 0]**2)
        s, c = np.sin(theta), np.cos(theta)

        if theta < self._EPS:
            R = np.eye(3) 
        else:
            R   = np.eye(3) + ((1 - c) / theta**2) * W + ((theta - s) / theta**3) * W2

        R_so3 = self.exp_SO3(W)
        exp_ad = np.zeros((6, 6))
        exp_ad[:3, :3] = R_so3
        exp_ad[3:, :3] = R_so3 @ R @ self._skew(v)
        exp_ad[3:, 3:] = R_so3

        #print(f"exp_adjoint:\n {exp_ad}")

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

def rotation_matrix_to_euler_zyx(Q):
    """
    Convert rotation matrix Q to ZYX Euler angles:
    yaw about z, pitch about y, roll about x.

    Q = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """

    q11, q12, q13 = Q[0, :]
    q21, q22, q23 = Q[1, :]
    q31, q32, q33 = Q[2, :]

    pitch = np.arcsin(-q31)
    roll  = np.arctan2(q32, q33)
    yaw   = np.arctan2(q21, q11)

    return np.array([roll, pitch, yaw])


def vee(S: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 skew-symmetric matrix to a 3-vector.
    
    If
        S = [[ 0, -wz,  wy],
             [ wz,  0, -wx],
             [-wy, wx,   0]]
    then vee(S) = [wx, wy, wz].
    """
    return np.array([
        S[2, 1],
        S[0, 2],
        S[1, 0]
    ])


def log_SO3(R: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SO(3) to so(3), returned as a 3-vector.

    Input:
        R: 3x3 rotation matrix

    Output:
        theta_vec: 3-vector = theta * axis
    """

    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix.")

    # Numerical safety for arccos
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    # Case 1: very small rotation
    if theta < 1e-8:
        # log(R) ≈ 0.5 * (R - R.T)
        return 0.5 * vee(R - R.T)

    # Case 2: rotation close to pi
    if np.pi - theta < 1e-6:
        # More stable handling near 180 degrees
        A = (R + np.eye(3)) / 2.0

        axis = np.zeros(3)
        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

        # Fix signs using off-diagonal terms
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-8:
            # Fallback
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm_axis

        return theta * axis

    # General case
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)
