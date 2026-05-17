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

    def adjoint(self, T: np.ndarray) -> np.ndarray:
        """
        Group adjoint Ad_T in R^{6x6} such that
        Ad_T @ xi = vee(T @ hat(xi) @ T^{-1}).

        Ad_T = [[ R,    hat(t) @ R ],
                [ 0_33, R          ]]
        """
        R     = T[:3, :3]
        t_hat = self._skew(T[:3, 3])
        Ad    = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[:3, 3:] = t_hat @ R
        Ad[3:, 3:] = R
        return Ad

    def coadjoint(self, Xi: np.ndarray) -> np.ndarray:
        """
        Co-adjoint ad_xi^* = ad_xi^T from a 4x4 se(3) matrix Xi = hat(xi).
        Used in the Euler-Poincaré equation: M*dxi/dt = coad(Xi) @ (M @ xi) + F

        coad = [[ W^T,      0_33 ],
                [ hat(v),   W^T  ]]
        where W = hat(w) so W^T = -hat(w).
        """
        v = Xi[:3, 3]
        W = Xi[:3, :3]   # hat(w), skew-symmetric
        coad = np.zeros((6, 6))
        coad[:3, :3] = W.T
        coad[3:, :3] = self._skew(v)
        coad[3:, 3:] = W.T
        return coad

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
