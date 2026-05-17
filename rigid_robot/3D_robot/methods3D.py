import numpy as np


class SE2LieAlgebra:
    """
    SE(2) Lie group / se(2) Lie algebra utilities.

    Convention: xi = [v1, v2, omega] in R^3
    T in SE(2) is a 3x3 homogeneous matrix:
        [ R  t ]
        [ 0  1 ]
    where R in SO(2), t in R^2.
    """

    _EPS = 1e-8

    # ------------------------------------------------------------------ #
    # Algebra operators                                                    #
    # ------------------------------------------------------------------ #

    def hat(self, xi: np.ndarray) -> np.ndarray:
        """Map R^3 -> se(2): vector to skew-symmetric matrix."""
        v1, v2, w = xi
        return np.array([[ 0, -w, v1],
                         [ w,  0, v2],
                         [ 0,  0,  0]], dtype=float)

    def vee(self, Xi: np.ndarray) -> np.ndarray:
        """Map se(2) -> R^3: matrix to vector (inverse of hat)."""
        return np.array([Xi[0, 2], Xi[1, 2], Xi[1, 0]])

    def bracket(self, xi1: np.ndarray, xi2: np.ndarray) -> np.ndarray:
        """Lie bracket [xi1, xi2] = vee(hat(xi1) @ hat(xi2) - hat(xi2) @ hat(xi1))."""
        return self.vee(self.hat(xi1) @ self.hat(xi2) - self.hat(xi2) @ self.hat(xi1))

    # ------------------------------------------------------------------ #
    # Group operators                                                      #
    # ------------------------------------------------------------------ #

    def inverse(self, T: np.ndarray) -> np.ndarray:
        """Inverse of T in SE(2): T^{-1} = [R^T, -R^T t; 0, 1]."""
        R = T[:2, :2]
        t = T[:2, 2]
        T_inv = np.eye(3)
        T_inv[:2, :2] = R.T
        T_inv[:2,  2] = -R.T @ t
        return T_inv

    # ------------------------------------------------------------------ #
    # Exponential and logarithmic maps                                     #
    # ------------------------------------------------------------------ #

    def exp(self, Xi: np.ndarray) -> np.ndarray:
        """
        Exponential map se(2) matrix -> SE(2).

        Xi =
        [[ 0, -w, v1],
        [ w,  0, v2],
        [ 0,  0,  0]]
        """

        v = Xi[:2, 2]
        w = Xi[1, 0]
        c, s = np.cos(w), np.sin(w)
        R = np.array([
            [c, -s],
            [s,  c]
        ])
        J = np.array([
            [0, -1],
            [1,  0]
        ])
        if np.abs(w) < self._EPS:
            V = np.eye(2) + 0.5 * w * J
        else:
            V = np.array([
                [s, -(1 - c)],
                [1 - c, s]
            ]) / w
        t = V @ v
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t

        return T


    def log(self, T: np.ndarray) -> np.ndarray:
        """
        Logarithmic map SE(2) matrix -> se(2) matrix.

        Returns Xi =
        [[ 0, -w, v1],
        [ w,  0, v2],
        [ 0,  0,  0]]
        """
        R = T[:2, :2]
        t = T[:2, 2]
        w = np.arctan2(R[1, 0], R[0, 0])
        J = np.array([
            [0, -1],
            [1,  0]
        ])
        if np.abs(w) < self._EPS:
            V_inv = np.eye(2) - 0.5 * w * J
        else:
            half_w = 0.5 * w
            factor = half_w / np.tan(half_w)
            V_inv = np.array([
                [factor, half_w],
                [-half_w, factor]
            ])
        v = V_inv @ t
        Xi = np.zeros((3, 3))
        Xi[0, 1] = -w
        Xi[1, 0] = w
        Xi[0, 2] = v[0]
        Xi[1, 2] = v[1]

        return Xi
    # ------------------------------------------------------------------ #
    # Adjoint representation                                               #
    # ------------------------------------------------------------------ #

    def adjoint(self, T: np.ndarray) -> np.ndarray:
        """
        Adjoint matrix Ad_T in R^{3x3} such that
        Ad_T @ xi = vee(T @ hat(xi) @ T^{-1}).
        """
        R = T[:2, :2]
        t = T[:2, 2]
        Ad = np.eye(3)
        Ad[:2, :2] = R
        Ad[0,  2]  =  t[1]
        Ad[1,  2]  = -t[0]
        return Ad

    def coadjoint(self, Xi: np.ndarray) -> np.ndarray:
        """
        Co-adjoint ad_xi^* = ad_xi^T from a 3x3 se(2) matrix Xi = hat(xi).
        Used in the Euler-Poincaré equation: M*dxi/dt = coad(Xi) @ (M @ xi) + F
        """
        v1, v2, w = Xi[0, 2], Xi[1, 2], Xi[1, 0]
        return np.array([[ 0,  w,  0],
                         [-w,  0,  0],
                         [v2, -v1,  0]], dtype=float)

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def rotation_matrix(self, theta: float) -> np.ndarray:
        """2x2 rotation matrix for angle theta."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def compose(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """Group composition T1 @ T2."""
        return T1 @ T2
