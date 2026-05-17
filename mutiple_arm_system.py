import numpy as np
from scipy.optimize import differential_evolution, minimize

from visualization_4_arm import plot_four_arm_static, plot_base_layout

EPS = 1e-9
PARAM_NAMES = ("theta1", "kappa1", "theta2", "kappa2")


class FourArmTwoSegment3DOptimizer:
    """
    Four identical two-segment 3D continuum arms mounted on one plane.

    Each arm has 4 state variables:
        [theta1, kappa1, theta2, kappa2]

    Total state dimension = 16.
    """

    def __init__(
        self,
        L1,
        L2,
        target,
        plane_origin=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        base_points=None,
        base_layout="square",
        layout_size= 2.0,
        theta1_bounds=(-np.pi, np.pi),
        kappa1_bounds=(-3.0, 3.0),
        theta2_bounds=(-np.pi, np.pi),
        kappa2_bounds=(-3.0, 3.0),
        p=70.0,
    ):
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.target = np.asarray(target, dtype=float)
        self.p = float(p)

        self.theta1_bounds = tuple(theta1_bounds)
        self.kappa1_bounds = tuple(kappa1_bounds)
        self.theta2_bounds = tuple(theta2_bounds)
        self.kappa2_bounds = tuple(kappa2_bounds)

        self.plane_origin = np.asarray(plane_origin, dtype=float)
        self.plane_normal = self.normalize(np.asarray(plane_normal, dtype=float))
        if np.linalg.norm(self.plane_normal) < EPS:
            raise ValueError("plane_normal must be nonzero.")

        if base_points is None:
            self.base_points = self.generate_base_points(
                layout=base_layout,
                size=layout_size,
                origin=self.plane_origin,
                normal=self.plane_normal,
            )
        else:
            base_points = np.asarray(base_points, dtype=float)
            if base_points.shape != (4, 3):
                raise ValueError("base_points must have shape (4, 3).")
            self.base_points = base_points

        self.base_rotation = self.frame_from_normal(self.plane_normal)

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < EPS:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def unit_u(theta):
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    @staticmethod
    def e_z():
        return np.array([0.0, 0.0, 1.0], dtype=float)

    @classmethod
    def _pick_reference_not_parallel(cls, v):
        v = cls.normalize(v)
        refs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        dots = [abs(np.dot(v, r)) for r in refs]
        return refs[int(np.argmin(dots))]

    @classmethod
    def frame_from_normal(cls, normal):
        z_axis = cls.normalize(normal)
        ref = cls._pick_reference_not_parallel(z_axis)
        x_axis = cls.normalize(np.cross(ref, z_axis))
        y_axis = cls.normalize(np.cross(z_axis, x_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def frame_from_tangent(self, tangent):
        z_axis = self.normalize(tangent)
        if np.linalg.norm(z_axis) < EPS:
            raise ValueError("Tangent vector is too small to define a frame.")
        ref = self._pick_reference_not_parallel(z_axis)
        x_axis = self.normalize(np.cross(ref, z_axis))
        y_axis = self.normalize(np.cross(z_axis, x_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    @classmethod
    def generate_base_points(cls, layout, size, origin, normal):
        origin = np.asarray(origin, dtype=float)
        normal = cls.normalize(np.asarray(normal, dtype=float))
        R = cls.frame_from_normal(normal)
        half = 0.5 * float(size)

        if layout == "square":
            local_pts = np.array([
                [-half, -half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
                [-half,  half, 0.0],
            ])
        elif layout == "diamond":
            local_pts = np.array([
                [ 0.0,  half, 0.0],
                [ half, 0.0, 0.0],
                [ 0.0, -half, 0.0],
                [-half, 0.0, 0.0],
            ])
        elif layout == "line":
            local_pts = np.array([
                [-1.5 * half, 0.0, 0.0],
                [-0.5 * half, 0.0, 0.0],
                [ 0.5 * half, 0.0, 0.0],
                [ 1.5 * half, 0.0, 0.0],
            ])
        else:
            raise ValueError("layout must be one of: 'square', 'diamond', 'line'.")

        return (R @ local_pts.T).T + origin

    def segment_tip_local(self, L, kappa, theta):
        u = self.unit_u(theta)
        ez = self.e_z()
        if abs(kappa) < EPS:
            return L * ez
        return ((1.0 - np.cos(kappa * L)) / kappa) * u + (np.sin(kappa * L) / kappa) * ez

    def segment_tangent_local(self, L, kappa, theta):
        u = self.unit_u(theta)
        ez = self.e_z()
        return self.normalize(np.sin(kappa * L) * u + np.cos(kappa * L) * ez)

    def segment_points_local(self, L, kappa, theta, n=200):
        s_vals = np.linspace(0.0, L, n)
        u = self.unit_u(theta)
        ez = self.e_z()
        if abs(kappa) < EPS:
            return np.outer(s_vals, ez)
        coeff_u = (1.0 - np.cos(kappa * s_vals)) / kappa
        coeff_z = np.sin(kappa * s_vals) / kappa
        return np.outer(coeff_u, u) + np.outer(coeff_z, ez)

    def segment_state_local(self, L, kappa, theta):
        r_tip = self.segment_tip_local(L, kappa, theta)
        t_tip = self.segment_tangent_local(L, kappa, theta)
        return r_tip, t_tip

    def forward_kinematics_arm(self, base_point, theta1, kappa1, theta2, kappa2, n_points=200):
        R_base = self.base_rotation

        r_tip1_local, t_tip1_local = self.segment_state_local(self.L1, kappa1, theta1)
        seg1_local = self.segment_points_local(self.L1, kappa1, theta1, n=n_points)

        seg1_global = (R_base @ seg1_local.T).T + base_point
        r_tip1 = base_point + R_base @ r_tip1_local
        t_tip1 = self.normalize(R_base @ t_tip1_local)

        R1 = self.frame_from_tangent(t_tip1)
        r_tip2_local, t_tip2_local = self.segment_state_local(self.L2, kappa2, theta2)
        seg2_local = self.segment_points_local(self.L2, kappa2, theta2, n=n_points)

        seg2_global = (R1 @ seg2_local.T).T + r_tip1
        tip_total = r_tip1 + R1 @ r_tip2_local
        tangent_total = self.normalize(R1 @ t_tip2_local)

        return {
            "base": np.asarray(base_point, dtype=float),
            "joint": r_tip1,
            "tip": tip_total,
            "tangent": tangent_total,
            "seg1_points": seg1_global,
            "seg2_points": seg2_global,
        }

    @staticmethod
    def unpack_state(x):
        x = np.asarray(x, dtype=float)
        if x.shape != (16,):
            raise ValueError("State x must have shape (16,).")
        return x.reshape(4, 4)

    def bounds(self):
        per_arm = [
            self.theta1_bounds,
            self.kappa1_bounds,
            self.theta2_bounds,
            self.kappa2_bounds,
        ]
        return per_arm * 4

    def sigma_arm(self, arm_index, theta1, kappa1, theta2, kappa2):
        state = self.forward_kinematics_arm(
            self.base_points[arm_index], theta1, kappa1, theta2, kappa2, n_points=60
        )
        gap = self.target - state["tip"]
        D = np.linalg.norm(gap)
        if D < EPS:
            return 1e12
        align = float(np.dot(state["tangent"], gap / D))
        return (max(0.0, align) ** self.p) / D

    def arm_metrics(self, arm_index, theta1, kappa1, theta2, kappa2, n_points=200):
        state = self.forward_kinematics_arm(
            self.base_points[arm_index], theta1, kappa1, theta2, kappa2, n_points=n_points
        )
        gap = self.target - state["tip"]
        D = np.linalg.norm(gap)
        align = 1.0 if D < EPS else float(np.dot(state["tangent"], gap / D))
        sigma_i = 1e12 if D < EPS else (max(0.0, align) ** self.p) / D
        out = dict(state)
        out.update({
            "distance": D,
            "alignment": align,
            "sigma": sigma_i,
            "gap": gap,
        })
        return out

    def sigma_total(self, x):
        arms = self.unpack_state(x)
        total = 0.0
        for i in range(4):
            theta1, kappa1, theta2, kappa2 = arms[i]
            total += self.sigma_arm(i, theta1, kappa1, theta2, kappa2)
        return total

    def objective(self, x):
        return -self.sigma_total(x)

    def optimize_global(self, maxiter=300, popsize=15, polish=True, seed=None, workers=1):
        return differential_evolution(
            self.objective,
            bounds=self.bounds(),
            maxiter=maxiter,
            popsize=popsize,
            polish=polish,
            seed=seed,
            workers=workers,
        )

    def optimize_local(self, x0, method="Nelder-Mead"):
        return minimize(self.objective, x0=np.asarray(x0, dtype=float), method=method)

    def summarize_solution(self, x, n_points=200):
        arms = self.unpack_state(x)
        arm_summaries = []
        total_sigma = 0.0

        for i in range(4):
            theta1, kappa1, theta2, kappa2 = arms[i]
            state = self.arm_metrics(i, theta1, kappa1, theta2, kappa2, n_points=n_points)
            state.update({
                "arm_index": i,
                "theta1": theta1,
                "kappa1": kappa1,
                "theta2": theta2,
                "kappa2": kappa2,
            })
            total_sigma += state["sigma"]
            arm_summaries.append(state)

        return {
            "target": self.target,
            "plane_origin": self.plane_origin,
            "plane_normal": self.plane_normal,
            "base_points": self.base_points,
            "total_sigma": total_sigma,
            "arms": arm_summaries,
        }

    def plot_base_layout(self, show=True):
        return plot_base_layout(
            base_points=self.base_points,
            plane_origin=self.plane_origin,
            plane_normal=self.plane_normal,
            show=show,
        )

    def plot_solution_static(self, x, n_points=200, tangent_scale=None, show_sigma_text=True, cuboids=None, cylinders=None, show=True):
        summary = self.summarize_solution(x, n_points=n_points)
        if tangent_scale is None:
            tangent_scale = 0.30 * max(self.L1, self.L2)
        return plot_four_arm_static(
            arm_states=summary["arms"],
            target=summary["target"],
            plane_normal=summary["plane_normal"],
            tangent_scale=tangent_scale,
            show_sigma_text=show_sigma_text,
            cuboids=cuboids,
            cylinders=cylinders,
            title=f"Four-arm two-segment optimizer | total sigma = {summary['total_sigma']:.4f}",
            show=show,
        )


if __name__ == "__main__":

    opt = FourArmTwoSegment3DOptimizer(
        L1=0.4363,
        L2=0.3363,
        target=[0.40, -0.05, 0.84],
        plane_origin=[0.0, 0.0, 2.4],
        plane_normal=[0.0, 0.0, -1.0],
        base_layout="square",
        layout_size=1.8,
        p=25.0,
    )

    opt.plot_base_layout(show=False)
    result = opt.optimize_global(maxiter=20, popsize=10, polish=True, seed=1)
    print("Best total sigma:", -result.fun)

    center = [0.0, 0.0, 0.3]
    size = [1.2, 0.8, 0.6]

    table  = [
        {
            "origin": [-0.2, -0.4, 0.0],  # lower-left-bottom corner
            "size": [1.2, 0.8, 0.82],  # width_x, width_y, height_z
            "color": "#B0B0B0",
            "alpha": 0.5,
        }
    ]

    obstacle = [
        {
            "center_xy": [0.4, 0.4],
            "radius": 0.22,
            "z0": 0.0,
            "height": 1.7,
            "color": "#D95F5F",
            "alpha": 0.6,
        },
        {
            "center_xy": [-0.3, 0.45],
            "radius": 0.22,
            "z0": 0.0,
            "height": 1.8,
            "color": "#5F8DD9",
            "alpha": 0.6,
        },
        {
            "center_xy": [-0.2, -0.35],
            "radius": 0.22,
            "z0": 0.1,
            "height": 1.75,
            "color": "#6DAA6D",
            "alpha": 0.6,
        },
    ]


    opt.plot_solution_static(result.x, cuboids = table, cylinders = obstacle)
