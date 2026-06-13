import numpy as np
from scipy.optimize import differential_evolution, minimize

from PCC_model.visualization_3d import plot_two_segment_arm_3d_interactive, plot_two_segment_arm_3d, plot_sigma_workspace_3d


EPS = 1e-9
PARAM_NAMES = ("theta1", "kappa1", "theta2", "kappa2")

class TwoSegmentArm3DOptimizer:
    def __init__(
        self,
        L1,
        L2,
        target,
        theta1_bounds=(-np.pi, np.pi),
        kappa1_bounds=(-3.0, 3.0),
        theta2_bounds=(-np.pi, np.pi),
        kappa2_bounds=(-3.0, 3.0),
        p=2.0,
    ):
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.target = np.array(target, dtype=float)
        self.theta1_bounds = theta1_bounds
        self.kappa1_bounds = kappa1_bounds
        self.theta2_bounds = theta2_bounds
        self.kappa2_bounds = kappa2_bounds
        self.p = float(p)

    # -----------------------------
    # Basic vector helpers
    # -----------------------------
    @staticmethod
    def unit_u(theta):
        """Horizontal bending direction for azimuth theta."""
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    @staticmethod
    def e_z():
        return np.array([0.0, 0.0, 1.0], dtype=float)

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < EPS:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < EPS:
            return np.zeros_like(v)
        return v / n

    def bounds_dict(self):
        return {
            "theta1": self.theta1_bounds,
            "kappa1": self.kappa1_bounds,
            "theta2": self.theta2_bounds,
            "kappa2": self.kappa2_bounds,
        }

    def default_midpoint_params(self):
        b = self.bounds_dict()
        return {name: 0.5 * (b[name][0] + b[name][1]) for name in PARAM_NAMES}

    # -----------------------------
    # One-segment 3D constant-curvature model
    # -----------------------------
    def segment_tip_local(self, L, kappa, theta):
        """
        Tip position of one constant-curvature segment in its local frame.
        Local base tangent is +z.
        """
        u = self.unit_u(theta)
        ez = self.e_z()

        if abs(kappa) < EPS:
            return L * ez

        return ((1.0 - np.cos(kappa * L)) / kappa) * u + (np.sin(kappa * L) / kappa) * ez

    def segment_tangent_local(self, L, kappa, theta):
        """
        Tip tangent of one constant-curvature segment in its local frame.
        """
        u = self.unit_u(theta)
        ez = self.e_z()
        return np.sin(kappa * L) * u + np.cos(kappa * L) * ez

    def segment_state_local(self, L, kappa, theta):
        """Return tip position and tip tangent in the local frame."""
        r_tip = self.segment_tip_local(L, kappa, theta)
        t_tip = self.segment_tangent_local(L, kappa, theta)
        return r_tip, self.normalize(t_tip)

    def segment_points_local(self, L, kappa, theta, n=200):
        """Discretized points of a constant-curvature segment in its local frame."""
        s_vals = np.linspace(0.0, L, n)
        u = self.unit_u(theta)
        ez = self.e_z()

        if abs(kappa) < EPS:
            return np.outer(s_vals, ez)

        coeff_u = (1.0 - np.cos(kappa * s_vals)) / kappa
        coeff_z = np.sin(kappa * s_vals) / kappa
        return np.outer(coeff_u, u) + np.outer(coeff_z, ez)

    # -----------------------------
    # Local-to-global frame construction for segment 2
    # -----------------------------
    def frame_from_tangent(self, tangent):
        """
        Build a rotation matrix R whose third column is the given tangent.
        This maps local segment-2 vectors into the global frame.
        """
        z_axis = self.normalize(tangent)
        if np.linalg.norm(z_axis) < EPS:
            raise ValueError("Tangent vector is too small to define a frame.")

        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(z_axis, ref)) > 1.0 - 1e-8:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)

        x_axis = self.normalize(np.cross(ref, z_axis))
        y_axis = self.normalize(np.cross(z_axis, x_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    # -----------------------------
    # Two-segment kinematics
    # -----------------------------
    def forward_kinematics(self, theta1, kappa1, theta2, kappa2, n_points=200):
        """
        Return the full 3D kinematic state for the two-segment arm.
        """
        base = np.zeros(3, dtype=float)

        r_tip1, t_tip1 = self.segment_state_local(self.L1, kappa1, theta1)
        seg1_points = self.segment_points_local(self.L1, kappa1, theta1, n=n_points)

        r_tip2_local, t_tip2_local = self.segment_state_local(self.L2, kappa2, theta2)
        seg2_points_local = self.segment_points_local(self.L2, kappa2, theta2, n=n_points)

        R1 = self.frame_from_tangent(t_tip1)

        seg2_points_global = (R1 @ seg2_points_local.T).T + r_tip1
        r_tip_total = r_tip1 + R1 @ r_tip2_local
        t_tip_total = self.normalize(R1 @ t_tip2_local)

        return {
            "base": base,
            "joint": r_tip1,
            "tip1": r_tip1,
            "tangent1": t_tip1,
            "R1": R1,
            "tip2_local": r_tip2_local,
            "tangent2_local": t_tip2_local,
            "tip_total": r_tip_total,
            "tangent_total": t_tip_total,
            "seg1_points": seg1_points,
            "seg2_points": seg2_points_global,
        }

    # -----------------------------
    # Score function sigma
    # -----------------------------
    def distance(self, theta1, kappa1, theta2, kappa2):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=50)
        return np.linalg.norm(self.target - state["tip_total"])

    def alignment(self, theta1, kappa1, theta2, kappa2):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=50)
        gap = self.target - state["tip_total"]
        gap_norm = np.linalg.norm(gap)
        if gap_norm < EPS:
            return 1.0
        g_hat = gap / gap_norm
        return float(np.dot(state["tangent_total"], g_hat))

    def sigma(self, theta1, kappa1, theta2, kappa2):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=50)
        gap = self.target - state["tip_total"]
        D = np.linalg.norm(gap)
        if D < EPS:
            return 1e12

        align = float(np.dot(state["tangent_total"], gap / D))
        return (max(0.0, align) ** self.p) / D

    def objective(self, x):
        """Minimize negative sigma to maximize sigma."""
        theta1, kappa1, theta2, kappa2 = x
        return -self.sigma(theta1, kappa1, theta2, kappa2)

    def _resolve_workspace_values(self, axis_names, fixed_params=None, grid_result=None, grid_sizes=None):
        if len(axis_names) != 3:
            raise ValueError("axis_names must contain exactly three entries.")
        if len(set(axis_names)) != 3:
            raise ValueError("axis_names must be unique.")
        for name in axis_names:
            if name not in PARAM_NAMES:
                raise ValueError(f"Invalid axis name '{name}'. Valid choices: {PARAM_NAMES}")

        fixed_params = {} if fixed_params is None else dict(fixed_params)
        for name in fixed_params:
            if name not in PARAM_NAMES:
                raise ValueError(f"Invalid fixed parameter '{name}'.")

        free_name = [name for name in PARAM_NAMES if name not in axis_names]
        if len(free_name) != 1:
            raise RuntimeError("Exactly one parameter should remain fixed.")
        free_name = free_name[0]

        defaults = self.default_midpoint_params()
        fixed_value = fixed_params.get(free_name, defaults[free_name])
        resolved_fixed = {free_name: float(fixed_value)}

        if grid_result is not None:
            value_lookup = {
                "theta1": np.asarray(grid_result["theta1_vals"], dtype=float),
                "kappa1": np.asarray(grid_result["kappa1_vals"], dtype=float),
                "theta2": np.asarray(grid_result["theta2_vals"], dtype=float),
                "kappa2": np.asarray(grid_result["kappa2_vals"], dtype=float),
            }
            sigma_map = np.asarray(grid_result["sigma_map"], dtype=float)
            index_map = {name: idx for idx, name in enumerate(PARAM_NAMES)}
            fixed_axis = index_map[free_name]
            fixed_grid_vals = value_lookup[free_name]
            fixed_idx = int(np.argmin(np.abs(fixed_grid_vals - fixed_value)))
            resolved_fixed[free_name] = float(fixed_grid_vals[fixed_idx])

            slicer = [slice(None)] * 4
            slicer[fixed_axis] = fixed_idx
            sigma_slice = sigma_map[tuple(slicer)]

            transpose_order = [PARAM_NAMES.index(name) for name in axis_names if name != free_name]
            remaining = [idx for idx in range(4) if idx != fixed_axis]
            current_names = [PARAM_NAMES[idx] for idx in remaining]
            perm = [current_names.index(name) for name in axis_names]
            sigma_plot = np.transpose(sigma_slice, axes=perm)
            grids = np.meshgrid(*(value_lookup[name] for name in axis_names), indexing="ij")
            return axis_names, grids, sigma_plot.ravel(), resolved_fixed

        if grid_sizes is None:
            grid_sizes = {name: 17 for name in PARAM_NAMES}
        elif isinstance(grid_sizes, int):
            grid_sizes = {name: int(grid_sizes) for name in PARAM_NAMES}
        else:
            grid_sizes = {name: int(grid_sizes.get(name, 17)) for name in PARAM_NAMES}

        b = self.bounds_dict()
        axis_vals = [np.linspace(b[name][0], b[name][1], grid_sizes[name]) for name in axis_names]
        mesh = np.meshgrid(*axis_vals, indexing="ij")
        sigma_vals = np.empty(mesh[0].shape, dtype=float)

        base_params = defaults
        base_params.update(resolved_fixed)

        for idx in np.ndindex(mesh[0].shape):
            params = dict(base_params)
            for axis_name, arr in zip(axis_names, mesh):
                params[axis_name] = float(arr[idx])
            sigma_vals[idx] = self.sigma(
                params["theta1"],
                params["kappa1"],
                params["theta2"],
                params["kappa2"],
            )

        return axis_names, mesh, sigma_vals.ravel(), resolved_fixed
    # -----------------------------
    # Search / optimization
    # -----------------------------
    def grid_search(self, n_theta1=21, n_kappa1=21, n_theta2=21, n_kappa2=21):
        """Brute-force 4D grid search. This can be expensive; start small."""
        theta1_vals = np.linspace(*self.theta1_bounds, n_theta1)
        kappa1_vals = np.linspace(*self.kappa1_bounds, n_kappa1)
        theta2_vals = np.linspace(*self.theta2_bounds, n_theta2)
        kappa2_vals = np.linspace(*self.kappa2_bounds, n_kappa2)

        best_sigma = -np.inf
        best_x = None
        sigma_map = np.empty((n_theta1, n_kappa1, n_theta2, n_kappa2), dtype=float)

        for i, theta1 in enumerate(theta1_vals):
            for j, kappa1 in enumerate(kappa1_vals):
                for k, theta2 in enumerate(theta2_vals):
                    for l, kappa2 in enumerate(kappa2_vals):
                        s = self.sigma(theta1, kappa1, theta2, kappa2)
                        sigma_map[i, j, k, l] = s
                        if s > best_sigma:
                            best_sigma = s
                            best_x = np.array([theta1, kappa1, theta2, kappa2], dtype=float)
        return {
            "best_x": best_x,
            "best_sigma": best_sigma,
            "theta1_vals": theta1_vals,
            "kappa1_vals": kappa1_vals,
            "theta2_vals": theta2_vals,
            "kappa2_vals": kappa2_vals,
            "sigma_map": sigma_map,
        }

    def optimize_global(self, polish=True):
        bounds = [
            self.theta1_bounds,
            self.kappa1_bounds,
            self.theta2_bounds,
            self.kappa2_bounds,
        ]
        return differential_evolution(self.objective, bounds=bounds, polish=polish)

    def optimize_local(self, x0, method="Nelder-Mead"):
        return minimize(self.objective, x0=x0, method=method)

    # -----------------------------
    # Reporting / visualization
    # -----------------------------
    def summarize_solution(self, theta1, kappa1, theta2, kappa2):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=200)
        gap = self.target - state["tip_total"]
        D = np.linalg.norm(gap)
        align = 1.0 if D < EPS else float(np.dot(state["tangent_total"], gap / D))

        return {
            "theta1": theta1,
            "kappa1": kappa1,
            "theta2": theta2,
            "kappa2": kappa2,
            "tip": state["tip_total"],
            "tangent": state["tangent_total"],
            "distance": D,
            "alignment": align,
            "sigma": self.sigma(theta1, kappa1, theta2, kappa2),
        }

    def plot_sigma_workspace_3d(
            self,
            axis_names=("theta1", "kappa1", "theta2"),
            fixed_params=None,
            grid_result=None,
            grid_sizes=None,
            show=True,
            marker_size=22.0,
            alpha=0.85,
    ):
        """
        Plot a 3D sigma workspace map.

        Parameters
        ----------
        axis_names : tuple[str, str, str]
            Pick any three of ('theta1', 'kappa1', 'theta2', 'kappa2') for the axes.
        fixed_params : dict | None
            Value for the remaining parameter not shown on the axes.
            Example: {'kappa2': 0.0}
        grid_result : dict | None
            Output of self.grid_search(...). If provided, the method slices that 4D sigma_map.
        grid_sizes : int | dict | None
            Used only when grid_result is None. Controls the sampling density.
        """
        axis_names, mesh, sigma_flat, resolved_fixed = self._resolve_workspace_values(
            axis_names=axis_names,
            fixed_params=fixed_params,
            grid_result=grid_result,
            grid_sizes=grid_sizes,
        )

        axis_values = tuple(arr.ravel() for arr in mesh)
        return plot_sigma_workspace_3d(
            axis_names=axis_names,
            axis_values=axis_values,
            sigma_values=sigma_flat,
            fixed_params=resolved_fixed,
            show=show,
            marker_size=marker_size,
            alpha=alpha,
        )

    def plot_arm_interactive(self, theta1, kappa1, theta2, kappa2, n_points=200, show=True):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=n_points)
        gap = self.target - state["tip_total"]
        D = np.linalg.norm(gap)
        align = 1.0 if D < EPS else float(np.dot(state["tangent_total"], gap / D))

        return plot_two_segment_arm_3d_interactive(
            seg1_points=state["seg1_points"],
            seg2_points=state["seg2_points"],
            base=state["base"],
            joint=state["joint"],
            tip=state["tip_total"],
            target=self.target,
            tangent_tip=state["tangent_total"],
            gap_vec=gap,
            sigma=self.sigma(theta1, kappa1, theta2, kappa2),
            align=align,
            distance=D,
            theta1=theta1,
            kappa1=kappa1,
            theta2=theta2,
            kappa2=kappa2,
            tangent_scale=max(self.L1, self.L2),
            show=show,
        )

    def plot_arm_static(self, theta1, kappa1, theta2, kappa2, n_points=200, show=True):
        state = self.forward_kinematics(theta1, kappa1, theta2, kappa2, n_points=n_points)
        gap = self.target - state["tip_total"]
        D = np.linalg.norm(gap)
        align = 1.0 if D < EPS else float(np.dot(state["tangent_total"], gap / D))

        return plot_two_segment_arm_3d(
            seg1_points=state["seg1_points"],
            seg2_points=state["seg2_points"],
            base=state["base"],
            joint=state["joint"],
            tip=state["tip_total"],
            target=self.target,
            tangent_tip=state["tangent_total"],
            gap_vec=gap,
            sigma=self.sigma(theta1, kappa1, theta2, kappa2),
            align=align,
            distance=D,
            theta1=theta1,
            kappa1=kappa1,
            theta2=theta2,
            kappa2=kappa2,
            tangent_scale=max(self.L1, self.L2),
            gap_scale = 1.0,
            show=show,
        )



if __name__ == "__main__":
    arm = TwoSegmentArm3DOptimizer(
        L1=0.5,
        L2=0.5,
        target=[0.5, 0.4, 1.4],
        theta1_bounds=(-np.pi, np.pi),
        kappa1_bounds=(-3.0, 3.0),
        theta2_bounds=(-np.pi, np.pi),
        kappa2_bounds=(-3.0, 3.0),
        p=2.0,
    )

    result = arm.optimize_global(polish=True)
    theta1, kappa1, theta2, kappa2 = result.x

    print("Global optimization result:")
    print(arm.summarize_solution(theta1, kappa1, theta2, kappa2))

    local = arm.optimize_local(result.x)
    theta1, kappa1, theta2, kappa2 = local.x

    print("\nLocal refinement result:")
    print(arm.summarize_solution(theta1, kappa1, theta2, kappa2))

    #arm.plot_arm_interactive(theta1, kappa1, theta2, kappa2)
    arm.plot_arm_static(theta1, kappa1, theta2, kappa2)

    # Option 2: use an existing 4D grid search result
    grid = arm.grid_search(n_theta1=20, n_kappa1=20, n_theta2=20, n_kappa2=20)

    arm.plot_sigma_workspace_3d(
        axis_names=("theta1", "kappa1", "kappa2"),
        fixed_params={"theta2": 0.0},
        grid_result=grid,
    )