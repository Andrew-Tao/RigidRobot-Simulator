import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

EPS = 1e-9

class TwoSegmentArmOptimizer:
    def __init__(
        self,
        L1,
        L2,
        d,
        theta_block1,
        theta_block2,
        kappa1_bounds=(-3.0, 3.0),
        kappa2_bounds=(-3.0, 3.0),
        target=None,
    ):
        self.L1 = L1
        self.L2 = L2
        self.d = d
        self.theta_block1 = theta_block1
        self.theta_block2 = theta_block2
        self.kappa1_bounds = kappa1_bounds
        self.kappa2_bounds = kappa2_bounds
        self.target = np.array([d, 0.0]) if target is None else np.array(target, dtype=float)

    # -----------------------------
    # Geometry functions
    # -----------------------------
    @staticmethod
    def segment_tip(kappa, L, theta0):
        """
        Tip displacement of one constant-curvature planar segment
        with length L, curvature kappa, and base angle theta0.
        """
        if abs(kappa) < EPS:
            return np.array([
                L * np.cos(theta0),
                L * np.sin(theta0)
            ])
        else:
            return np.array([
                (np.sin(theta0 + kappa * L) - np.sin(theta0)) / kappa,
                (-np.cos(theta0 + kappa * L) + np.cos(theta0)) / kappa
            ])

    def tip_position(self, kappa1, kappa2):
        """
        Total tip position of two connected segments.
        Segment 1 starts from origin with theta0 = 0.
        Segment 2 starts from tip of segment 1 with theta0 = kappa1*L.
        """
        r1 = self.segment_tip(kappa1, self.L1, theta0=0.0)
        theta1 = kappa1 * self.L1
        r2 = self.segment_tip(kappa2, self.L2, theta0=theta1)
        return r1 + r2

    def total_theta(self, kappa1, kappa2):
        """
        Total tip tangent angle.
        """
        return (kappa1 * self.L1 + kappa2 * self.L2)

    @staticmethod
    def tangent_tip(theta):
        return np.array([np.cos(theta), np.sin(theta)])

    def distance_squared(self, kappa1, kappa2):
        """
        Squared distance from tip to target.
        """
        r_tip = self.tip_position(kappa1, kappa2)
        diff = r_tip - self.target
        return np.dot(diff, diff)

    # -----------------------------
    # Blocked-angle logic
    # -----------------------------
    def is_blocked(self, theta):
        """
        Returns True if theta lies inside blocked interval.
        Assumes theta_block1 <= theta_block2.
        """
        return self.theta_block1 <= theta <= self.theta_block2

    def sigma(self, kappa1, kappa2):
        """
        Score function:
        sigma = pointing_score / distance_penalty outside blocked region
        sigma = 0 inside blocked region
        """
        theta = self.total_theta(kappa1, kappa2)

        if self.is_blocked(theta):
            return 0.0

        D2 = self.distance_squared(kappa1, kappa2)
        tip = self.tip_position(kappa1, kappa2)
        v_gap = self.target - tip
        v_gap_norm = np.linalg.norm(v_gap)

        if v_gap_norm < EPS:
            # tip exactly at target
            return 1e12

        v_e = v_gap / v_gap_norm
        align = np.dot(self.tangent_tip(theta), v_e)

        return max(1e-9, align**71) / (max(D2, 1e-12)**(1/5))

    # -----------------------------
    # Objective for minimization
    # -----------------------------
    def objective_with_penalty(self, x, penalty=1e6):
        """
        Minimize pointing-aware distance score,
        but add large penalty if theta is blocked.
        """
        k1, k2 = x
        D2 = self.distance_squared(k1, k2)
        theta = self.total_theta(k1, k2)
        tip = self.tip_position(k1, k2)

        v_gap = self.target - tip
        v_gap_norm = np.linalg.norm(v_gap)

        if v_gap_norm < EPS:
            pointing_score = 0.0
        else:
            direction = v_gap / v_gap_norm
            align = np.dot(self.tangent_tip(theta), direction)
            pointing_score = D2 / max(1e-9, align)**2

        if self.is_blocked(theta):
            return pointing_score + penalty
        return pointing_score

    # -----------------------------
    # Grid search
    # -----------------------------
    def grid_search(self, n1=300, n2=300):
        """
        Brute-force search over a grid.
        """
        k1_vals = np.linspace(self.kappa1_bounds[0], self.kappa1_bounds[1], n1)
        k2_vals = np.linspace(self.kappa2_bounds[0], self.kappa2_bounds[1], n2)

        best_sigma = -np.inf
        best_k1 = None
        best_k2 = None

        sigma_map = np.zeros((n2, n1))

        for i, k1 in enumerate(k1_vals):
            for j, k2 in enumerate(k2_vals):
                s = self.sigma(k1, k2)
                sigma_map[j, i] = s

                if s > best_sigma:
                    best_sigma = s
                    best_k1 = k1
                    best_k2 = k2

        return {
            "best_k1": best_k1,
            "best_k2": best_k2,
            "best_sigma": best_sigma,
            "k1_vals": k1_vals,
            "k2_vals": k2_vals,
            "sigma_map": sigma_map,
        }

    # -----------------------------
    # Differential evolution
    # -----------------------------
    def optimize_global(self, polish=True):
        bounds = [self.kappa1_bounds, self.kappa2_bounds]
        result = differential_evolution(
            self.objective_with_penalty,
            bounds=bounds,
            polish=polish
        )
        return result

    # -----------------------------
    # Local refinement
    # -----------------------------
    def optimize_local(self, x0, method="Nelder-Mead"):
        result = minimize(
            self.objective_with_penalty,
            x0=x0,
            method=method
        )
        return result

    # -----------------------------
    # Reporting
    # -----------------------------
    def summarize_solution(self, k1, k2):
        tip = self.tip_position(k1, k2)
        theta = self.total_theta(k1, k2)
        D2 = self.distance_squared(k1, k2)

        return {
            "kappa1": k1,
            "kappa2": k2,
            "tip": tip,
            "theta": theta,
            "distance": np.sqrt(D2),
            "distance_squared": D2,
            "sigma": self.sigma(k1, k2),
            "blocked": self.is_blocked(theta),
        }

    # -----------------------------
    # Visualization
    # -----------------------------
    def plot_sigma_map(self, sigma_map, best_k1, best_k2):
        plt.figure(figsize=(8, 6))
        plt.imshow(
            sigma_map,
            extent=[
                self.kappa1_bounds[0], self.kappa1_bounds[1],
                self.kappa2_bounds[0], self.kappa2_bounds[1]
            ],
            origin='lower',
            aspect='auto'
        )
        plt.colorbar(label='sigma')
        plt.scatter([best_k1], [best_k2], marker='x', s=100, label='Grid optimum')
        plt.xlabel('kappa1')
        plt.ylabel('kappa2')
        plt.title('Objective landscape')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from visualization import plot_two_segment_setup

    # -----------------------------
    # User settings
    # -----------------------------
    L1 = 1
    L2 = 0.4
    L1_tem = L1 / (L1 + L2)
    L2_tem = L2 / (L1 + L2)

    L1 = L1_tem
    L2 = L2_tem

    d0 = (L1 + L2) * 1.1

    theta_block1 = 0.6
    theta_block2 = 0.5

    # search bounds for kappa1 and kappa2
    kappa1_bounds = (-8, 8)
    kappa2_bounds = (-8, 8)

    # -----------------------------
    # Build optimizer
    # -----------------------------
    target = np.array([d0, 1])

    arm = TwoSegmentArmOptimizer(
        L1,
        L2,
        d0,
        theta_block1,
        theta_block2,
        kappa1_bounds=kappa1_bounds,
        kappa2_bounds=kappa2_bounds,
        target = target,
    )

    # -----------------------------
    # Run grid search
    # -----------------------------
    grid_result = arm.grid_search(n1=250, n2=250)

    best_k1_g = grid_result["best_k1"]
    best_k2_g = grid_result["best_k2"]
    best_sigma_g = grid_result["best_sigma"]
    k1_vals = grid_result["k1_vals"]
    k2_vals = grid_result["k2_vals"]
    sigma_map = grid_result["sigma_map"]

    print("Grid search result:")
    print(f"  kappa1 = {best_k1_g:.6f}")
    print(f"  kappa2 = {best_k2_g:.6f}")
    print(f"  sigma  = {best_sigma_g:.6f}")

    r_best_g = arm.tip_position(best_k1_g, best_k2_g)
    theta_best_g = arm.total_theta(best_k1_g, best_k2_g)
    D2_best_g = arm.distance_squared(best_k1_g, best_k2_g)

    print(f"  r_tip  = {r_best_g}")
    print(f"  theta  = {theta_best_g:.6f} rad")
    print(f"  D      = {np.sqrt(D2_best_g):.6f}")

    print("target =", target)
    print("tip tangent =", arm.tangent_tip(theta_best_g))
    print("target - tip =", target - arm.tip_position(best_k1_g, best_k2_g))
    print(
        "dot =",
        np.dot(
            arm.tangent_tip(theta_best_g),
            target - arm.tip_position(best_k1_g, best_k2_g)
        )
    )

    # -----------------------------
    # Visualization: sigma map
    # -----------------------------
    arm.plot_sigma_map(sigma_map, best_k1_g, best_k2_g)

    # -----------------------------
    # Visualization: arm setup
    # -----------------------------
    plot_two_segment_setup(
        c1=best_k1_g,
        c2=best_k2_g,
        L1=L1,
        L2=L2,
        target=target,
        theta_b1=theta_block1,
        theta_b2=theta_block2
    )
