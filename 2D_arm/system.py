import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# small number to avoid division by zero
EPS = 1e-9


# -----------------------------
# Geometry functions
# -----------------------------
def segment_tip(kappa, L, theta0):
    """
    Tip displacement of one constant-curvature planar segment
    with length L, curvature kappa, and base angle theta0.
    """
    if abs(kappa) < EPS:
        # straight segment limit
        return np.array([
            L * np.cos(theta0),
            L * np.sin(theta0)
        ])
    else:
        return np.array([
            (np.sin(theta0 + kappa * L) - np.sin(theta0)) / kappa,
            (-np.cos(theta0 + kappa * L) + np.cos(theta0)) / kappa
        ])


def tip_position(kappa1, kappa2, L):
    """
    Total tip position of two connected segments.
    Segment 1 starts from origin with theta0 = 0.
    Segment 2 starts from tip of segment 1 with theta0 = kappa1*L.
    """
    r1 = segment_tip(kappa1, L, theta0=0.0)
    theta1 = kappa1 * L
    r2 = segment_tip(kappa2, L, theta0=theta1)
    return r1 + r2


def total_theta(kappa1, kappa2, L):
    """
    Total tip tangent angle.
    """
    return (kappa1 + kappa2) * L

def tangent_tip(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def distance_squared(kappa1, kappa2, L, d):
    """
    Squared distance from tip to target [d, 0].
    """
    r_tip = tip_position(kappa1, kappa2, L)
    r_target = np.array([d, 0.0])
    diff = r_tip - r_target
    return np.dot(diff, diff)


# -----------------------------
# Blocked-angle logic
# -----------------------------
def is_blocked(theta, theta_b1, theta_b2):
    """
    Returns True if theta lies inside blocked interval.
    Assumes theta_b1 <= theta_b2.
    """
    return theta_b1 <= theta <= theta_b2


def sigma(kappa1, kappa2, L, d, theta_b1, theta_b2, target):
    """
    Your score function:
    sigma = 1/D^2 outside blocked region
    sigma = 0 inside blocked region
    """
    theta = total_theta(kappa1, kappa2, L)

    if is_blocked(theta, theta_b1, theta_b2):
        return 0.0

    D2 = distance_squared(kappa1, kappa2, L, d)
    tip = tip_position(kappa1, kappa2, L)
    v_gap = target - tip

    v_gap_norm = np.linalg.norm(v_gap)
    v_e = v_gap / v_gap_norm

    # avoid blowup if D2 is extremely small
    return max(1e-9,np.dot(tangent_tip(theta), v_e)**71) / ((max(D2, 1e-12))**(1/5))
    #return 1 / max(D2, 1e-12)


# -----------------------------
# Objective for minimization
# -----------------------------
def objective_with_penalty(x, L, d, theta_b1, theta_b2, target, penalty=1e6):
    """
    Minimize D^2, but add large penalty if theta is blocked.
    """
    k1, k2 = x
    D2 = distance_squared(k1, k2, L, d)
    theta = total_theta(k1, k2, L)
    tip = tip_position(k1, k2, L)
    Pointing_score = D2 / max(1e-9,np.dot(tangent_tip(theta), ((target - tip)/np.linalg.norm(target - tip))))**2

    if is_blocked(theta, theta_b1, theta_b2):
        return Pointing_score + penalty
    return Pointing_score


# -----------------------------
# Grid search
# -----------------------------
def grid_search(L, d, theta_b1, theta_b2,
                k1_bounds, k2_bounds,target,
                n1=300, n2=300):
    """
    Brute-force search over a grid.
    """
    k1_vals = np.linspace(k1_bounds[0], k1_bounds[1], n1)
    k2_vals = np.linspace(k2_bounds[0], k2_bounds[1], n2)

    best_sigma = -np.inf
    best_k1 = None
    best_k2 = None

    sigma_map = np.zeros((n2, n1))

    for i, k1 in enumerate(k1_vals):
        for j, k2 in enumerate(k2_vals):
            s = sigma(k1, k2, L, d, theta_b1, theta_b2,target)
            sigma_map[j, i] = s

            if s > best_sigma:
                best_sigma = s
                best_k1 = k1
                best_k2 = k2

    return best_k1, best_k2, best_sigma, k1_vals, k2_vals, sigma_map


# -----------------------------
# User settings
# -----------------------------
L0 = 0.5
d0 = 3

theta_block1 = -0.71
theta_block2 = 0.56

# search bounds for kappa1 and kappa2
kappa1_bounds = (-3.0, 3.0)
kappa2_bounds = (-3.0, 3.0)


# -----------------------------
# Run grid search
# -----------------------------
target = np.array([d0, 0])
best_k1_g, best_k2_g, best_sigma_g, k1_vals, k2_vals, sigma_map = grid_search(
    L0, d0, theta_block1, theta_block2,
    kappa1_bounds, kappa2_bounds,target,
    n1=250, n2=250
)

print("Grid search result:")
print(f"  kappa1 = {best_k1_g:.6f}")
print(f"  kappa2 = {best_k2_g:.6f}")
print(f"  sigma  = {best_sigma_g:.6f}")

r_best_g = tip_position(best_k1_g, best_k2_g, L0)
theta_best_g = total_theta(best_k1_g, best_k2_g, L0)
D2_best_g = distance_squared(best_k1_g, best_k2_g, L0, d0)

print(f"  r_tip  = {r_best_g}")
print(f"  theta  = {theta_best_g:.6f} rad")
print(f"  D      = {np.sqrt(D2_best_g):.6f}")

print(target)
print(tangent_tip(theta_best_g))
print(target - tip_position(best_k1_g,best_k2_g, L0))
print(np.dot(tangent_tip(theta_best_g),target - tip_position(best_k1_g,best_k2_g, L0)))
"""
# -----------------------------
# Global optimization
# -----------------------------
bounds = [kappa1_bounds, kappa2_bounds]

result_de = differential_evolution(
    objective_with_penalty,
    bounds=bounds,
    args=(L0, d0, theta_block1, theta_block2, target),
    polish=True
)

k1_de, k2_de = result_de.x
D2_de = distance_squared(k1_de, k2_de, L0, d0)
theta_de = total_theta(k1_de, k2_de, L0)
sigma_de = sigma(k1_de, k2_de, L0, d0, theta_block1, theta_block2)

print("\nDifferential evolution result:")
print(f"  kappa1 = {k1_de:.6f}")
print(f"  kappa2 = {k2_de:.6f}")
print(f"  sigma  = {sigma_de:.6f}")
print(f"  D      = {np.sqrt(D2_de):.6f}")
print(f"  theta  = {theta_de:.6f} rad")
print(f"  r_tip  = {tip_position(k1_de, k2_de, L0)}")


# -----------------------------
# Optional local refinement
# -----------------------------
result_local = minimize(
    objective_with_penalty,
    x0=result_de.x,
    args=(L0, d0, theta_block1, theta_block2,target),
    method="Nelder-Mead"
)

k1_local, k2_local = result_local.x
D2_local = distance_squared(k1_local, k2_local, L0, d0)
theta_local = total_theta(k1_local, k2_local, L0)
sigma_local = sigma(k1_local, k2_local, L0, d0, theta_block1, theta_block2)

print("\nLocal refinement result:")
print(f"  kappa1 = {k1_local:.6f}")
print(f"  kappa2 = {k2_local:.6f}")
print(f"  sigma  = {sigma_local:.6f}")
print(f"  D      = {np.sqrt(D2_local):.6f}")
print(f"  theta  = {theta_local:.6f} rad")
print(f"  r_tip  = {tip_position(k1_local, k2_local, L0)}")



"""
# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(8, 6))
plt.imshow(
    sigma_map,
    extent=[kappa1_bounds[0], kappa1_bounds[1], kappa2_bounds[0], kappa2_bounds[1]],
    origin='lower',
    aspect='auto'
)

plt.colorbar(label='sigma')
plt.scatter([best_k1_g], [best_k2_g], marker='x', s=100, label='Grid optimum')
#plt.scatter([k1_de], [k2_de], marker='o', s=60, label='DE optimum')
plt.xlabel('kappa1')
plt.ylabel('kappa2')
plt.title('Objective landscape')
plt.legend()
plt.show()

from visualization import plot_two_segment_setup
plot_two_segment_setup(best_k1_g, best_k2_g, L0, d0, theta_block2, theta_block1)