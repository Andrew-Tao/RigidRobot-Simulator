import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge



# ============================================================
# GEOMETRY WITH THETA=0 ALONG +Y (VERTICAL UP)
# ============================================================
def rot_from_vertical(theta):
    """
    Unit direction vector when theta = 0 points upward.
    """
    return np.array([np.sin(theta), np.cos(theta)])


def segment_tip_vertical(kappa, L, theta0):
    """
    Tip position of one constant-curvature segment in global coordinates,
    with theta measured from +Y axis.
    """
    if abs(kappa) < EPS:
        return L * rot_from_vertical(theta0)

    theta1 = theta0 + kappa * L
    x = (-np.cos(theta1) + np.cos(theta0)) / kappa
    y = ( np.sin(theta1) - np.sin(theta0)) / kappa
    return np.array([x, y])


def segment_points_vertical(kappa, L, theta0, n=300):
    """
    Sample points along one segment in global coordinates.
    """
    s_vals = np.linspace(0, L, n)

    if abs(kappa) < EPS:
        x = s_vals * np.sin(theta0)
        y = s_vals * np.cos(theta0)
        return x, y

    theta_s = theta0 + kappa * s_vals
    x = (-np.cos(theta_s) + np.cos(theta0)) / kappa
    y = ( np.sin(theta_s) - np.sin(theta0)) / kappa
    return x, y


def two_segment_shape(c1, c2, L1, L2, n=300):
    """
    Returns points for segment 1 and segment 2, plus joint/tip/theta.
    """
    # Segment 1 starts at origin, theta0 = 0 (straight up)
    x1, y1 = segment_points_vertical(c1, L1, theta0=0.0, n=n)
    joint = np.array([x1[-1], y1[-1]])

    # Segment 2 starts at joint, with base angle = c1*L0
    theta1 = c1 * L1
    x2_local, y2_local = segment_points_vertical(c2, L2, theta0=theta1, n=n)

    # local formula already starts from origin, so shift by joint
    x2 = x2_local + joint[0]
    y2 = y2_local + joint[1]

    tip = np.array([x2[-1], y2[-1]])
    theta_tip = (c1 * L1 + c2 * L2)

    return x1, y1, x2, y2, joint, tip, theta_tip


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def is_blocked(theta, theta_b1, theta_b2):
    """
    Checks if theta lies in blocked interval.
    Assumes theta_b1 <= theta_b2 in the same branch.
    """
    return theta_b1 <= theta <= theta_b2


# ============================================================
# PLOTTING
# ============================================================
def draw_vertical_angle_ray(ax, theta, radius, origin=(0.0, 0.0), **kwargs):
    """
    Draw ray starting from a custom origin.
    Theta is measured from +Y axis.
    """
    x0, y0 = origin

    dx = radius * np.sin(theta)
    dy = radius * np.cos(theta)

    ax.plot([x0, x0 + dx], [y0, y0 + dy], **kwargs)

def plot_two_segment_setup(c1, c2, L1, L2, target, theta_b1, theta_b2):
    x1, y1, x2, y2, joint, tip, theta_tip = two_segment_shape(c1, c2, L1, L2, n=400)
    target_tem = np.array([target[1], target[0]])
    target = target_tem
    D = np.linalg.norm(tip - target)

    all_x = np.concatenate([x1, x2, [0, target[0]]])
    all_y = np.concatenate([y1, y2, [0, target[1]]])

    span = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), 0.5)
    R = 1.25 * span + 0.4

    fig, ax = plt.subplots(figsize=(8, 8))

    # blocked region as wedge
    # matplotlib wedge uses degrees from +x axis CCW
    # our theta is from +y axis, so convert:
    # phi_deg = 90 - theta_deg
    phi1 = (90-np.degrees(theta_b1)+180)
    phi2 = (90-np.degrees(theta_b2)+180)

    # Wedge goes CCW from theta1 to theta2 in matplotlib coordinates.
    # To fill the smaller intended sector, sort carefully.
    wedge = Wedge(
        center= tuple(target),
        r=R,
        theta1=phi2,
        theta2=phi1,
        facecolor='red',
        alpha=0.15,
        edgecolor='none'
    )
    ax.add_patch(wedge)

    # blocked boundary rays
    draw_vertical_angle_ray(ax, theta_b1+np.pi, R, color='red', origin= target, linestyle='--', linewidth=1.8)
    draw_vertical_angle_ray(ax, theta_b2+np.pi, R, color='red', origin= target, linestyle='--', linewidth=1.8)

    # current theta ray
    draw_vertical_angle_ray(ax, theta_tip, R, color='green', origin= tip,linewidth=2.5, label='Tip angle')

    # robot segments
    ax.plot(x1, y1, linewidth=4, label='Segment 1')
    ax.plot(x2, y2, linewidth=4, label='Segment 2')

    # base, joint, tip, target
    ax.scatter([0], [0], s=80, color='black', zorder=5, label='Base')
    ax.scatter([joint[0]], [joint[1]], s=60, color='purple', zorder=5, label='Joint')
    ax.scatter([tip[0]], [tip[1]], s=90, color='orange', zorder=6, label='Tip')
    ax.scatter([target[0]], [target[1]], s=110, marker='x', color='red', zorder=6, label='Target')

    # distance line
    ax.plot([tip[0], target[0]], [tip[1], target[1]],
            color='gray', linestyle=':', linewidth=2)

    blocked_flag = is_blocked(theta_tip, theta_b1, theta_b2)

    title = (
        f"Two-Segment Setup\n"
        f"c1={c1:.3f}, c2={c2:.3f}, L0={L0:.3f}, d0={d0:.3f}\n"
        f"theta_tip={(theta_tip):.3f} rad, D={D:.4f}"
    )
    if blocked_flag:
        title += "   [BLOCKED]"
    ax.set_title(title)

    ax.text(tip[0], tip[1], "  tip", fontsize=11)
    ax.text(target[0], target[1], "  target", fontsize=11)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    ax.set_xlim(-R, R)
    ax.set_ylim(-0.2 * R, R)

    plt.show()


EPS = 1e-9


# ============================================================
# USER INPUTS
# ============================================================
c1 =  -0.686747     # curvature of segment 1
c2 = 1.096386  # curvature of segment 2
L0 = 1.0          # length of each segment
d0 = 3         # target at [0, d0]

theta_b1 = -0.5    # blocked angle start (rad), measured from vertical
theta_b2 = -0.4    # blocked angle end   (rad), measured from vertical

# ============================================================
# RUN
# ============================================================
#plot_two_segment_setup(c1, c2, L0, d0, theta_b1, theta_b2)