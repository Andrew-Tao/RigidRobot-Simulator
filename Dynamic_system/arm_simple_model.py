import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
m1  = 1
k1  = 4
c1 = 30

k2 = 200
c2 = 20

m = m1
k = k1 + k2
c = c1 + c2


def x2(t):
    return 10*np.sin(2*np.pi*t/4) -10

def make_force(x2_func, k2, c2, dt=1e-5):

    def x2_dot(t):
        return (x2_func(t + dt) - x2_func(t - dt)) / (2*dt)

    def force(t):

        return min(k2 * x2_func(t) + c2 * x2_dot(t),0)

    return force

force = make_force(x2, k2, c2)


# Define ODE system
def ode_system(t,y):

    u = y[0]
    v = y[1]

    a = (force(t) -(c * v) - (k * u))/m

    return np.array([v,a])


y0 = np.array([0,0])

t_eval = np.linspace(0, 10, 2000)

sol = solve_ivp(
    ode_system,
    (t_eval[0], t_eval[-1]),
    y0,
    t_eval=t_eval,
)

t = sol.t
x1 = sol.y[0]
x2_vals = np.array([x2(ti) for ti in t])
force_vals = np.array([force(t) for t in t])

# ====================================================
# Plot positions
# ====================================================
# =========================
# Plot BOTH
# =========================
plt.figure(figsize=(8, 5))
plt.plot(t, x1, label='x1(t) (response)')
plt.plot(t, x2_vals, '--', label='x2(t) (input motion)')
plt.plot(t, force_vals/1e8*5, '--', label='force(t) (input motion)')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('x1 vs x2')
plt.grid(True)
plt.legend()
plt.show()
