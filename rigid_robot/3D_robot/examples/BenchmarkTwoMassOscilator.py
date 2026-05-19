"""
Two Point Masses in Series: Vertical Spring-Damper Simulator
===========================================================

System:
    Base is fixed at y = 0.
    Mass 1 hangs below the base.
    Mass 2 hangs below mass 1.

Coordinates:
    y is positive downward.

Connections:
    base ---- spring/damper 1 ---- mass 1 ---- spring/damper 2 ---- mass 2

Forces:
    Gravity acts in +y direction.
    Spring force depends on extension relative to rest length.
    Damping force depends on relative velocity across each spring.

Run:
    python two_mass_series_spring_simulator.py

Dependencies:
    numpy
    scipy
    matplotlib

Install if needed:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from dataclasses import dataclass


# ============================================================
# User Settings
# ============================================================

@dataclass
class SimulationSettings:
    # Physical parameters
    m1: float = 0.0855       # kg, mass of top mass
    m2: float = 0.0855      # kg, mass of bottom mass
    k1: float = 1.0 # N/m, stiffness between base and mass 1
    k2: float = 1.0         # N/m, stiffness between mass 1 and mass 2
    c1: float = 0.1          # N*s/m, damping between base and mass 1
    c2: float = 0.1          # N*s/m, damping between mass 1 and mass 2
    L1: float = 0.04          # m, rest length of spring 1
    L2: float = 0.04         # m, rest length of spring 2
    g: float = 0.0         # m/s^2, gravity

    # Initial conditions
    y1_0: float = 0.04      # m, initial position of mass 1, positive downward
    y2_0: float = 0.08       # m, initial position of mass 2, positive downward

    v1_0: float = 0.0        # m/s, initial velocity of mass 1
    v2_0: float = 0.0        # m/s, initial velocity of mass 2

    # Simulation time
    t_start: float = 0.0     # s
    t_end: float = 20.0      # s
    num_points: int = 3000   # number of time samples

    # External forces
    # These are extra forces applied directly to each mass.
    # Positive means downward, negative means upward.
    force1_amplitude: float = 0.0      # N
    force2_amplitude: float = 0.0      # N
    force_start_time: float = 0.0      # s
    force_end_time: float = 0.0        # s

    # Plot and animation controls
    show_animation: bool = True
    animation_speed: float = 1.0       # larger = faster playback
    save_animation: bool = False
    animation_filename: str = "two_mass_series_animation.gif"


# ============================================================
# Simulator
# ============================================================

class TwoMassSeriesSpringSimulator:
    def __init__(self, settings: SimulationSettings):
        self.s = settings
        self.t_eval = np.linspace(self.s.t_start, self.s.t_end, self.s.num_points)
        self.solution = None

    def external_force(self, t: float):
        """
        Optional rectangular pulse force.
        Positive force points downward.
        """
        if self.s.force_start_time <= t <= self.s.force_end_time:
            return self.s.force1_amplitude, self.s.force2_amplitude
        return 0.0, 0.0

    def dynamics(self, t: float, state: np.ndarray):
        """
        State vector:
            state = [y1, y2, v1, v2]

        where:
            y1 = downward position of mass 1
            y2 = downward position of mass 2
            v1 = downward velocity of mass 1
            v2 = downward velocity of mass 2
        """
        y1, y2, v1, v2 = state
        s = self.s

        # Spring 1 extension and relative velocity
        # Spring 1 connects fixed base y = 0 to mass 1.
        x1 = y1 - s.L1
        x1_dot = v1

        # Spring 2 extension and relative velocity
        # Spring 2 connects mass 1 to mass 2.
        x2 = (y2 - y1) - s.L2
        x2_dot = v2 - v1

        # Force from spring/damper 1 on mass 1.
        # If y1 > L1, spring 1 is stretched and pulls mass 1 upward, so negative.
        F_spring1_on_m1 = -s.k1 * x1 - s.c1 * x1_dot

        # Force from spring/damper 2.
        # If y2 - y1 > L2, spring 2 is stretched.
        # It pulls mass 1 downward and mass 2 upward.
        F_spring2_on_m1 = s.k2 * x2 + s.c2 * x2_dot
        F_spring2_on_m2 = -s.k2 * x2 - s.c2 * x2_dot

        # Optional external force input
        F_ext1, F_ext2 = self.external_force(t)

        # Total forces, positive downward
        F1 = s.m1 * s.g + F_spring1_on_m1 + F_spring2_on_m1 + F_ext1
        F2 = s.m2 * s.g + F_spring2_on_m2 + F_ext2

        # Accelerations
        a1 = F1 / s.m1
        a2 = F2 / s.m2

        return np.array([v1, v2, a1, a2])

    def run(self):
        """Solve the ODE system."""
        initial_state = np.array([
            self.s.y1_0,
            self.s.y2_0,
            self.s.v1_0,
            self.s.v2_0,
        ], dtype=float)

        self.solution = solve_ivp(
            fun=self.dynamics,
            t_span=(self.s.t_start, self.s.t_end),
            y0=initial_state,
            t_eval=self.t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not self.solution.success:
            raise RuntimeError(f"ODE solver failed: {self.solution.message}")

        return self.solution

    def equilibrium_positions(self):
        """
        Static equilibrium under gravity.

        For the lower spring:
            k2 * extension2 = m2 * g

        For the upper spring:
            k1 * extension1 = (m1 + m2) * g
        """
        s = self.s
        y1_eq = s.L1 + (s.m1 + s.m2) * s.g / s.k1
        y2_eq = y1_eq + s.L2 + s.m2 * s.g / s.k2
        return y1_eq, y2_eq

    def get_results(self):
        """Return time, positions, velocities, and accelerations."""
        if self.solution is None:
            raise RuntimeError("Run the simulation first using simulator.run().")

        t = self.solution.t
        y1, y2, v1, v2 = self.solution.y

        accelerations = np.array([
            self.dynamics(ti, self.solution.y[:, i])[2:4]
            for i, ti in enumerate(t)
        ])
        a1 = accelerations[:, 0]
        a2 = accelerations[:, 1]

        return t, y1, y2, v1, v2, a1, a2

    def plot_results(self):
        """Plot positions, velocities, and spring extensions."""
        t, y1, y2, v1, v2, a1, a2 = self.get_results()
        s = self.s

        y1_eq, y2_eq = self.equilibrium_positions()

        extension1 = y1 - s.L1
        extension2 = (y2 - y1) - s.L2

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(t, -y1, label="mass 1 position y1")
        ax.plot(t, -y2, label="mass 2 position y2")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position y [m]")
        ax.set_title("Two Masses Hanging in Series")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def animate(self):
        """Animate the two-mass hanging system."""
        if self.solution is None:
            raise RuntimeError("Run the simulation first using simulator.run().")

        t, y1, y2, *_ = self.get_results()

        # Downsample for smoother animation performance
        step = max(1, len(t) // 600)
        t_anim = t[::step]
        y1_anim = y1[::step]
        y2_anim = y2[::step]

        y_max = max(np.max(y2_anim) + 0.5, self.s.L1 + self.s.L2 + 1.0)
        y_min = -0.2

        fig, ax = plt.subplots(figsize=(5, 8))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(y_max, y_min)  # inverted so downward appears downward visually
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title("Two Mass Series Spring-Damper Animation")
        ax.set_ylabel("Vertical position y [m]")

        base_point, = ax.plot([0], [0], marker="s", markersize=12, label="fixed base")
        spring1_line, = ax.plot([], [], linewidth=2, label="spring 1")
        spring2_line, = ax.plot([], [], linewidth=2, label="spring 2")
        mass1_point, = ax.plot([], [], marker="o", markersize=18, label="mass 1")
        mass2_point, = ax.plot([], [], marker="o", markersize=18, label="mass 2")
        time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes)
        ax.legend(loc="lower right")

        def make_spring_x(y_start, y_end, amplitude=0.08, coils=10):
            ys = np.linspace(y_start, y_end, 200)
            xs = amplitude * np.sin(np.linspace(0, 2 * np.pi * coils, 200))
            xs[0] = 0.0
            xs[-1] = 0.0
            return xs, ys

        def update(frame):
            y1_now = y1_anim[frame]
            y2_now = y2_anim[frame]

            x_s1, y_s1 = make_spring_x(0.0, y1_now)
            x_s2, y_s2 = make_spring_x(y1_now, y2_now)

            spring1_line.set_data(x_s1, y_s1)
            spring2_line.set_data(x_s2, y_s2)
            mass1_point.set_data([0], [y1_now])
            mass2_point.set_data([0], [y2_now])
            time_text.set_text(f"t = {t_anim[frame]:.2f} s")

            return spring1_line, spring2_line, mass1_point, mass2_point, time_text

        interval_ms = 1000 * (t_anim[1] - t_anim[0]) / self.s.animation_speed if len(t_anim) > 1 else 20
        anim = FuncAnimation(
            fig,
            update,
            frames=len(t_anim),
            interval=interval_ms,
            blit=True,
        )

        if self.s.save_animation:
            anim.save(self.s.animation_filename, writer="pillow", fps=30)
            print(f"Animation saved to: {self.s.animation_filename}")

        plt.show()
        return anim


# ============================================================
# Main Run Script
# ============================================================

if __name__ == "__main__":
    settings = SimulationSettings(
        # Masses
        m1=0.0855,
        m2=0.0855,

        # Springs
        k1=1.0,
        k2=1.0,
        L1=0.04,
        L2=0.04,

        # Dampers
        c1=0.1,
        c2=0.1,

        # Initial conditions
        y1_0=0.04,
        y2_0=0.08,
        v1_0=0.0,
        v2_0=0.0,

        # Time
        t_end=20.0,
        num_points=3000,

        # Optional force pulse example
        # Set amplitudes and time window to create a disturbance.
        force1_amplitude=0.0,
        force2_amplitude=0.01,
        force_start_time=0.0,
        force_end_time=1000,

        # Animation
        show_animation=False,
        save_animation=False,
    )

    simulator = TwoMassSeriesSpringSimulator(settings)
    simulator.run()

    y1_eq, y2_eq = simulator.equilibrium_positions()
    print("Static equilibrium positions:")
    print(f"  y1_eq = {y1_eq:.6f} m")
    print(f"  y2_eq = {y2_eq:.6f} m")

    simulator.plot_results()

    if settings.show_animation:
        simulator.animate()
