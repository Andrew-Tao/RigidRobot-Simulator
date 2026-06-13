"""
One Point Mass Vertical Spring-Damper Simulator
==============================================

System:
    Base is fixed at y = 0.
    One point mass is connected to the base by one spring-damper.

Coordinate convention:
    y is a signed vertical coordinate.
    Positive y points upward.
    Negative y points downward.

Therefore, if the mass is hanging below the base, its y position is negative.
For example:
    base: y = 0
    mass: y = -0.04

This simulator uses geometric spring length:
    spring length = abs(y)

So if:
    y0 = -L
    v0 = 0
    g = 0

then the spring is exactly at rest length and the system is exactly static.

Run:
    python one_mass_vertical_spring_simulator.py

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
    m: float = 0.0855        # kg, mass
    k: float = 1.0           # N/m, spring stiffness
    c: float = 0.1           # N*s/m, damping coefficient
    L: float = 0.04          # m, spring rest length
    g: float = 0.0           # m/s^2, gravity magnitude. Gravity force is downward, so force = -m*g.

    # Initial conditions
    # Because positive y is upward, hanging below the base means negative y.
    y0: float = -0.04        # m, initial position
    v0: float = 0.0          # m/s, initial velocity

    # Simulation time
    t_start: float = 0.0
    t_end: float = 20.0
    num_points: int = 3000

    # Optional external force pulse
    # Positive force points upward.
    # Negative force points downward.
    force_amplitude: float = 0.0
    force_start_time: float = 0.0
    force_end_time: float = 0.0

    # Plot and animation controls
    show_animation: bool = True
    animation_speed: float = 1.0
    save_animation: bool = False
    animation_filename: str = "one_mass_vertical_animation.gif"


# ============================================================
# Simulator
# ============================================================

class OneMassVerticalSpringSimulator:
    def __init__(self, settings: SimulationSettings):
        self.s = settings
        self.t_eval = np.linspace(self.s.t_start, self.s.t_end, self.s.num_points)
        self.solution = None

    @staticmethod
    def safe_unit(value: float, fallback: float = -1.0):
        """
        Return the 1D direction of the spring.

        If the mass is below the base, y < 0 and unit = -1.
        If the mass is above the base, y > 0 and unit = +1.
        The fallback prevents division by zero when y is exactly 0.
        """
        if abs(value) < 1e-12:
            return fallback
        return np.sign(value)

    def external_force(self, t: float):
        """
        Optional rectangular pulse force.

        Positive force points upward.
        Negative force points downward.
        """
        if self.s.force_start_time <= t <= self.s.force_end_time:
            return self.s.force_amplitude
        return 0.0

    def dynamics(self, t: float, state: np.ndarray):
        """
        State vector:
            state = [y, v]

        Coordinate convention:
            y > 0 means above the base.
            y < 0 means below the base.
            Gravity points in the negative-y direction.
        """
        y, v = state
        s = self.s

        # Spring geometry
        r = y - 0.0               # signed vector from base to mass
        length = abs(r)           # actual spring length
        unit = self.safe_unit(r)  # direction from base to mass

        # Spring extension
        extension = length - s.L

        # Velocity projected along spring direction
        relative_speed = v * unit

        # Spring-damper force on the mass
        # If the spring is stretched, it pulls the mass back toward the base.
        F_spring_damper = -(s.k * extension + s.c * relative_speed) * unit

        # Gravity force
        F_gravity = -s.m * s.g

        # Optional external force
        F_external = self.external_force(t)

        # Total force and acceleration
        F_total = F_spring_damper + F_gravity + F_external
        a = F_total / s.m

        return np.array([v, a], dtype=float)

    def run(self):
        """Solve the ODE system."""
        initial_state = np.array([self.s.y0, self.s.v0], dtype=float)

        # Print initial acceleration to verify static cases.
        initial_derivative = self.dynamics(self.s.t_start, initial_state)
        print("Initial acceleration check:")
        print(f"  a(0) = {initial_derivative[1]: .12e} m/s^2")

        self.solution = solve_ivp(
            fun=self.dynamics,
            t_span=(self.s.t_start, self.s.t_end),
            y0=initial_state,
            t_eval=self.t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )

        if not self.solution.success:
            raise RuntimeError(f"ODE solver failed: {self.solution.message}")

        return self.solution

    def static_equilibrium_position(self, hanging_down=True):
        """
        Static equilibrium under gravity.

        Positive y = upward.
        Gravity points downward.

        For a hanging mass below the base:
            spring extension = m*g/k
            equilibrium length = L + m*g/k
            y_eq = -(L + m*g/k)
        """
        s = self.s
        extension_eq = s.m * s.g / s.k
        length_eq = s.L + extension_eq
        direction = -1.0 if hanging_down else 1.0
        y_eq = direction * length_eq
        return y_eq

    def get_results(self):
        """Return time, position, velocity, and acceleration."""
        if self.solution is None:
            raise RuntimeError("Run the simulation first using simulator.run().")

        t = self.solution.t
        y, v = self.solution.y

        acceleration = np.array([
            self.dynamics(ti, self.solution.y[:, i])[1]
            for i, ti in enumerate(t)
        ])

        return t, y, v, acceleration

    def spring_length_and_extension(self, y):
        """Compute geometric spring length and extension."""
        length = np.abs(y)
        extension = length - self.s.L
        return length, extension

    def plot_results(self):
        """Plot position."""
        t, y, v, a = self.get_results()
        y_eq = self.static_equilibrium_position(hanging_down=True)
        length, extension = self.spring_length_and_extension(y)

        print("Starting position check:")
        print(f"  y_start = {y[0]: .12f} m")
        print(f"  spring length start = {length[0]: .12f} m")
        print(f"  spring extension start = {extension[0]: .12e} m")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(t, y, label="mass position y")
        ax.scatter([t[0]], [y[0]], marker="o", label="start position")
        ax.axhline(y_eq, linestyle="--", label="static equilibrium")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position y [m]")
        ax.set_title("One Mass Vertical Spring-Damper System")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def animate(self):
        """Animate the one-mass vertical system."""
        if self.solution is None:
            raise RuntimeError("Run the simulation first using simulator.run().")

        t, y, *_ = self.get_results()

        # Downsample for smoother animation performance
        step = max(1, len(t) // 600)
        t_anim = t[::step]
        y_anim = y[::step]

        y_min = min(np.min(y_anim), 0.0) - 0.1
        y_max = max(np.max(y_anim), 0.0) + 0.1

        fig, ax = plt.subplots(figsize=(5, 8))
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title("One Mass Spring-Damper Animation")
        ax.set_ylabel("Vertical position y [m]")

        ax.axhline(0.0, linewidth=1)
        base_point, = ax.plot([0], [0], marker="s", markersize=12, label="fixed base")
        spring_line, = ax.plot([], [], linewidth=2, label="spring")
        mass_point, = ax.plot([], [], marker="o", markersize=18, label="mass")
        time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes)
        ax.legend(loc="best")

        def make_spring_x(y_start, y_end, amplitude=0.025, coils=8):
            ys = np.linspace(y_start, y_end, 200)
            xs = amplitude * np.sin(np.linspace(0, 2 * np.pi * coils, 200))
            xs[0] = 0.0
            xs[-1] = 0.0
            return xs, ys

        def update(frame):
            y_now = y_anim[frame]
            x_s, y_s = make_spring_x(0.0, y_now)

            spring_line.set_data(x_s, y_s)
            mass_point.set_data([0], [y_now])
            time_text.set_text(f"t = {t_anim[frame]:.2f} s")

            return spring_line, mass_point, time_text

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
        # Mass
        m=0.0855,

        # Spring-damper
        k=1.0,
        c=0.1,
        L=0.04,

        # Gravity
        # If g=0 and y0=-L with v0=0, the mass stays perfectly static.
        g=0.0,

        # Initial condition
        y0=-0.04,
        v0=0.0,

        # Time
        t_end=20.0,
        num_points=3000,

        # Optional external force pulse
        force_amplitude=-0.01,
        force_start_time=0.0,
        force_end_time=100000,

        # Animation
        show_animation=True,
        save_animation=False,
    )

    simulator = OneMassVerticalSpringSimulator(settings)
    simulator.run()

    y_eq = simulator.static_equilibrium_position(hanging_down=True)
    print("Static equilibrium position:")
    print(f"  y_eq = {y_eq:.12f} m")

    simulator.plot_results()

    if settings.show_animation:
        simulator.animate()
