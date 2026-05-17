import numpy as np 
from methods2D import SE2LieAlgebra
import matplotlib.pyplot as plt
lie2 = SE2LieAlgebra()

class RigidRobot2D:
    def __init__(
            self, 
            position, 
            orientation, 
            mass, 
            inertia, 
            linear_velocity=np.zeros(2), 
            angular_velocity=0.0,
            radius = 0.2,
            track_width_between_wheels = 0.15,
        ):

        """
        position: (x, y) in R^2
        orientation: theta in radians
        mass: scalar mass of the robot
        inertia: scalar moment of inertia around the center of mass
        linear_velocity: (v_x, v_y) in R^2
        angular_velocity: omega in radians per second
        environment_resitriction: for environment interaction logic
        """

        self.radius = radius
        self.track_width_between_wheels = track_width_between_wheels
        self.posture = self.compute_posture(position, orientation)
        self.velocity_matrix = self.compute_velocity_matrix(linear_velocity, angular_velocity)
        self.mass_matrix = np.diag([mass, mass, inertia])  # Mass matrix in SE(2)
        self.momentum = self.mass_matrix @ lie2.vee(self.velocity_matrix)  # 3-vector μ = M·ξ
        self.environment_resitriction = None
        self.force = None  # Generalized force in R^3 vector form [f_x, f_y, tau]
        self.control_input = np.zeros(2)  # Control input (force_l, force_r)
        self.friction_coefficient = 0.6  # Simple friction model

    def compute_posture(self, position, orientation):
        c, s = np.cos(orientation), np.sin(orientation)
        R = np.array([[c, -s], [s, c]])
        T = np.eye(3)
        T[:2,:2] = R
        T[:2,2] = position
        return T

    def compute_velocity_matrix(self, linear_velocity, angular_velocity):
        velocity_vector = np.array([linear_velocity[0], linear_velocity[1], angular_velocity])
        velocity_matrix = lie2.hat(velocity_vector)
        return velocity_matrix

    def compute_force_local(self, control_input):
        #TODO：Implement Contact model with the Wall and Friction

        v1, v2 = self.velocity_matrix[:2, 2]   # body-frame linear velocity
        omega   = self.velocity_matrix[1, 0]    # angular velocity
        d  = self.track_width_between_wheels
        mu = self.friction_coefficient

        # Wheel longitudinal velocities in body frame
        v_l = v1 - omega * (d / 2)
        v_r = v1 + omega * (d / 2)


        force_l = control_input[0] - mu * v_l
        force_r = control_input[1] - mu * v_r
        #print(f"v_l: {v_l:.3f}, v_r: {v_r:.3f}, force_l: {force_l:.3f}, force_r: {force_r:.3f}")

        f_x = force_l + force_r
        #print(f"force_l: {force_l:.3f}, force_r: {force_r:.3f}, f_x: {f_x:.3f}")
        f_y = 0.0
        tau = (force_r - force_l) * (d / 2)

        return np.array([f_x, f_y, tau])

    def contact(self, environment):
        """
        environment: func (position) -> True if within boundary, False otherwise)
        position in R^2
        """
        self.environment_resitriction = environment

    def is_in_contact(self):
        robot_current_center_position = self.posture[:2, 2]
        radius_vector = np.array([0, self.radius])  
        for theta in np.linspace(0, 2 * np.pi, num=8):  # Check 8 points around the circumference
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[c, -s], [s, c]])
            radius_vector_rotated = rotation_matrix @ radius_vector
            contact_point = robot_current_center_position + radius_vector_rotated
            if not self.environment_resitriction(contact_point):
                return True  # Contact detected
            
        return False  # No contact detected
    
    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(6, 8))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#0f0f1a')

        # ── I-shape environment ────────────────────────────────────────
        i_verts = np.array([
            [0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.4, 1.0],
            [2.4, 3.0], [3.0, 3.0], [3.0, 4.0], [0.0, 4.0],
            [0.0, 3.0], [0.6, 3.0], [0.6, 1.0], [0.0, 1.0],
        ])
        ax.add_patch(MplPolygon(i_verts, closed=True,
                                facecolor='#1e2a4a', edgecolor='#56b4f7',
                                linewidth=2.0, zorder=1))

        # ── extract state ──────────────────────────────────────────────
        pos     = self.posture[:2, 2]
        R       = self.posture[:2, :2]
        heading = R[:, 0]   # robot x-axis (forward)
        lateral = R[:, 1]   # robot y-axis (left)
        omega   = self.velocity_matrix[1, 0]
        lin_vel = np.array([self.velocity_matrix[0, 2], self.velocity_matrix[1, 2]])
        speed   = np.linalg.norm(lin_vel)

        # ── wheels ─────────────────────────────────────────────────────
        wl, ww = 0.08, 0.04   # length along heading, width along lateral
        for sign in (+1, -1):
            wc = pos + sign * (self.track_width_between_wheels / 2) * lateral
            verts = np.array([
                wc + wl / 2 * heading + ww / 2 * lateral,
                wc + wl / 2 * heading - ww / 2 * lateral,
                wc - wl / 2 * heading - ww / 2 * lateral,
                wc - wl / 2 * heading + ww / 2 * lateral,
            ])
            ax.add_patch(MplPolygon(verts, closed=True,
                                    facecolor='#3a3a5c', edgecolor='#aaaacc',
                                    linewidth=1.2, zorder=4))

        # ── robot body ─────────────────────────────────────────────────
        ax.add_patch(mpatches.Circle(pos, self.radius,
                                     facecolor='#c0392b', edgecolor='white',
                                     linewidth=1.8, zorder=5))

        # ── orientation vectors ────────────────────────────────────────
        arr_len  = self.radius * 1.8
        label_d  = arr_len + 0.07
        ax.annotate('', xy=pos + heading * arr_len, xytext=pos,
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b',
                                    lw=2.2, mutation_scale=14), zorder=7)
        ax.annotate('', xy=pos + lateral * arr_len, xytext=pos,
                    arrowprops=dict(arrowstyle='->', color='#4fc3f7',
                                    lw=2.2, mutation_scale=14), zorder=7)
        ax.text(*(pos + heading * label_d), 'x̂', color='#ff6b6b',
                fontsize=10, fontweight='bold', ha='center', va='center', zorder=8)
        ax.text(*(pos + lateral * label_d), 'ŷ', color='#4fc3f7',
                fontsize=10, fontweight='bold', ha='center', va='center', zorder=8)

        # ── linear velocity arrow ──────────────────────────────────────
        if speed > 1e-4:
            ax.annotate('', xy=pos + lin_vel * 0.25, xytext=pos,
                        arrowprops=dict(arrowstyle='->', color='#f9c74f',
                                        lw=2.5, mutation_scale=14), zorder=7)
            mid = pos + lin_vel * 0.14
            ax.text(mid[0] + 0.04, mid[1] + 0.04,
                    f'|v|={speed:.2f} m/s', color='#f9c74f',
                    fontsize=7.5, fontweight='bold', zorder=9)

        # ── angular velocity arc ───────────────────────────────────────
        if np.abs(omega) > 1e-4:
            arc_r    = self.radius * 1.5
            arc_deg  = np.clip(np.abs(omega) * 60, 45, 260)
            direction = np.sign(omega)
            t_arc = np.linspace(np.radians(20),
                                np.radians(20 + direction * arc_deg), 80)
            ax.plot(pos[0] + arc_r * np.cos(t_arc),
                    pos[1] + arc_r * np.sin(t_arc),
                    color='#90be6d', lw=2.0, zorder=6)
            dt  = t_arc[-1] - t_arc[-2]
            tip = np.array([pos[0] + arc_r * np.cos(t_arc[-1]),
                            pos[1] + arc_r * np.sin(t_arc[-1])])
            tangent = np.array([-np.sin(t_arc[-1]),
                                 np.cos(t_arc[-1])]) * np.sign(dt)
            ax.annotate('', xy=tip + tangent * 1e-3, xytext=tip - tangent * 0.02,
                        arrowprops=dict(arrowstyle='->', color='#90be6d',
                                        lw=2.0, mutation_scale=12), zorder=7)
            ax.text(pos[0], pos[1] - arc_r - 0.10,
                    f'ω={omega:.2f} rad/s', color='#90be6d',
                    fontsize=7.5, fontweight='bold', ha='center', zorder=9)

        # ── axes, grid, labels ─────────────────────────────────────────
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(-0.35, 4.35)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', color='#888899', fontsize=10)
        ax.set_ylabel('y (m)', color='#888899', fontsize=10)
        ax.set_title('2D Rigid Robot — Current Frame', color='white',
                     fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors='#888899')
        for spine in ax.spines.values():
            spine.set_color('#333344')
        ax.grid(True, color='#1e2a3a', linewidth=0.6, linestyle='--')

        # ── legend ─────────────────────────────────────────────────────
        ax.legend(handles=[
            Line2D([0], [0], color='#ff6b6b', lw=2,   label='Heading x̂'),
            Line2D([0], [0], color='#4fc3f7', lw=2,   label='Lateral ŷ'),
            Line2D([0], [0], color='#f9c74f', lw=2.5, label='Linear velocity'),
            Line2D([0], [0], color='#90be6d', lw=2,   label='Angular velocity'),
        ], loc='upper right', facecolor='#1e2a4a', edgecolor='#56b4f7',
           labelcolor='white', fontsize=8, framealpha=0.9)

        plt.tight_layout()
        plt.show()
        
