import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from robot3d.robot.ConnectedRigidRobot3D import ConnectedRigidRobots3D
from robot3d.robot.RigidRobot3D import RigidRobot3D
from robot3d.stepper.SimulatorConnectedRobot3D import MutiRobotSimulator3D
from robot3d.stepper.SimulatorRigidBody import RigidObjectSimulator
from robot3d.force.CableDrivenForce import CableDrivenForce, GravityForce
import numpy as np
from robot3d.methods3D import SE3LieAlgebra, rot_x, rot_y, rot_z
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from robot3d.contact.DiskSelfContact import SelfAngleContact

lie3 = SE3LieAlgebra()

# Solid Object Now can only be cylinder just as the RigidRobot3D 
class SolidObject(RigidRobot3D):
    def __init__(self, position, orientation, mass, inertia, linear_velocity=..., angular_velocity=..., radius=0.2, thickness=0.2, track_width_between_wheels=0.15):
        super().__init__(position, orientation, mass, inertia, linear_velocity, angular_velocity, radius, thickness, track_width_between_wheels)


def _make_cylinder_mesh(posture, radius, height):
    theta = np.linspace(0, 2 * np.pi, 32)
    z_side = np.array([-height / 2, height / 2])
    TH, ZZ = np.meshgrid(theta, z_side)
    x_l = radius * np.cos(TH)
    y_l = radius * np.sin(TH)
    z_l = ZZ
    R_mat = posture[:3, :3]
    t_vec = posture[:3, 3]
    pts = np.stack([x_l.ravel(), y_l.ravel(), z_l.ravel()])
    pw = R_mat @ pts + t_vec[:, None]
    return pw[0].reshape(TH.shape), pw[1].reshape(TH.shape), pw[2].reshape(TH.shape)

def _make_disk_mesh(posture, radius, z_offset):
    theta = np.linspace(0, 2 * np.pi, 32)
    r_vals = np.linspace(0, radius, 12)
    TH, RR = np.meshgrid(theta, r_vals)
    x_l = RR * np.cos(TH)
    y_l = RR * np.sin(TH)
    z_l = np.full_like(x_l, z_offset)
    R_mat = posture[:3, :3]
    t_vec = posture[:3, 3]
    pts = np.stack([x_l.ravel(), y_l.ravel(), z_l.ravel()])
    pw = R_mat @ pts + t_vec[:, None]
    return pw[0].reshape(TH.shape), pw[1].reshape(TH.shape), pw[2].reshape(TH.shape)

def _make_plane_mesh(posture, width, length, resolution=10):
    x_vals = np.linspace(-width / 2, width / 2, resolution)
    y_vals = np.linspace(-length / 2, length / 2, resolution)
    XX, YY = np.meshgrid(x_vals, y_vals)
    ZZ = np.zeros_like(XX)
    local_points = np.stack([XX.ravel(),YY.ravel(),ZZ.ravel()
    ])
    R = posture[:3, :3]
    p = posture[:3, 3]
    world_points = R @ local_points + p[:, None]
    X = world_points[0].reshape(XX.shape)
    Y = world_points[1].reshape(YY.shape)
    Z = world_points[2].reshape(ZZ.shape)
    return X, Y, Z


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation

    radius = 0.1
    height = 0.6
    density = 1e3
    volume = np.pi * (radius ** 2) * height
    mass = volume * density
    Iz = (1/2) * mass * radius ** 2
    Ix = (1/12) * mass * (3 * radius ** 2 + height ** 2)
    Iy = Ix
    moment_of_inertia = np.array([Ix, Iy, Iz])

    initial_orientation = rot_z(90) @ rot_x(90)
    
    solid1 = RigidRobot3D(
        position=np.array([0.0, 0.0, 0.5]),
        orientation=initial_orientation,    
        mass=mass,
        inertia=moment_of_inertia,
        linear_velocity=np.zeros(3),
        angular_velocity=np.array([0,0,0]),
        radius=radius,
        thickness=height,
    )
    # Apply a boundary at the z = 0 plan 


    solid2 = RigidRobot3D(
        position=np.array([0.3, 0.0, 2.0]),
        orientation=np.eye(3),
        mass=mass,
        inertia=moment_of_inertia,
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        radius=radius,
        thickness=height,
    )

    objects = [solid1]
    COLORS  = ['steelblue']

    simulator = RigidObjectSimulator(
        time_step=0.001,
        duration=1,
        control_logic=None,
        stepper='position_verlet',
        show_progress=True,
    )
    for obj in objects:
        simulator.attach(obj)
    simulator.add_external_force(GravityForce())

    while simulator.run():
        simulator.step()
        simulator.record()

    n_obj = len(objects)
    times = np.array(simulator.time_collection)
    # positions[t, i, 3] — time × object × xyz
    positions = np.array([
        [simulator.posture_collection[t][i][:3, 3] for i in range(n_obj)]
        for t in range(len(times))
    ])
    all_postures = [
        [simulator.posture_collection[t][i] for i in range(n_obj)]
        for t in range(len(times))
    ]

    # ── End positions + penetration ───────────────────────────────────────────
    penetration      = np.array(simulator.penetration_collection)
    #end_positions    = np.array(simulator.end_position_collection)   # (T, 2)
    time_contact     = np.array(simulator.time_collection)
    F_wall_collection = np.array(simulator.F_wall_collection)



    fig, ax = plt.subplots(figsize=(10, 4))
    #ax.plot(time_contact, end_positions[:, 0], label='End A (z)', color='royalblue')
    #ax.plot(time_contact, end_positions[:, 1], label='End B (z)', color='darkorange')
    ax.plot(time_contact, penetration[::2],          label='Penetration', color='crimson', linestyle='--')
    ax.plot(time_contact,  F_wall_collection[::2,2], label = "resistance_force")
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position / Penetration (m)')
    ax.set_title('Cylinder End Positions and Ground Penetration vs Time')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    plan_posture = np.zeros((4,4))
    plan_posture[:3, :3] = rot_x(30)


    # ── Position vs time ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for k, (ax, label) in enumerate(zip(axes, ['x', 'y', 'z'])):
        for i, color in enumerate(COLORS):
            ax.plot(times, positions[:, i, k], color=color, label=f'Object {i+1}')
        ax.set_ylabel(f'{label} (m)')
        ax.grid(True)
    axes[0].legend()
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Cylinder Position vs Time')
    plt.tight_layout()
    plt.show()

    # ── 3-D animation ─────────────────────────────────────────────────────────
    stride = max(1, len(times) // 300)
    frame_indices = list(range(0, len(times), stride))

    # Equal-scale limits across all objects and all axes
    all_pos_flat = positions.reshape(-1, 3)
    center = all_pos_flat.mean(axis=0)
    half_range = max(
        (all_pos_flat.max(axis=0) - all_pos_flat.min(axis=0)).max() / 2,
        radius * 4,
    )
    x_lim = [center[0] - half_range, center[0] + half_range]
    y_lim = [center[1] - half_range, center[1] + half_range]
    z_lim = [center[2] - half_range, center[2] + half_range]

    fig_anim = plt.figure(figsize=(8, 8))
    ax3d = fig_anim.add_subplot(111, projection='3d')

    def animate(fi):
        ax3d.cla()
        t_idx = frame_indices[fi]

        for i, color in enumerate(COLORS):
            posture = all_postures[t_idx][i]
            obj_r = objects[i].radius
            obj_h = objects[i].thickness


            Xp, Yp, Zp = _make_plane_mesh(plan_posture, 3.0, 3.0)
            ax3d.plot_surface(Xp, Yp, Zp, color="grey", alpha=0.2, linewidth=0)

            X, Y, Z = _make_cylinder_mesh(posture, obj_r, obj_h)
            ax3d.plot_surface(X, Y, Z, color=color, alpha=1.0, linewidth=0)
            for z_off in (-obj_h / 2, obj_h / 2):
                Xc, Yc, Zc = _make_disk_mesh(posture, obj_r, z_off)
                ax3d.plot_surface(Xc, Yc, Zc, color=color, alpha=1.0, linewidth=0)


            ax3d.plot(positions[:t_idx + 1, i, 0], positions[:t_idx + 1, i, 1],

                      positions[:t_idx + 1, i, 2], color=color, alpha=0.4, lw=1)
            
        ax3d.set_xlim(x_lim); ax3d.set_ylim(y_lim); ax3d.set_zlim(z_lim)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.view_init(elev=30, azim=0)
        ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
        ax3d.set_title(f't = {times[t_idx]:.3f} s')

    anim = FuncAnimation(fig_anim, animate, frames=len(frame_indices), interval=20, repeat=True)
    plt.show()









