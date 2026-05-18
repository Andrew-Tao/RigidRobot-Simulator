import numpy as np 
from methods2D import SE2LieAlgebra
import matplotlib.pyplot as plt
lie2 = SE2LieAlgebra()
from RigidRobot2D import RigidRobot2D
from Simulator2D import Simulator2D
lie2 = SE2LieAlgebra()

def boundary(position: np.ndarray) -> bool:
    x, y = position
    in_bottom = (0 <= x <= 3)   and (0 <= y <= 1)
    in_middle = (0.6 <= x <= 2.4) and (1 <= y <= 3)
    in_top    = (0 <= x <= 3)   and (3 <= y <= 4)
    return bool(in_bottom or in_middle or in_top)

def control_logic(time):
    input_force_left = 0.15 if time < 0.2 else 0.0
    input_force_right = 0.14 if time < 0.2 else 0.0
    return np.array([input_force_left, input_force_right])
    
if __name__ == "__main__":
    env = boundary

    robot = RigidRobot2D(
        position=np.array([1.5, 1.0]), 
        orientation=np.pi / 2, 
        mass=2.0, 
        inertia=0.05, 
        linear_velocity=np.zeros(2), 
        angular_velocity=0.0,
    )
    robot.friction_coefficient = 0.0
    
    robot.contact(env)
    #robot.visualize()
    
    simulator = Simulator2D(
        time_step=2, 
        duration = 10.0, 
        control_logic=control_logic, 
        stepper='position_verlet')
    
    simulator.attach(robot)


    # Run the simulation for 10 seconds
    while simulator.run():
        robot.control = np.random.rand(2)  # Random control input for demonstration
        simulator.step()
        simulator.record()

    #---------------Data Collection from Simulator----------------
    velocity = np.array( [simulator.velocity_matrix_collection[i][0,1] for i in range(len(simulator.velocity_matrix_collection)) ])
    velocity_matrix = np.array( [simulator.velocity_matrix_collection[i] for i in range(len(simulator.velocity_matrix_collection)) ])
    position = np.array( [simulator.posture_collection[i][:2,2] for i in range(len(simulator.posture_collection)) ] )
    momentum = np.array( [simulator.momentum_collection[i] for i in range(len(simulator.momentum_collection)) ] )

    print(momentum.shape)
    x_pos = position[:, 0]
    y_pos = position[:, 1]
    force = np.array( [simulator.force_collection[i][0] for i in range(len(simulator.force_collection)) ])
    time = np.array(simulator.time_collection)

    energy  = np.array([0.5 * np.dot(momentum[i], lie2.vee(velocity_matrix[i])) for i in range(len(simulator.posture_collection))])
    """
    
    
    #plt.plot(time, velocity, label='Linear Velocity (v_x)')
    plt.plot(time[5:], energy[5:], label='Kinetic Energy')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Kinetic Energy over Time')
    plt.grid()
    plt.show()

   
    plt.scatter(x_pos, y_pos, c=force, cmap='viridis', label='Position (color by force)', edgecolors='k')
    
    plt.xlabel(' x')
    plt.ylabel(' y')
    plt.title(' Position over Time')
    plt.grid()
    plt.show()
    """
    # ── Trajectory plot ───────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import LineCollection

    xs     = np.array([T[0, 2] for T in simulator.posture_collection])
    ys     = np.array([T[1, 2] for T in simulator.posture_collection])
    speeds = np.array([np.hypot(V[0, 2], V[1, 2])
                       for V in simulator.velocity_matrix_collection])

    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # I-shape environment
    i_verts = np.array([
        [0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.4, 1.0],
        [2.4, 3.0], [3.0, 3.0], [3.0, 4.0], [0.0, 4.0],
        [0.0, 3.0], [0.6, 3.0], [0.6, 1.0], [0.0, 1.0],
    ])
    ax.add_patch(MplPolygon(i_verts, closed=True,
                            facecolor='#1e2a4a', edgecolor='#56b4f7',
                            linewidth=2.0, zorder=1))

    # Colour-mapped path (speed → plasma)
    pts  = np.stack([xs, ys], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap='plasma',
                          norm=plt.Normalize(speeds.min(), speeds.max()),
                          linewidth=2.5, zorder=3)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label('speed (m/s)', color='#888899')
    cb.ax.yaxis.set_tick_params(color='#888899')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888899')

    # Ghost robot poses (heading arrows) at ~8 evenly-spaced frames
    n_ghosts = 8
    r = robot.radius
    for idx in np.linspace(0, len(simulator.posture_collection) - 1,
                           n_ghosts, dtype=int):
        T = simulator.posture_collection[idx]
        pos     = T[:2, 2]
        heading = T[:2, 0]
        lateral = T[:2, 1]
        ax.add_patch(mpatches.Circle(pos, r, facecolor='#c0392b',
                                     edgecolor='white', linewidth=1.2,
                                     alpha=0.4, zorder=5))
        ax.annotate('', xy=pos + heading * r * 1.6, xytext=pos,
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b',
                                   lw=1.5, mutation_scale=10),
                    alpha=0.6, zorder=6)

    # Start / end markers
    ax.scatter([xs[0]],  [ys[0]],  s=90, color='#90be6d', zorder=8,
               label='start', marker='o')
    ax.scatter([xs[-1]], [ys[-1]], s=90, color='#e76f51', zorder=8,
               label='end',   marker='*')

    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.2, 4.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', color='#888899')
    ax.set_ylabel('y (m)', color='#888899')
    ax.set_title('Robot Trajectory', color='white', fontweight='bold', pad=10)
    ax.tick_params(colors='#888899')
    for spine in ax.spines.values():
        spine.set_color('#333344')
    ax.grid(True, color='#1e2a3a', linewidth=0.6, linestyle='--')
    ax.legend(facecolor='#1e2a4a', edgecolor='#56b4f7',
              labelcolor='white', fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.show()
 