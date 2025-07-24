import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import PillowWriter

matplotlib.use('MacOSX')  # This should enable animations in PyCharm

# -- set font to Roboto
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Helvetica', 'Arial', 'DejaVu Sans']

# --- parameters
x_range = np.linspace(-5, 5, 40)
y_range = np.linspace(-5, 5, 40)
x, y = np.meshgrid(x_range, y_range)

def create_peaks_terrain(x, y):
    """create base terrain"""
    z = 3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
        - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
        - 1/3 * np.exp(-(x + 1)**2 - y**2)
    return z

def add_gaussian_bowl(z, x, y, center_x, center_y, depth, width):
    """add depressions and valleys to landscape"""
    bowl = depth * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))
    return z + bowl

def create_healthy_landscape(x, y):
    """healthy brain --> balanced landscape"""
    z = create_peaks_terrain(x, y)
    z = add_gaussian_bowl(z, x, y, 0, 0, -6, 4)
    z = add_gaussian_bowl(z, x, y, 0, 1.5, -4, 6)
    z = add_gaussian_bowl(z, x, y, -0.6, -0.8, -3, 2.5)
    return zscore(z)

def create_anxiety_landscape(x, y):
    """anxiety / ocd --> deep & narrow valleys/basins"""
    z = create_peaks_terrain(x, y)
    z = add_gaussian_bowl(z, x, y, 0, 0, -15, 1.8)
    z = add_gaussian_bowl(z, x, y, -1.2, 1.2, -12, 1.2)
    z = add_gaussian_bowl(z, x, y, 1.8, -0.8, -10, 1.0)
    z = add_gaussian_bowl(z, x, y, -0.3, -2.2, -8, 0.8)
    return zscore(z) * 2.5

def create_adhd_landscape(x, y):
    """adhd/add --> rugged, shallow, multiple narrow attractors basins and peaks"""
    z = create_peaks_terrain(x, y) * 0.2
    # --- existing broad attractors
    z = add_gaussian_bowl(z, x, y, 0, 0, -1.5, 2.5)
    z = add_gaussian_bowl(z, x, y, 2.5, 1.2, -1.2, 1.2)
    z = add_gaussian_bowl(z, x, y, -2.2, 1.8, -1.2, 1.0)
    z = add_gaussian_bowl(z, x, y, 1.5, -2.3, -1.0, 1.0)
    z = add_gaussian_bowl(z, x, y, -2.8, -1.2, -1.0, 1.0)
    z = add_gaussian_bowl(z, x, y, 0.8, 3.2, -0.8, 1.0)
    # --- new narrow valleys negative depth, narrow width
    z = add_gaussian_bowl(z, x, y, 1.8, 2.5, -0.9, 0.7)
    z = add_gaussian_bowl(z, x, y, -1.5, 2.7, -1.1, 0.8)
    z = add_gaussian_bowl(z, x, y, -3.5, -2.5, -0.8, 0.6)
    z = add_gaussian_bowl(z, x, y, 3.0, -1.0, -1.0, 0.9)
    z = add_gaussian_bowl(z, x, y, 2.0, -3.5, -0.7, 0.7)
    # --- new narrow peaks positive depth, narrow width
    z = add_gaussian_bowl(z, x, y, -2.0, 0.0, 1.0, 0.8)
    z = add_gaussian_bowl(z, x, y, 0.0, -3.0, 0.8, 0.7)
    z = add_gaussian_bowl(z, x, y, 3.5, 3.5, 0.9, 0.9)
    z = add_gaussian_bowl(z, x, y, -3.5, 3.0, 1.1, 1.0)
    z = add_gaussian_bowl(z, x, y, 2.5, -2.5, 0.7, 0.7)
    return zscore(z) * 0.18  # --- keep it shallow but rugged

# --- make the landscapes
z_healthy = create_healthy_landscape(x, y)
z_anxiety = create_anxiety_landscape(x, y)
z_adhd = create_adhd_landscape(x, y)

# --- apply circular mask (recommended)
radius = 4
mask = (x**2 + y**2) <= radius**2
z_healthy[~mask] = np.nan
z_anxiety[~mask] = np.nan
z_adhd[~mask] = np.nan

# --- calc gradients
dx = x[0, 1] - x[0, 0]
dy = y[1, 0] - y[0, 0]
dzdx_healthy, dzdy_healthy = np.gradient(z_healthy, dx, dy)
dzdx_anxiety, dzdy_anxiety = np.gradient(z_anxiety, dx, dy)
dzdx_adhd, dzdy_adhd = np.gradient(z_adhd, dx, dy)

# --- setup interpolators
interp_dzdx_healthy = RegularGridInterpolator((x_range, y_range), dzdx_healthy.T, method='linear', bounds_error=False, fill_value=0)
interp_dzdy_healthy = RegularGridInterpolator((x_range, y_range), dzdy_healthy.T, method='linear', bounds_error=False, fill_value=0)
interp_dzdx_anxiety = RegularGridInterpolator((x_range, y_range), dzdx_anxiety.T, method='linear', bounds_error=False, fill_value=0)
interp_dzdy_anxiety = RegularGridInterpolator((x_range, y_range), dzdy_anxiety.T, method='linear', bounds_error=False, fill_value=0)
interp_dzdx_adhd = RegularGridInterpolator((x_range, y_range), dzdx_adhd.T, method='linear', bounds_error=False, fill_value=0)
interp_dzdy_adhd = RegularGridInterpolator((x_range, y_range), dzdy_adhd.T, method='linear', bounds_error=False, fill_value=0)

interp_healthy = RegularGridInterpolator((x_range, y_range), z_healthy.T, method='linear', bounds_error=False, fill_value=np.nan)
interp_anxiety = RegularGridInterpolator((x_range, y_range), z_anxiety.T, method='linear', bounds_error=False, fill_value=np.nan)
interp_adhd = RegularGridInterpolator((x_range, y_range), z_adhd.T, method='linear', bounds_error=False, fill_value=np.nan)

# --- anim setup
num_frames = 300
jitter_amplitude = 0.2
jitter_x = jitter_amplitude * (np.random.rand(num_frames) - 0.5)
jitter_y = jitter_amplitude * (np.random.rand(num_frames) - 0.5)

# --- sim marble starting positions
marble_pos = {
    'healthy': {'x': 0, 'y': -1},
    'anxiety': {'x': 0, 'y': -1},
    'adhd': {'x': 0, 'y': -1}
}

# --- save trajectories
trajectories = {
    'healthy': {'x': [], 'y': []},
    'anxiety': {'x': [], 'y': []},
    'adhd': {'x': [], 'y': []}
}

def update_marble(pos, interp_dzdx, interp_dzdy, jx, jy, step_size=0.05):
    """update marble position"""
    slope_x = interp_dzdx([pos['x'], pos['y']])[0]
    slope_y = interp_dzdy([pos['x'], pos['y']])[0]
    
    new_x = pos['x'] - step_size * slope_x + jx
    new_y = pos['y'] - step_size * slope_y + jy
    
    # --- keep within circular boundary
    if (new_x**2 + new_y**2) < (radius - 0.5)**2:
        pos['x'] = new_x
        pos['y'] = new_y
    else:
        angle = np.arctan2(new_y, new_x)
        pos['x'] = (radius - 0.55) * np.cos(angle)
        pos['y'] = (radius - 0.55) * np.sin(angle)
    
    return pos

# --- make figure
fig = plt.figure(figsize=(15, 10), facecolor='white')
plt.subplots_adjust(hspace=0.3)

def animate(frame):
    """animation function"""
    fig.clear()
    
    # -- adding drift for healthy brain
    t = frame / num_frames * 2 * np.pi  # normalized time for oscillation
    drift_x = 0.08 * np.cos(t * 0.5)  # slow circular drift
    drift_y = 0.08 * np.sin(t * 0.5)
    
    # --- continuously update marble positions
    marble_pos['healthy'] = update_marble(marble_pos['healthy'], interp_dzdx_healthy, interp_dzdy_healthy, 
                                         jitter_x[frame]*3 + drift_x, jitter_y[frame]*3 + drift_y, 0.3)
    marble_pos['anxiety'] = update_marble(marble_pos['anxiety'], interp_dzdx_anxiety, interp_dzdy_anxiety, 
                                         jitter_x[frame]*0.3, jitter_y[frame]*0.3, 0.02)
    marble_pos['adhd'] = update_marble(marble_pos['adhd'], interp_dzdx_adhd, interp_dzdy_adhd, 
                                      jitter_x[frame]*10, jitter_y[frame]*10, 0.9)
    
    # --- marble trajectories
    for condition in ['healthy', 'anxiety', 'adhd']:
        trajectories[condition]['x'].append(marble_pos[condition]['x'])
        trajectories[condition]['y'].append(marble_pos[condition]['y'])
        # Keep only last 100 points of the trajectory "tail effect"
        if len(trajectories[condition]['x']) > 1000:
            trajectories[condition]['x'].pop(0)
            trajectories[condition]['y'].pop(0)
    
    # Data for plotting
    landscapes = [(z_healthy, 'healthy brain', 'black'), 
                  (z_anxiety, 'anxiety/ocd', 'black'), 
                  (z_adhd, 'adhd', 'black')]
    conditions = ['healthy', 'anxiety', 'adhd']
    interps = [interp_healthy, interp_anxiety, interp_adhd]
    
    # --- top row 3d landscapes with marbles
    for i, ((z, title, color), condition, interp_func) in enumerate(zip(landscapes, conditions, interps)):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # --- landscape
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7, edgecolor='none')
        
        # --- marble
        pos = marble_pos[condition]
        marble_z = interp_func([pos['x'], pos['y']])[0]
        # --- use a smaller offset for adhd
        z_offset = 0.2 if condition != 'adhd' else 0.05
        ax.scatter([pos['x']], [pos['y']], [marble_z + z_offset], 
                  color='#FF0000', s=150, edgecolor='white', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        ax.view_init(elev=20, azim=-30)
        ax.axis('off')
    
    # --- bottom 2d trajectories
    for i, ((z, title, color), condition) in enumerate(zip(landscapes, conditions)):
        ax = fig.add_subplot(2, 3, i+4)
        
        # --- contour
        ax.contourf(x, y, z, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(x, y, z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # --- trajectory
        traj_x = trajectories[condition]['x']
        traj_y = trajectories[condition]['y']
        
        if len(traj_x) > 1:
            # --- plot trajectory with tail fading effect (reduce clutter)
            for j in range(1, len(traj_x)):
                alpha = j / len(traj_x)
                ax.plot([traj_x[j-1], traj_x[j]], [traj_y[j-1], traj_y[j]], 
                       color='white', alpha=alpha, linewidth=2)
        
        # --- current marble position
        pos = marble_pos[condition]
        ax.scatter([pos['x']], [pos['y']], color='#FF0000', s=150, 
                  edgecolor='white', linewidth=2, zorder=10)
        
        # --- boundary
        circle = plt.Circle((0, 0), radius, fill=False, color='white', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_title(f'{title} - trajectory', fontsize=12, color=color)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()

# --- run animation
print("starting brain state energy landscape simulation...")
print("close the window to stop the animation...")
anim = FuncAnimation(fig, animate, frames=num_frames, interval=25, repeat=False) # --- animation loops itself inifinitely when set to "True"
plt.show()


# ---- save animation
# anim.save('brain_statespace.mp4', writer='ffmpeg', fps=30)

# --- save as .gif with specific settings
anim.save('cognitive_energy_landscape.gif', writer='pillow', fps=30, bitrate=1800)
print("animation saved...")

# --- save with progress bar
# writer = PillowWriter(fps=30)
# anim.save('cognitive_energy_landscape.gif', writer=writer)


print("""
INTERPRETATION:
--> healthy brain: balanced movement, moderate valleys
--> anxiety: gets stuck in deep valleys (rumination)  
--> ADHD: jumpy movement between shallow attractors (distractibility) very rugged terrain
""")