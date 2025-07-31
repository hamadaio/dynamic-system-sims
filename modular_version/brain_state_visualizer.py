"""
Animation Visualizer Module

This module handles all visualization and animation functionality
for the brain state space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


class BrainStateVisualizer:
    """Handles 3D animation and visualization of brain state space"""
    
    def __init__(self, fnirs_data, state_space, time_vector, channel_names, explained_variance):
        self.fnirs_data = fnirs_data
        self.state_space = state_space
        self.time = time_vector
        self.channel_names = channel_names
        self.explained_variance = explained_variance
        
        # Animation components
        self.fig = None
        self.ax_3d = None
        self.ax_ts1 = None
        self.ax_ts2 = None
        self.ax_phase = None
        self.anim = None
        
        # Plot elements
        self.trajectory_line = None
        self.current_point = None
        self.rest_points = None
        self.focus_points = None
        self.time_text = None
        self.state_text = None
        
    def setup_animation(self):
        """Setup the 3D animation and subplots"""
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        self.fig = plt.figure(figsize=(15, 10))

        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_facecolor('black')

        # Time series plots
        self.ax_ts1 = self.fig.add_subplot(222)
        self.ax_ts2 = self.fig.add_subplot(223)
        self.ax_phase = self.fig.add_subplot(224)

        # Initialize empty plots
        self.trajectory_line, = self.ax_3d.plot([], [], [], 'cyan', alpha=0.6, linewidth=2)
        self.current_point = self.ax_3d.scatter([], [], [], c='red', s=100, alpha=0.8)
        self.rest_points = self.ax_3d.scatter([], [], [], c='blue', s=20, alpha=0.3)
        self.focus_points = self.ax_3d.scatter([], [], [], c='orange', s=20, alpha=0.3)

        # Set labels and title
        self.ax_3d.set_xlabel(f'pc1 ({self.explained_variance[0]:.1%} var)', fontsize=10)
        self.ax_3d.set_ylabel(f'pc2 ({self.explained_variance[1]:.1%} var)', fontsize=10)
        self.ax_3d.set_zlabel(f'pc3 ({self.explained_variance[2]:.1%} var)', fontsize=10)
        self.ax_3d.set_title('rest --> attention space-state trajectory\n(fnirs dst)', fontsize=12, pad=20)

        # Set axis limits
        self._set_axis_limits()

        # Setup time series plots
        self._setup_time_series_plots()

        # Add text annotations
        self.time_text = self.ax_3d.text2D(0.02, 0.98, '', transform=self.ax_3d.transAxes,
                                           fontsize=12, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        self.state_text = self.ax_3d.text2D(0.02, 0.88, '', transform=self.ax_3d.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        plt.tight_layout()
        
    def _set_axis_limits(self):
        """Set appropriate axis limits for 3D plot"""
        margin = 0.1
        for ax_method, data_col in zip([self.ax_3d.set_xlim, self.ax_3d.set_ylim, self.ax_3d.set_zlim],
                                       [0, 1, 2]):
            data_range = self.state_space[:, data_col]
            ax_method([data_range.min() - margin, data_range.max() + margin])

    def _setup_time_series_plots(self):
        """Setup auxiliary time series plots"""
        # Raw fNIRS signals from key regions
        self.ax_ts1.set_title('fNIRS sig (key regions)', fontsize=10)
        self.ax_ts1.set_xlabel('time (s)')
        self.ax_ts1.set_ylabel('ΔHbO₂ (μM)')

        # State space components over time
        self.ax_ts2.set_title('state space components', fontsize=10)
        self.ax_ts2.set_xlabel('time (s)')
        self.ax_ts2.set_ylabel('comp value')

        # Phase portrait (PC1 vs PC2)
        self.ax_phase.set_title('phase portrait (pc1 vs pc2)', fontsize=10)
        self.ax_phase.set_xlabel('pc1')
        self.ax_phase.set_ylabel('pc2')

    def animate(self, frame):
        """Animation function"""
        if frame < len(self.time):
            current_time = self.time[frame]

            # Update 3D trajectory
            trajectory_data = self.state_space[:frame + 1]
            if len(trajectory_data) > 1:
                self.trajectory_line.set_data_3d(trajectory_data[:, 0],
                                                 trajectory_data[:, 1],
                                                 trajectory_data[:, 2])

            # Update current point
            current_pos = self.state_space[frame]
            self.current_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])

            # Update rest/focus point clouds
            rest_mask = self.time[:frame + 1] < 10
            focus_mask = self.time[:frame + 1] >= 10

            if np.any(rest_mask):
                rest_data = self.state_space[:frame + 1][rest_mask]
                self.rest_points._offsets3d = (rest_data[:, 0], rest_data[:, 1], rest_data[:, 2])

            if np.any(focus_mask):
                focus_data = self.state_space[:frame + 1][focus_mask]
                self.focus_points._offsets3d = (focus_data[:, 0], focus_data[:, 1], focus_data[:, 2])

            # Update time series plots
            self._update_time_series(frame)

            # Update text
            brain_state = "resting / baseline" if current_time < 10 else "focused attention"
            self.time_text.set_text(f'Time: {current_time:.1f}s')
            self.state_text.set_text(f'cognitive state: {brain_state}')

            # Rotate view slowly
            self.ax_3d.view_init(elev=20, azim=frame * 0.5)

        return [self.trajectory_line, self.current_point, self.rest_points, self.focus_points]

    def _update_time_series(self, frame):
        """Update time series subplots"""
        current_frame = min(frame, len(self.time) - 1)
        time_slice = self.time[:current_frame + 1]

        # Clear and update raw signals
        self.ax_ts1.clear()
        self.ax_ts1.set_title('fNIRS sig (key regions)', fontsize=10)
        self.ax_ts1.set_xlabel('time (s)')
        self.ax_ts1.set_ylabel('ΔHbO₂ (μM)')

        # Plot all three channels
        colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

        for i, (color, name) in enumerate(zip(colors, self.channel_names)):
            self.ax_ts1.plot(time_slice, self.fnirs_data[i, :current_frame + 1],
                             color=color, linewidth=2, alpha=0.8, label=name)

        self.ax_ts1.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='focus/attention onset')
        self.ax_ts1.legend(fontsize=8)
        self.ax_ts1.grid(True, alpha=0.3)

        # Clear and update state space components
        self.ax_ts2.clear()
        self.ax_ts2.set_title('state space comps', fontsize=10)
        self.ax_ts2.set_xlabel('time (s)')
        self.ax_ts2.set_ylabel('component val')

        component_colors = ['cyan', 'magenta', 'yellow']
        for i in range(3):
            self.ax_ts2.plot(time_slice, self.state_space[:current_frame + 1, i],
                             color=component_colors[i], linewidth=2, alpha=0.8,
                             label=f'PC{i + 1}')

        self.ax_ts2.axvline(x=10, color='red', linestyle='--', alpha=0.7)
        self.ax_ts2.legend(fontsize=8)
        self.ax_ts2.grid(True, alpha=0.3)

        # Update phase portrait
        self._update_phase_portrait(current_frame)

    def _update_phase_portrait(self, current_frame):
        """Update phase portrait subplot"""
        self.ax_phase.clear()
        self.ax_phase.set_title('phase port (pc1 vs pc2)', fontsize=10)
        self.ax_phase.set_xlabel('pc1')
        self.ax_phase.set_ylabel('pc2')

        # Plot trajectory in phase space
        if current_frame > 0:
            phase_data = self.state_space[:current_frame + 1, :2]
            rest_mask = self.time[:current_frame + 1] < 10
            focus_mask = self.time[:current_frame + 1] >= 10

            if np.any(rest_mask):
                self.ax_phase.scatter(phase_data[rest_mask, 0], phase_data[rest_mask, 1],
                                      c='blue', s=20, alpha=0.6, label='Rest')
            if np.any(focus_mask):
                self.ax_phase.scatter(phase_data[focus_mask, 0], phase_data[focus_mask, 1],
                                      c='orange', s=20, alpha=0.6, label='Focus')

            # Current point
            current_pos = self.state_space[current_frame, :2]
            self.ax_phase.scatter(current_pos[0], current_pos[1], c='red', s=100, alpha=1.0,
                                  edgecolors='white', linewidths=2, label='Current')

            # Trajectory line
            self.ax_phase.plot(phase_data[:, 0], phase_data[:, 1], 'cyan', alpha=0.5, linewidth=1)

        self.ax_phase.legend(fontsize=8)
        self.ax_phase.grid(True, alpha=0.3)

    def run_animation(self, interval=100, save_as=None):
        """Run the animation"""
        frames = len(self.time)
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=frames,
                                            interval=interval, blit=False, repeat=True)

        if save_as:
            print(f"Saving animation as {save_as}...")
            self.anim.save(save_as, writer='pillow', fps=10, dpi=100)
            print("Animation saved!")

        plt.show(block=True)
        return self.anim
