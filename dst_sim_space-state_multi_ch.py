import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # alternative use 'Qt5Agg'

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")


class BrainStateSpaceAnimator:
    def __init__(self, duration=30, sampling_rate=10):
        self.duration = duration  # seconds
        self.fs = sampling_rate  # Hz
        self.time = np.linspace(0, duration, duration * sampling_rate)
        self.n_timepoints = len(self.time)

        # Generate simulated fNIRS data
        self.generate_fnirs_data()

        # Perform dimensionality reduction (simulate state space)
        self.create_state_space()

        # Setup animation
        self.setup_animation()

    def generate_fnirs_data(self):
        """Generate realistic fNIRS hemodynamic signals"""
        # Simulate multiple brain regions
        n_channels = 20

        # Create transition from rest to focus at t=10s
        transition_time = 10
        transition_idx = int(transition_time * self.fs)

        # Hemodynamic response function (HRF)
        def hrf(t, delay=2, dispersion=1):
            """Canonical hemodynamic response function"""
            return (t / delay) ** 2 * np.exp(-(t / delay)) / (2 * delay * dispersion ** 2)

        # Generate baseline activity (rest state)
        baseline_activity = np.random.randn(n_channels, self.n_timepoints) * 0.1

        # Add low-frequency oscillations (0.1 Hz) for resting state networks
        rest_oscillation = np.sin(2 * np.pi * 0.1 * self.time) * 0.3

        # Create focus-related activation
        focus_regions = [2, 5, 8, 12, 15]  # Simulated attention network regions

        self.fnirs_data = baseline_activity.copy()

        for i, channel in enumerate(self.fnirs_data):
            # Add resting state oscillations to all channels
            channel += rest_oscillation * (0.5 + np.random.rand() * 0.5)

            # Add focus-related activation to specific regions
            if i in focus_regions:
                # Create step function for attention onset
                attention_signal = np.zeros_like(self.time)
                attention_signal[transition_idx:] = 1.0

                # Convolve with HRF for realistic hemodynamic response
                hrf_kernel = hrf(np.linspace(0, 10, 50))
                convolved = np.convolve(attention_signal, hrf_kernel, mode='same')

                # Scale and add noise
                channel += convolved * (1.5 + np.random.rand() * 0.5)

        # Apply bandpass filter (typical fNIRS preprocessing)
        self.filter_data()

    def filter_data(self):
        """Apply bandpass filter to fNIRS data"""
        nyquist = self.fs / 2
        low, high = 0.01, 0.2  # Typical fNIRS frequency range
        b, a = butter(4, [low / nyquist, high / nyquist], btype='band')

        for i in range(self.fnirs_data.shape[0]):
            self.fnirs_data[i] = filtfilt(b, a, self.fnirs_data[i])

    def create_state_space(self):
        """Create 3D state space representation using PCA"""
        from sklearn.decomposition import PCA

        # Transpose for PCA (timepoints x channels)
        data_matrix = self.fnirs_data.T

        # Apply PCA to reduce to 3D state space
        pca = PCA(n_components=3)
        self.state_space = pca.fit_transform(data_matrix)

        # Store explained variance for visualization
        self.explained_variance = pca.explained_variance_ratio_

        # Define rest and focus periods for coloring
        self.rest_indices = self.time < 10
        self.focus_indices = self.time >= 10

    def setup_animation(self):
        """Setup the 3D animation"""
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
        self.ax_3d.set_xlabel(f'PC1 ({self.explained_variance[0]:.1%} var)', fontsize=10)
        self.ax_3d.set_ylabel(f'PC2 ({self.explained_variance[1]:.1%} var)', fontsize=10)
        self.ax_3d.set_zlabel(f'PC3 ({self.explained_variance[2]:.1%} var)', fontsize=10)
        self.ax_3d.set_title('Brain State-Space Trajectory\n(fNIRS Dynamic System)', fontsize=12, pad=20)

        # Set axis limits
        margin = 0.1
        for ax_method, data_col in zip([self.ax_3d.set_xlim, self.ax_3d.set_ylim, self.ax_3d.set_zlim],
                                       [0, 1, 2]):
            data_range = self.state_space[:, data_col]
            ax_method([data_range.min() - margin, data_range.max() + margin])

        # Setup time series plots
        self.setup_time_series_plots()

        # Add text annotations
        self.time_text = self.ax_3d.text2D(0.02, 0.98, '', transform=self.ax_3d.transAxes,
                                           fontsize=12, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        self.state_text = self.ax_3d.text2D(0.02, 0.88, '', transform=self.ax_3d.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        plt.tight_layout()

    def setup_time_series_plots(self):
        """Setup auxiliary time series plots"""
        # Raw fNIRS signals from key regions
        self.ax_ts1.set_title('fNIRS Signals (Key Regions)', fontsize=10)
        self.ax_ts1.set_xlabel('Time (s)')
        self.ax_ts1.set_ylabel('ΔHbO₂ (μM)')

        # State space components over time
        self.ax_ts2.set_title('State Space Components', fontsize=10)
        self.ax_ts2.set_xlabel('Time (s)')
        self.ax_ts2.set_ylabel('Component Value')

        # Phase portrait (PC1 vs PC2)
        self.ax_phase.set_title('Phase Portrait (PC1 vs PC2)', fontsize=10)
        self.ax_phase.set_xlabel('PC1')
        self.ax_phase.set_ylabel('PC2')

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
            self.update_time_series(frame)

            # Update text
            brain_state = "Resting State" if current_time < 10 else "Focused Attention"
            self.time_text.set_text(f'Time: {current_time:.1f}s')
            self.state_text.set_text(f'State: {brain_state}')

            # Rotate view slowly
            self.ax_3d.view_init(elev=20, azim=frame * 0.5)

        return [self.trajectory_line, self.current_point, self.rest_points, self.focus_points]

    def update_time_series(self, frame):
        """Update time series subplots"""
        current_frame = min(frame, len(self.time) - 1)
        time_slice = self.time[:current_frame + 1]

        # Clear and update raw signals
        self.ax_ts1.clear()
        self.ax_ts1.set_title('fNIRS Signals (Key Regions)', fontsize=10)
        self.ax_ts1.set_xlabel('Time (s)')
        self.ax_ts1.set_ylabel('ΔHbO₂ (μM)')

        # Plot a few representative channels
        key_channels = [2, 5, 8, 12, 15]
        colors = plt.cm.Set3(np.linspace(0, 1, len(key_channels)))

        for i, (ch, color) in enumerate(zip(key_channels, colors)):
            self.ax_ts1.plot(time_slice, self.fnirs_data[ch, :current_frame + 1],
                             color=color, linewidth=1.5, alpha=0.8, label=f'Ch {ch + 1}')

        self.ax_ts1.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Focus Onset')
        self.ax_ts1.legend(fontsize=8)
        self.ax_ts1.grid(True, alpha=0.3)

        # Clear and update state space components
        self.ax_ts2.clear()
        self.ax_ts2.set_title('State Space Components', fontsize=10)
        self.ax_ts2.set_xlabel('Time (s)')
        self.ax_ts2.set_ylabel('Component Value')

        component_colors = ['cyan', 'magenta', 'yellow']
        for i in range(3):
            self.ax_ts2.plot(time_slice, self.state_space[:current_frame + 1, i],
                             color=component_colors[i], linewidth=2, alpha=0.8,
                             label=f'PC{i + 1}')

        self.ax_ts2.axvline(x=10, color='red', linestyle='--', alpha=0.7)
        self.ax_ts2.legend(fontsize=8)
        self.ax_ts2.grid(True, alpha=0.3)

        # update phase portrait
        self.ax_phase.clear()
        self.ax_phase.set_title('Phase Portrait (PC1 vs PC2)', fontsize=10)
        self.ax_phase.set_xlabel('PC1')
        self.ax_phase.set_ylabel('PC2')

        # plot trajectory in phase space
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

            # current point
            current_pos = self.state_space[current_frame, :2]
            self.ax_phase.scatter(current_pos[0], current_pos[1], c='red', s=100, alpha=1.0,
                                  edgecolors='white', linewidths=2, label='Current')

            # trajectory line
            self.ax_phase.plot(phase_data[:, 0], phase_data[:, 1], 'cyan', alpha=0.5, linewidth=1)

        self.ax_phase.legend(fontsize=8)
        self.ax_phase.grid(True, alpha=0.3)

    def run_animation(self, interval=100, save_as=None):
        """run animation"""
        frames = len(self.time)
        anim = animation.FuncAnimation(self.fig, self.animate, frames=frames,
                                       interval=interval, blit=False, repeat=True)

        if save_as:
            print(f"Saving animation as {save_as}...")
            anim.save(save_as, writer='pillow', fps=10, dpi=100)
            print("Animation saved!")

        plt.show()
        return anim


# create and run the animation
if __name__ == "__main__":
    print("Initializing Brain State-Space Animation...")
    print("Simulating fNIRS data with transition from rest to focused attention...")

    # create the animator
    animator = BrainStateSpaceAnimator(duration=30, sampling_rate=10)

    print("Starting animation...")
    print("- Blue points: Resting state")
    print("- Orange points: Focused attention state")
    print("- Red point: Current brain state")
    print("- Transition occurs at t=10s")

    # run the animation
    # uncomment to save as .gif
    # anim = animator.run_animation(interval=100, save_as='brain_statespace.gif')
    anim = animator.run_animation(interval=100)