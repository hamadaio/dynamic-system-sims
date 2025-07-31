import matplotlib
matplotlib.use('TkAgg')  # --- this should enable animations in PyCharm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
import seaborn as sns

# --- enable interactive mode
plt.ion()

# --- set canvas style/color
plt.style.use('dark_background')
sns.set_palette("husl")


class BrainStateSpaceAnimator:
    def __init__(self, duration=30, sampling_rate=10):
        self.duration = duration  # seconds
        self.fs = sampling_rate  # Hz
        self.time = np.linspace(0, duration, duration * sampling_rate)
        self.n_timepoints = len(self.time)

        # --- generate simulated fNIRS data
        self.generate_fnirs_data()

        # --- perform dimensionality reduction --> simulate state space
        self.create_state_space()

        # --- create energy landscape
        self.create_energy_landscape()

        # --- setup animation
        self.setup_animation()

    def generate_fnirs_data(self):
        """Generate realistic fNIRS signals for prefrontal cortex channels"""
        # Three channels: Fp1 (left PFC), Fp2 (right PFC), short separation (physiological noise)
        self.channel_names = ['fp1 (left-pfc)', 'fp2 (right-pfc)', 'short-sep ch (sys noise)']
        n_channels = 3

        # --- create transition from rest to focus at t=10s
        transition_time = 10
        transition_idx = int(transition_time * self.fs)

        # --- hemodynamic response function (HRF)
        def hrf(t, delay=2, dispersion=1):
            """Canonical hemodynamic response function"""
            return (t / delay) ** 2 * np.exp(-(t / delay)) / (2 * delay * dispersion ** 2)

        # --- initialize data array
        self.fnirs_data = np.zeros((n_channels, self.n_timepoints))

        # --- generate HRF kernel for convolution
        hrf_kernel = hrf(np.linspace(0, 10, 50))

        # --- channel 0: Fp1 (left pfc) - long separation channel
        # --- baseline oscillations --> resting state network activity
        fp1_baseline = 0.2 * np.sin(2 * np.pi * 0.08 * self.time)  # --- 0.08 Hz resting oscillation
        fp1_baseline += 0.1 * np.sin(2 * np.pi * 0.12 * self.time)  # --- additional frequency component
        fp1_baseline += np.random.randn(self.n_timepoints) * 0.05  # --- neural noise

        # --- attention-related activation stronger in left pfc for cognitive tasks
        attention_signal = np.zeros_like(self.time)
        attention_signal[transition_idx:] = 1.0
        fp1_activation = np.convolve(attention_signal, hrf_kernel, mode='same') * 0.8

        self.fnirs_data[0] = fp1_baseline + fp1_activation

        # --- channel 1: fp2 (right pfc) - long separation channel
        # --- similar to fp1 but with different baseline and slightly less activation
        fp2_baseline = 0.15 * np.sin(2 * np.pi * 0.09 * self.time + np.pi / 4)  # --- phase-shifted
        fp2_baseline += 0.08 * np.sin(2 * np.pi * 0.11 * self.time)
        fp2_baseline += np.random.randn(self.n_timepoints) * 0.05

        # --- right PFC shows moderate activation during attention tasks
        fp2_activation = np.convolve(attention_signal, hrf_kernel, mode='same') * 0.5

        self.fnirs_data[1] = fp2_baseline + fp2_activation

        # --- channel 2: short separation channel --> superficial physiological signals
        # --- primarily cardiac and vasomotion, minimal neural signal

        # --- heart rate variability ~1 Hz cardiac 60 BPM baseline
        heart_rate_base = 60  # --- beat per minute
        hr_variation = 5 * np.sin(2 * np.pi * 0.1 * self.time)  # --- HRV at 0.1 Hz
        instantaneous_hr = heart_rate_base + hr_variation

        # --- generate cardiac pulsation
        cardiac_signal = np.zeros_like(self.time)
        for i, t in enumerate(self.time):
            # --- cumulative cardiac phase
            if i > 0:
                dt = self.time[i] - self.time[i - 1]
                cardiac_phase = (instantaneous_hr[i] / 60) * 2 * np.pi * dt
                cardiac_signal[i] = 0.3 * np.sin(cardiac_phase * i)

        # --- vasomotion (~0.04 Hz)
        vasomotion = 0.15 * np.sin(2 * np.pi * 0.04 * self.time)

        # --- respiration artifact (~0.25 Hz, 15 breaths/min)
        respiration = 0.1 * np.sin(2 * np.pi * 0.25 * self.time)

        # --- motion artifacts (random spikes)
        motion_artifacts = np.random.randn(self.n_timepoints) * 0.02
        motion_spikes = np.random.rand(self.n_timepoints) < 0.01  # --- 1% chance of spike
        motion_artifacts[motion_spikes] += np.random.randn(np.sum(motion_spikes)) * 0.5

        # --- short separation has minimal brain signal, mostly physiological
        minimal_brain_signal = 0.05 * (fp1_baseline + fp2_baseline) / 2  # --- 5% of brain signal

        self.fnirs_data[2] = cardiac_signal + vasomotion + respiration + motion_artifacts + minimal_brain_signal

        # --- apply bandpass filter (typical fNIRS preprocessing)
        self.filter_data()

    def filter_data(self):
        """Apply bandpass filter to fNIRS data"""
        nyquist = self.fs / 2
        low, high = 0.01, 0.2  # --- typical fNIRS frequency range
        b, a = butter(4, [low / nyquist, high / nyquist], btype='band')

        for i in range(self.fnirs_data.shape[0]):
            self.fnirs_data[i] = filtfilt(b, a, self.fnirs_data[i])

    def create_state_space(self):
        """Create 3D state space representation using PCA"""
        from sklearn.decomposition import PCA

        # --- transpose for PCA (timepoints x channels)
        data_matrix = self.fnirs_data.T

        # --- apply PCA to reduce to 3D state space
        pca = PCA(n_components=3)
        self.state_space = pca.fit_transform(data_matrix)

        # --- store PCA information for interpretation
        self.pca_components = pca.components_
        self.explained_variance = pca.explained_variance_ratio_

        # --- def rest and focus periods for coloring
        self.rest_indices = self.time < 10
        self.focus_indices = self.time >= 10

        print(f"\npca decomp results:")
        print(f"PC1 explains {self.explained_variance[0]:.1%} of variance")
        print(f"PC2 explains {self.explained_variance[1]:.1%} of variance")
        print(f"PC3 explains {self.explained_variance[2]:.1%} of variance")
        print(f"Total explained: {sum(self.explained_variance):.1%}")

        print(f"\nch contribution to pc:")
        for i, channel in enumerate(self.channel_names):
            print(f"{channel}:")
            print(f"  PC1: {self.pca_components[0, i]:.3f}")
            print(f"  PC2: {self.pca_components[1, i]:.3f}")
            print(f"  PC3: {self.pca_components[2, i]:.3f}")

    def create_energy_landscape(self):
        """Create cognitive energy landscape based on state space"""
        # --- create coordinate grid for energy landscape
        pc1_range = np.linspace(self.state_space[:, 0].min() * 1.2, self.state_space[:, 0].max() * 1.2, 100)
        pc2_range = np.linspace(self.state_space[:, 1].min() * 1.2, self.state_space[:, 1].max() * 1.2, 100)
        self.landscape_x, self.landscape_y = np.meshgrid(pc1_range, pc2_range)

        # --- calculate average positions for rest and attention states
        rest_center = np.mean(self.state_space[self.rest_indices, :2], axis=0)
        attention_center = np.mean(self.state_space[self.focus_indices, :2], axis=0)

        print(f"\nenergy landscape centers:")
        print(f"rest center: pc1={rest_center[0]:.3f}, PC2={rest_center[1]:.3f}")
        print(f"attention center: pc1={attention_center[0]:.3f}, PC2={attention_center[1]:.3f}")

        # --- create energy surface
        # --- rest state = deep valley / basin (low energy)
        rest_valley = -2.0 * np.exp(-((self.landscape_x - rest_center[0]) ** 2 +
                                      (self.landscape_y - rest_center[1]) ** 2) / (2 * 0.3 ** 2))

        # --- attention state = elevated plateau (higher energy, requires effort)
        attention_hill = 1.0 * np.exp(-((self.landscape_x - attention_center[0]) ** 2 +
                                        (self.landscape_y - attention_center[1]) ** 2) / (2 * 0.5 ** 2))

        # --- base energy surface (slight upward trend away from rest)
        base_energy = 0.1 * (self.landscape_x ** 2 + self.landscape_y ** 2)

        # --- add some noise for realistic landscape
        noise = 0.1 * np.random.randn(*self.landscape_x.shape)
        noise = gaussian_filter(noise, sigma=2)

        # --- combine to create full energy landscape
        self.energy_landscape = base_energy + rest_valley + attention_hill + noise

        # --- smooth the landscape
        self.energy_landscape = gaussian_filter(self.energy_landscape, sigma=1)

        # --- calculate energy trajectory (marble position over time)
        self.marble_positions = self.state_space[:, :2]  # Use PC1 and PC2

        # --- calculate energy values for each time point
        self.energy_trajectory = np.zeros(self.n_timepoints)
        for i in range(self.n_timepoints):
            # --- interpolate energy at marble position
            pc1_pos = self.marble_positions[i, 0]
            pc2_pos = self.marble_positions[i, 1]

            # --- find closest grid points for interpolation
            pc1_idx = np.argmin(np.abs(pc1_range - pc1_pos))
            pc2_idx = np.argmin(np.abs(pc2_range - pc2_pos))

            self.energy_trajectory[i] = self.energy_landscape[pc2_idx, pc1_idx]

    def setup_animation(self):
        """Setup the animation with 2x2 subplot layout including energy landscape"""
        self.fig = plt.figure(figsize=(16, 12))

        # --- main 3D plot (top left)
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_facecolor('black')

        # --- energy landscape (top right)
        self.ax_energy = self.fig.add_subplot(222)
        self.ax_energy.set_facecolor('black')

        # Remove time series plots (bottom)
        # self.ax_ts1 = self.fig.add_subplot(223)
        # self.ax_ts2 = self.fig.add_subplot(224)

        # --- initialize 3D state space plot
        self.trajectory_line, = self.ax_3d.plot([], [], [], 'cyan', alpha=0.6, linewidth=2)
        self.current_point = self.ax_3d.scatter([], [], [], c='red', s=100, alpha=0.8)
        self.rest_points = self.ax_3d.scatter([], [], [], c='blue', s=20, alpha=0.3)
        self.focus_points = self.ax_3d.scatter([], [], [], c='orange', s=20, alpha=0.3)

        # --- set labels and title for 3D plot
        self.ax_3d.set_xlabel(f'pc1 ({self.explained_variance[0]:.1%} var)', fontsize=10)
        self.ax_3d.set_ylabel(f'pc2 ({self.explained_variance[1]:.1%} var)', fontsize=10)
        self.ax_3d.set_zlabel(f'pc3 ({self.explained_variance[2]:.1%} var)', fontsize=10)
        self.ax_3d.set_title('cognitive state-space trajectory\n(3d pca space)', fontsize=12, pad=20)

        # --- set axis limits for 3D plot
        margin = 0.1
        for ax_method, data_col in zip([self.ax_3d.set_xlim, self.ax_3d.set_ylim, self.ax_3d.set_zlim],
                                       [0, 1, 2]):
            data_range = self.state_space[:, data_col]
            ax_method([data_range.min() - margin, data_range.max() + margin])

        # --- initialize energy landscape plot
        self.energy_contour = self.ax_energy.contourf(self.landscape_x, self.landscape_y,
                                                      self.energy_landscape, levels=20,
                                                      cmap='plasma', alpha=0.8)
        self.ax_energy.contour(self.landscape_x, self.landscape_y, self.energy_landscape,
                               levels=10, colors='white', alpha=0.3, linewidths=0.5)

        # --- initialize marble on energy landscape
        self.marble_dot, = self.ax_energy.plot([], [], 'ro', markersize=12,
                                               markerfacecolor='red', markeredgecolor='white',
                                               markeredgewidth=2, alpha=1.0, zorder=10)
        self.marble_trail, = self.ax_energy.plot([], [], 'cyan', alpha=0.6, linewidth=2, zorder=5)

        self.ax_energy.set_xlabel('pc1 (brain state d1)', fontsize=10)
        self.ax_energy.set_ylabel('pc2 (brain state d2)', fontsize=10)
        self.ax_energy.set_title('cognitive energy landscape\n(marble = current cognitive state)', fontsize=12)

        # --- add colorbar for energy landscape
        cbar = plt.colorbar(self.energy_contour, ax=self.ax_energy, shrink=0.8)
        cbar.set_label('cognitive energy', fontsize=9)

        # Remove time series plot setup
        # self.setup_time_series_plots()

        # --- add text annotations
        self.time_text = self.ax_3d.text2D(0.02, 0.98, '', transform=self.ax_3d.transAxes,
                                           fontsize=10, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        self.state_text = self.ax_3d.text2D(0.02, 0.88, '', transform=self.ax_3d.transAxes,
                                            fontsize=9, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # --- energy text annotation
        self.energy_text = self.ax_energy.text(0.02, 0.98, '', transform=self.ax_energy.transAxes,
                                               fontsize=10, verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                                               color='white')

        plt.tight_layout()

    def animate(self, frame):
        """Animation function"""
        if frame < len(self.time):
            current_time = self.time[frame]

            # --- update 3D trajectory
            trajectory_data = self.state_space[:frame + 1]
            if len(trajectory_data) > 1:
                self.trajectory_line.set_data_3d(trajectory_data[:, 0],
                                                 trajectory_data[:, 1],
                                                 trajectory_data[:, 2])

            # --- update current point in 3D
            current_pos = self.state_space[frame]
            self.current_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])

            # --- update rest/focus point clouds in 3D
            rest_mask = self.time[:frame + 1] < 10
            focus_mask = self.time[:frame + 1] >= 10

            if np.any(rest_mask):
                rest_data = self.state_space[:frame + 1][rest_mask]
                self.rest_points._offsets3d = (rest_data[:, 0], rest_data[:, 1], rest_data[:, 2])

            if np.any(focus_mask):
                focus_data = self.state_space[:frame + 1][focus_mask]
                self.focus_points._offsets3d = (focus_data[:, 0], focus_data[:, 1], focus_data[:, 2])

            # --- update energy landscape marble
            marble_pos = self.marble_positions[frame]
            self.marble_dot.set_data([marble_pos[0]], [marble_pos[1]])

            # --- update marble trail
            trail_data = self.marble_positions[:frame + 1]
            self.marble_trail.set_data(trail_data[:, 0], trail_data[:, 1])

            # Remove time series update
            # self.update_time_series(frame)

            # --- update text annotations
            brain_state = "resting state" if current_time < 10 else "focused attention"
            energy_level = self.energy_trajectory[frame]

            self.time_text.set_text(f'time: {current_time:.1f}s')
            self.state_text.set_text(f'state: {brain_state}')
            self.energy_text.set_text(f'energy: {energy_level:.2f}\nstate: {brain_state}')

            # --- rotate 3D view slowly
            self.ax_3d.view_init(elev=20, azim=frame * 0.5)

        return [self.trajectory_line, self.current_point, self.rest_points, self.focus_points,
                self.marble_dot, self.marble_trail]

    def run_animation(self, interval=100, save_as=None):
        """Run the animation"""
        frames = len(self.time)
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=frames,
                                            interval=interval, blit=False, repeat=True)

        if save_as:
            print(f"saving animation as {save_as}...")
            self.anim.save(save_as, writer='pillow', fps=10, dpi=100)
            print("animation saved relative dir")

        plt.show(block=True)
        return self.anim


# --- create and run the animation
if __name__ == "__main__":
    print("initializing brain state-space + energy landscape animation...")
    print("simulating fnirs data with cognitive energy dynamics...")

    # --- create the animator
    animator = BrainStateSpaceAnimator(duration=30, sampling_rate=10)

    print("starting animation...")
    print("- top left: 3d brain state-space trajectory")
    print("- top right: cognitive energy landscape with marble")
    print("- bottom: time series of signals and components")
    print("- red marble shows current brain state and energy level")
    print("- cognitive state transition occurs at t=10s")
    print("- energy increases when transitioning from rest to attention")

    # --- run the animation
    # anim = animator.run_animation(interval=100, save_as='brain_statespace_energy.gif') # --- Uncomment the next line to save as GIF
    anim = animator.run_animation(interval=100)