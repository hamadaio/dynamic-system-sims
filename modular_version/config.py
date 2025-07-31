"""
Configuration Module

This module contains configuration settings and parameters
for the brain state space simulation.
"""


class SimulationConfig:
    """Configuration settings for the brain state space simulation"""
    
    # Time parameters
    DURATION = 30  # seconds
    SAMPLING_RATE = 10  # Hz similar to Mendi fNIRS device
    TRANSITION_TIME = 10  # seconds - when attention task begins
    
    # Animation parameters
    ANIMATION_INTERVAL = 100  # milliseconds between frames
    SAVE_ANIMATION = True
    OUTPUT_FILENAME = 'brain_statespace.gif'
    
    # Signal generation parameters
    CHANNELS = {
        'fp1': 'fp1 (left-pfc)',
        'fp2': 'fp2 (right-pfc)', 
        'short_sep': 'short-sep ch (physio noise)'
    }
    
    # HRF parameters (Single Gamma Function)
    # Current implementation uses a simplified Single Gamma HRF:
    # Formula: (t/delay)² × exp(-(t/delay)) / (2 × delay × dispersion²)
    # - Produces single positive peak without negative undershoot
    # - More realistic models: Double Gamma (SPM canonical) or Triple Gamma
    # - Time-to-peak typically 4-6 seconds for BOLD/fNIRS responses
    HRF_DELAY = 2  # seconds (time-to-peak parameter)
    HRF_DISPERSION = 1  # shape parameter (controls width)
    
    # Filter parameters
    # NOTE: When increasing SAMPLING_RATE above 10 Hz, you may notice "shallower" 
    # state-space trajectories and different visual appearance. This occurs because:
    # 1. Nyquist frequency changes: Higher sampling = higher Nyquist, making the same
    #    filter cutoffs (0.01-0.2 Hz) relatively more restrictive
    # 2. Signal-to-noise ratio: More timepoints = better noise averaging = smoother signals
    # 3. Temporal resolution: Higher sampling captures more intermediate values, 
    #    creating smoother transitions that appear less dramatic
    # 4. PCA variance distribution: More data points can change how variance is
    #    distributed across principal components
    # 
    # To compensate for higher sampling rates (e.g., 31.25 Hz), consider:
    # - Increasing FILTER_HIGH (e.g., 0.3 Hz for 31.25 Hz sampling)
    # - Slightly decreasing FILTER_LOW (e.g., 0.008 Hz)
    # - Adding amplitude scaling in the signal generator if needed
    FILTER_LOW = 0.01  # Hz
    FILTER_HIGH = 0.2  # Hz
    FILTER_ORDER = 4
    
    # PCA parameters
    N_COMPONENTS = 3
    
    # Visualization parameters
    FIGURE_SIZE = (15, 10)
    DPI = 100
    FPS = 10
    
    # Colors
    COLORS = {
        'fp1': '#e74c3c',  # Red
        'fp2': '#3498db',  # Blue
        'short_sep': '#2ecc71',  # Green
        'pc1': 'cyan',
        'pc2': 'magenta',
        'pc3': 'yellow',
        'rest': 'blue',
        'focus': 'orange',
        'current': 'red',
        'trajectory': 'cyan'
    }
    
    @classmethod
    def get_channel_names(cls):
        """Get list of channel names in order"""
        return [cls.CHANNELS['fp1'], cls.CHANNELS['fp2'], cls.CHANNELS['short_sep']]
    
    @classmethod
    def get_component_colors(cls):
        """Get list of component colors in order"""
        return [cls.COLORS['pc1'], cls.COLORS['pc2'], cls.COLORS['pc3']]
    
    @classmethod
    def get_channel_colors(cls):
        """Get list of channel colors in order"""
        return [cls.COLORS['fp1'], cls.COLORS['fp2'], cls.COLORS['short_sep']]
