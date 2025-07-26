"""
fNIRS Data Generator Module

This module generates realistic fNIRS signals for prefrontal cortex channels,
simulating brain activity during rest-to-attention transitions.
"""

import numpy as np
from scipy.signal import butter, filtfilt


class FNIRSDataGenerator:
    """Generator for realistic fNIRS signals with physiological components"""
    
    def __init__(self, duration=30, sampling_rate=10):
        self.duration = duration  # seconds
        self.fs = sampling_rate  # Hz
        self.time = np.linspace(0, duration, int(duration * sampling_rate))
        self.n_timepoints = len(self.time)
        self.channel_names = ['fp1 (left-pfc)', 'fp2 (right-pfc)', 'short-sep ch (physio noise)']
        self.fnirs_data = None
        
    def generate_fnirs_data(self):
        """Generate realistic fNIRS signals for prefrontal cortex channels"""
        n_channels = 3

        # Create transition from rest to focus at t=10s
        transition_time = 10
        transition_idx = int(transition_time * self.fs)

        # Initialize data array
        self.fnirs_data = np.zeros((n_channels, self.n_timepoints))

        # Generate HRF kernel for convolution
        hrf_kernel = self._generate_hrf_kernel()

        # Generate each channel
        self.fnirs_data[0] = self._generate_fp1_channel(transition_idx, hrf_kernel)
        self.fnirs_data[1] = self._generate_fp2_channel(transition_idx, hrf_kernel)
        self.fnirs_data[2] = self._generate_short_separation_channel()

        # Apply bandpass filter (typical fNIRS preprocessing)
        self._filter_data()
        
        return self.fnirs_data, self.channel_names, self.time
        
    def _generate_hrf_kernel(self):
        """Generate hemodynamic response function kernel
        
        Uses a Single Gamma HRF function (simplified model):
        - Formula: (t/delay)² x exp(-(t/delay)) / (2 x delay x dispersion²)
        - Characteristics: Single positive peak, no negative undershoot
        - Time-to-peak: ~4-6 seconds (realistic for BOLD/fNIRS)
        - Note: More realistic models use Double Gamma (with negative undershoot)
          or Triple Gamma functions for complex physiological responses
        """
        def hrf(t, delay=2, dispersion=1):
            """Single Gamma hemodynamic response function
            
            Args:
                t: time vector (seconds)
                delay: time-to-peak parameter (seconds) 
                dispersion: shape parameter (unitless)
            
            Returns:
                HRF values at time points t
            """
            return (t / delay) ** 2 * np.exp(-(t / delay)) / (2 * delay * dispersion ** 2)
        
        return hrf(np.linspace(0, 10, 50))
    
    def _generate_fp1_channel(self, transition_idx, hrf_kernel):
        """Generate Fp1 (Left Prefrontal Cortex) channel data"""
        # Baseline oscillations (resting state network activity)
        fp1_baseline = 0.2 * np.sin(2 * np.pi * 0.08 * self.time)  # 0.08 Hz resting oscillation
        fp1_baseline += 0.1 * np.sin(2 * np.pi * 0.12 * self.time)  # Additional frequency component
        fp1_baseline += np.random.randn(self.n_timepoints) * 0.05  # Neural noise

        # Attention-related activation (stronger in left PFC for cognitive tasks)
        attention_signal = np.zeros_like(self.time)
        attention_signal[transition_idx:] = 1.0
        fp1_activation = np.convolve(attention_signal, hrf_kernel, mode='same') * 0.8

        return fp1_baseline + fp1_activation
    
    def _generate_fp2_channel(self, transition_idx, hrf_kernel):
        """Generate Fp2 (Right Prefrontal Cortex) channel data"""
        # Similar to Fp1 but with different baseline and slightly less activation
        fp2_baseline = 0.15 * np.sin(2 * np.pi * 0.09 * self.time + np.pi / 4)  # Phase-shifted
        fp2_baseline += 0.08 * np.sin(2 * np.pi * 0.11 * self.time)
        fp2_baseline += np.random.randn(self.n_timepoints) * 0.05

        # Right PFC shows moderate activation during attention tasks
        attention_signal = np.zeros_like(self.time)
        attention_signal[transition_idx:] = 1.0
        fp2_activation = np.convolve(attention_signal, hrf_kernel, mode='same') * 0.5

        return fp2_baseline + fp2_activation
    
    def _generate_short_separation_channel(self):
        """Generate short separation channel (superficial physiological signals)"""
        # Heart rate variability (~1 Hz cardiac, 60 BPM baseline)
        heart_rate_base = 60  # BPM
        hr_variation = 5 * np.sin(2 * np.pi * 0.1 * self.time)  # HRV at 0.1 Hz
        instantaneous_hr = heart_rate_base + hr_variation

        # Generate cardiac pulsation
        cardiac_signal = np.zeros_like(self.time)
        for i, _ in enumerate(self.time):
            # Cumulative cardiac phase
            if i > 0:
                dt = self.time[i] - self.time[i - 1]
                cardiac_phase = (instantaneous_hr[i] / 60) * 2 * np.pi * dt
                cardiac_signal[i] = 0.3 * np.sin(cardiac_phase * i)

        # Vasomotion (~0.04 Hz)
        vasomotion = 0.15 * np.sin(2 * np.pi * 0.04 * self.time)

        # Respiration artifact (~0.25 Hz, 15 breaths/min)
        respiration = 0.1 * np.sin(2 * np.pi * 0.25 * self.time)

        # Motion artifacts (random spikes)
        motion_artifacts = np.random.randn(self.n_timepoints) * 0.02
        motion_spikes = np.random.rand(self.n_timepoints) < 0.01  # 1% chance of spike
        motion_artifacts[motion_spikes] += np.random.randn(np.sum(motion_spikes)) * 0.5

        # Short separation has minimal brain signal, mostly physiological
        minimal_brain_signal = 0.05 * (
            self._generate_fp1_channel(int(10 * self.fs), self._generate_hrf_kernel()) + 
            self._generate_fp2_channel(int(10 * self.fs), self._generate_hrf_kernel())
        ) / 2  # 5% of brain signal

        return cardiac_signal + vasomotion + respiration + motion_artifacts + minimal_brain_signal

    def _filter_data(self):
        """Apply bandpass filter to fNIRS data"""
        nyquist = self.fs / 2
        low, high = 0.01, 0.2  # Typical fNIRS frequency range
        b, a = butter(4, [low / nyquist, high / nyquist], btype='band')

        for i in range(self.fnirs_data.shape[0]):
            self.fnirs_data[i] = filtfilt(b, a, self.fnirs_data[i])
