# Brain state-space simulation using HRF - modular

This is a modular implementation of a brain state space simulation using fNIRS data. The simulation demonstrates the transition from resting state to focused attention using dynamic systems theory and 3D visualization.

## Project structure

```
dynamic-systems-sims/
├── main_run.py                    # Main entry point - runs the complete simulation
├── fnirs_data_generator.py        # Generates synthetic fNIRS signals using HRF
├── state_space_analyzer.py        # Performs PCA and state space analysis  
├── brain_state_visualizer.py      # Handles 3D animation and visualization
├── config.py                      # Configuration settings and parameters
├── requirements.txt               # Python dependencies
└── README_modular.md              # This file
```

## Module description

### `main_run.py`
- **Purpose**: Main orchestrator that coordinates all modules
- **Functions**: Runs the complete simulation pipeline from data generation to visualization
- **Usage**: `python main_run.py`

### `fnirs_data_generator.py`
- **Purpose**: Generates realistic fNIRS signals for prefrontal cortex channels
- **Key features**:
  - Simulates 3 channels: Fp1 (left PFC), Fp2 (right PFC), short separation (physio noise)
  - Includes Single Gamma hemodynamic response function (HRF)
  - Generates physiological artifacts (cardiac, respiration, motion)
  - Applies bandpass filtering

### `state_space_analyzer.py` 
- **Purpose**: Performs dimensionality reduction using Principal Component Analysis
- **Key features**:
  - Reduces 3-channel fNIRS data to 3D state space
  - Calculates explained variance for each component
  - Identifies rest vs focus periods
  - Provides interpretation of component contributions

### `brain_state_visualizer.py`
- **Purpose**: Creates 3D animation and visualization of brain state dynamics
- **Key features**:
  - Real-time 3D trajectory visualization
  - Time series plots of raw fNIRS signals
  - State space component plots over time
  - Phase portrait (PC1 vs PC2)
  - Animated transition from rest to attention state

### `config.py`
- **Purpose**: Centralized configuration management
- **Contains**:
  - Time parameters (duration, sampling rate)
  - Signal generation parameters
  - Animation settings
  - Color schemes
  - Filter parameters

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main_run.py
```

## Configuration

Modify `config.py` to adjust simulation parameters:

- `DURATION`: Simulation length in seconds
- `SAMPLING_RATE`: Data sampling frequency in Hz
- `TRANSITION_TIME`: When attention task begins
- `SAVE_ANIMATION`: Whether to save animation as GIF
- `OUTPUT_FILENAME`: Name of output animation file

## Key feature

1. **Realistic fNIRS simulation**: Generates physiologically plausible brain signals
2. **State space analysis**: Uses PCA to visualize brain state dynamics
3. **Real-time animation**: 3D visualization of cognitive state transitions
4. **Modular design**: Easy to modify individual components
5. **Configurable**: Adjustable parameters without code changes

## Output

The simulation produces:
- Console output with PCA analysis results
- Interactive 3D animation window
- Optional saved GIF animation file
- Multiple subplot views (3D trajectory, time series, phase portrait)

## Cognitive states

- **Blue dots**: Resting/baseline brain state (t < 10s)
- **Orange dots**: Focused attention state (t ≥ 10s)  
- **Red dot**: Current brain state
- **Cyan line**: State space trajectory

## Scientific background

This simulation demonstrates concepts from:
- **Dynamic Systems Theory**: Brain as a dynamical system with attractor states
- **fNIRS Neuroimaging**: Near-infrared spectroscopy for measuring brain activity
- **State Space Analysis**: Dimensionality reduction for understanding neural dynamics
- **Cognitive Neuroscience**: Prefrontal cortex role in attention and executive control

### Hemodynamic Response Function (HRF)

The simulation uses a **Single Gamma HRF function**:
- **Formula**: `(t/delay)² × exp(-(t/delay)) / (2 × delay × dispersion²)`
- **Characteristics**: Single positive peak, time-to-peak ~4-6 seconds
- **Limitation**: No negative undershoot (present in real BOLD/fNIRS responses)
- **Alternative models**: Double Gamma (SPM canonical) or Triple Gamma for more realism
