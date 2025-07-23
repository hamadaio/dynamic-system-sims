# Dynamic Systems Simulations

This repository contains Python scripts for simulating and visualizing dynamic systems, particularly focusing on brain state-space dynamics using fNIRS (functional Near-Infrared Spectroscopy) data.

## Scripts

-   `dst_space-state_basin_attractor.py`: Simulates brain state-space trajectories and visualizes them within a cognitive energy landscape, demonstrating concepts like basin attractors.
-   `dst_state-space_3ch_mendi.py`: Generates and visualizes 3-channel fNIRS data, projecting it into a 3D state-space to illustrate transitions between cognitive states (e.g., rest to attention).
-   `dst_space-state_multi_ch.py`: (Likely) Extends the state-space analysis to multiple fNIRS channels.

## Features

-   **fNIRS Data Simulation**: Generates realistic fNIRS signals for various brain regions.
-   **State-Space Analysis**: Utilizes PCA (Principal Component Analysis) to reduce high-dimensional fNIRS data into a lower-dimensional brain state-space.
-   **Cognitive Energy Landscapes**: Visualizes brain states as a "marble" moving across an energy landscape, representing the energetic cost of cognitive transitions.
-   **Animations**: Generates animated visualizations of brain state trajectories and energy landscape dynamics.