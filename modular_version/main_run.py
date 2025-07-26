"""
Main Runner for Brain State Space Animation

This script orchestrates the entire brain state space simulation,
from data generation to visualization.
"""

import matplotlib
matplotlib.use('TkAgg')  # Enable animations in PyCharm

import matplotlib.pyplot as plt
from fnirs_data_generator import FNIRSDataGenerator
from state_space_analyzer import StateSpaceAnalyzer
from brain_state_visualizer import BrainStateVisualizer
from config import SimulationConfig

# Enable interactive mode
plt.ion()


class BrainStateSpaceSimulation:
    """Main orchestrator for the brain state space simulation"""
    
    def __init__(self, config=None):
        self.config = config or SimulationConfig()
        self.data_generator = None
        self.state_analyzer = None
        self.visualizer = None
        
        # Data containers
        self.fnirs_data = None
        self.channel_names = None
        self.time_vector = None
        self.state_space = None
        self.explained_variance = None
        
    def run_simulation(self):
        """Run the complete simulation pipeline"""
        print("Initializing Brain State-Space Animation...")
        print("Simulating fNIRS data with transition from rest to focused attention...")
        
        # Step 1: Generate fNIRS data
        self._generate_data()
        
        # Step 2: Perform state space analysis
        self._analyze_state_space()
        
        # Step 3: Setup and run visualization
        self._visualize_results()
        
    def _generate_data(self):
        """Generate synthetic fNIRS data"""
        print("Generating fNIRS data...")
        
        self.data_generator = FNIRSDataGenerator(
            duration=self.config.DURATION,
            sampling_rate=self.config.SAMPLING_RATE
        )
        
        self.fnirs_data, self.channel_names, self.time_vector = self.data_generator.generate_fnirs_data()
        print(f"Generated {self.fnirs_data.shape[0]} channels with {self.fnirs_data.shape[1]} timepoints")
        
    def _analyze_state_space(self):
        """Perform state space analysis using PCA"""
        print("Performing state space analysis...")
        
        self.state_analyzer = StateSpaceAnalyzer()
        self.state_space, self.explained_variance = self.state_analyzer.create_state_space(
            self.fnirs_data, self.time_vector, self.channel_names
        )
        
    def _visualize_results(self):
        """Setup and run the visualization"""
        print("Setting up visualization...")
        
        self.visualizer = BrainStateVisualizer(
            fnirs_data=self.fnirs_data,
            state_space=self.state_space,
            time_vector=self.time_vector,
            channel_names=self.channel_names,
            explained_variance=self.explained_variance
        )
        
        self.visualizer.setup_animation()
        
        print("Starting animation...")
        print("- blue dots: resting state")
        print("- orange dots: focused attention state")
        print("- red dots: Current brain state")
        print(f"- cognitive state transition occurs at t={self.config.TRANSITION_TIME}s")
        
        # Run the animation
        save_file = self.config.OUTPUT_FILENAME if self.config.SAVE_ANIMATION else None
        anim = self.visualizer.run_animation(
            interval=self.config.ANIMATION_INTERVAL,
            save_as=save_file
        )
        
        return anim


def main():
    """Main entry point"""
    # Create and run simulation
    simulation = BrainStateSpaceSimulation()
    animation = simulation.run_simulation()
    
    return animation


if __name__ == "__main__":
    main()
