"""
State Space Analysis Module

This module performs dimensionality reduction and state space analysis
on fNIRS data using Principal Component Analysis (PCA).
"""

from sklearn.decomposition import PCA


class StateSpaceAnalyzer:
    """Performs state space analysis on fNIRS data using PCA"""
    
    def __init__(self):
        self.state_space = None
        self.pca_components = None
        self.explained_variance = None
        self.rest_indices = None
        self.focus_indices = None
        
    def create_state_space(self, fnirs_data, time_vector, channel_names):
        """Create 3D state space representation using PCA"""
        # Transpose for PCA (timepoints x channels)
        data_matrix = fnirs_data.T

        # Apply PCA to reduce to 3D state space
        pca = PCA(n_components=3)
        self.state_space = pca.fit_transform(data_matrix)

        # Store PCA information for interpretation
        self.pca_components = pca.components_
        self.explained_variance = pca.explained_variance_ratio_

        # Define rest and focus periods for coloring
        self.rest_indices = time_vector < 10
        self.focus_indices = time_vector >= 10

        # Print analysis results
        self._print_pca_results(channel_names)
        
        return self.state_space, self.explained_variance
        
    def _print_pca_results(self, channel_names):
        """Print PCA decomposition results"""
        print("\nPCA decomp results:")
        print(f"PC1 explains {self.explained_variance[0]:.1%} of variance")
        print(f"PC2 explains {self.explained_variance[1]:.1%} of variance")
        print(f"PC3 explains {self.explained_variance[2]:.1%} of variance")
        print(f"Total explained: {sum(self.explained_variance):.1%}")

        print("\nch contribution to PC:")
        for i, channel in enumerate(channel_names):
            print(f"{channel}:")
            print(f"  PC1: {self.pca_components[0, i]:.3f}")
            print(f"  PC2: {self.pca_components[1, i]:.3f}")
            print(f"  PC3: {self.pca_components[2, i]:.3f}")
            
    def get_rest_focus_masks(self, time_vector, current_frame):
        """Get boolean masks for rest and focus periods up to current frame"""
        rest_mask = time_vector[:current_frame + 1] < 10
        focus_mask = time_vector[:current_frame + 1] >= 10
        return rest_mask, focus_mask
