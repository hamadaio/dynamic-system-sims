import unittest
from unittest.mock import patch
import numpy as np
from dst_sim_attr_landscape import BrainStateSpaceAnimator

class TestBrainStateSpaceAnimator(unittest.TestCase):
    def setUp(self):
        """Set up a common BrainStateSpaceAnimator instance for tests."""
        with patch.object(BrainStateSpaceAnimator, 'setup_animation', return_value=None):
            self.animator = BrainStateSpaceAnimator(duration=10, sampling_rate=5)

    def test_initialization(self):
        """Test if the animator is initialized with the correct parameters."""
        self.assertEqual(self.animator.duration, 10)
        self.assertEqual(self.animator.fs, 5)
        self.assertEqual(self.animator.n_timepoints, 50)
        self.assertEqual(len(self.animator.time), 50)

    def test_generate_fnirs_data(self):
        """Test the shape and basic properties of the generated fNIRS data."""
        self.assertEqual(self.animator.fnirs_data.shape, (3, 50))
        self.assertFalse(np.all(self.animator.fnirs_data == 0))

    def test_filter_data(self):
        """Test that the bandpass filter is applied correctly."""
        unfiltered_data = self.animator.fnirs_data.copy()
        self.animator.filter_data()
        self.assertFalse(np.all(unfiltered_data == self.animator.fnirs_data))

    def test_create_state_space(self):
        """Test the shape of the state space and the explained variance."""
        self.assertEqual(self.animator.state_space.shape, (50, 3))
        self.assertEqual(len(self.animator.explained_variance), 3)
        self.assertAlmostEqual(sum(self.animator.explained_variance), 1.0, places=1)

    def test_create_energy_landscape(self):
        """Test the shape of the energy landscape and the energy trajectory."""
        self.assertEqual(self.animator.energy_landscape.shape, (100, 100))
        self.assertEqual(len(self.animator.energy_trajectory), 50)

if __name__ == '__main__':
    unittest.main()
