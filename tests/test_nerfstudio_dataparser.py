import unittest
from pathlib import Path

import numpy as np

from bad_gaussians.nerf_studio_dataparser import get_split_indices


class NerfstudioDataparserSplitTest(unittest.TestCase):
    def setUp(self):
        self.filenames = [Path(f"frame_{index:04}.png") for index in range(17)]

    def test_interval_split_is_disjoint_and_complete(self):
        train, evaluation = get_split_indices(self.filenames, "interval", 0.8, 8)
        np.testing.assert_array_equal(evaluation, np.array([0, 8, 16]))
        self.assertFalse(set(train).intersection(evaluation))
        self.assertEqual(set(train).union(evaluation), set(range(17)))

    def test_all_split_preserves_all_indices(self):
        train, evaluation = get_split_indices(self.filenames, "all", 0.8, 8)
        np.testing.assert_array_equal(train, np.arange(17))
        np.testing.assert_array_equal(evaluation, np.arange(17))


if __name__ == "__main__":
    unittest.main()
