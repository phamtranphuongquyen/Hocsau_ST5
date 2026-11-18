import unittest
from pathlib import Path

import torch

from train import balanced_class_weights


class BalancedClassWeightsTests(unittest.TestCase):
    def test_inverse_frequency_weights_have_mean_one(self):
        samples = [
            (Path("angry-1.jpg"), 0),
            (Path("angry-2.jpg"), 0),
            (Path("happy-1.jpg"), 1),
        ]

        weights = balanced_class_weights(samples, num_classes=2)

        torch.testing.assert_close(weights, torch.tensor([0.75, 1.5]))
        sample_weighted_mean = (weights[0] * 2 + weights[1]) / 3
        self.assertAlmostEqual(sample_weighted_mean.item(), 1.0)

    def test_empty_class_is_rejected(self):
        with self.assertRaises(ValueError):
            balanced_class_weights([(Path("only.jpg"), 0)], num_classes=2)


if __name__ == "__main__":
    unittest.main()
