import unittest

import torch

from fisher_edl.metrics import (
    compute_aupr,
    compute_auroc,
    compute_ece,
    compute_fpr_at_tpr,
    compute_aurc,
)


class MetricTests(unittest.TestCase):
    def test_auroc_perfect_separation(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        auroc = compute_auroc(scores, labels)
        self.assertAlmostEqual(auroc, 1.0, places=6)

    def test_aupr_perfect_separation(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        aupr = compute_aupr(scores, labels)
        self.assertAlmostEqual(aupr, 1.0, places=6)

    def test_fpr95_bounds(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        fpr95 = compute_fpr_at_tpr(scores, labels, tpr_target=0.95)
        self.assertGreaterEqual(fpr95, 0.0)
        self.assertLessEqual(fpr95, 1.0)

    def test_ece_perfect_calibration(self):
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        ece = compute_ece(probs, labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_aurc_reasonable_range(self):
        logits = torch.tensor([[5.0, 1.0], [4.0, 0.5], [0.2, 3.0], [0.1, 2.0]])
        labels = torch.tensor([0, 0, 1, 1])
        aurc = compute_aurc(logits, labels)
        self.assertGreaterEqual(aurc, 0.0)
        self.assertLessEqual(aurc, 1.0)


if __name__ == "__main__":
    unittest.main()
