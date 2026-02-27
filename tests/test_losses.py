import unittest

import torch

from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss, fisher_weight


class LossTests(unittest.TestCase):
    def test_extreme_logits_are_finite(self):
        logits = torch.tensor([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 10.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

        edl_loss, _ = edl_mse_loss(logits, targets, kl_weight=1.0)
        fisher_loss, _ = fisher_edl_mse_loss(logits, targets, beta=1.0, gamma=1.0)

        self.assertTrue(torch.isfinite(edl_loss))
        self.assertTrue(torch.isfinite(fisher_loss))

    def test_weight_monotonic_under_evidence_proxy(self):
        alpha_low_info = torch.full((4, 3), 1.1)
        alpha_high_info = torch.full((4, 3), 15.0)

        w_low = fisher_weight(alpha_low_info, beta=1.0, gamma=1.0, info_type="evidence", gate_type="exp")
        w_high = fisher_weight(alpha_high_info, beta=1.0, gamma=1.0, info_type="evidence", gate_type="exp")

        self.assertGreater(w_low.mean().item(), w_high.mean().item())

    def test_kl_only_objective(self):
        logits = torch.randn(8, 5)
        targets = torch.nn.functional.one_hot(torch.randint(0, 5, size=(8,)), num_classes=5).float()

        loss, stats = fisher_edl_mse_loss(
            logits,
            targets,
            beta=1.0,
            gamma=1.0,
            objective="kl_only",
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(stats["kl_weighted"], 0.0)


if __name__ == "__main__":
    unittest.main()
