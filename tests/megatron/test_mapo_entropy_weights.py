import unittest

import torch


class TestMAPOEntropyWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.megatron.trainers.mapo_trainer import MegatronMAPOTrainer
            cls.trainer_cls = MegatronMAPOTrainer
            cls.available = True
        except Exception as e:  # noqa: BLE001
            print(f'Warning: MegatronMAPOTrainer import unavailable: {e}')
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest('MegatronMAPOTrainer unavailable in this environment')

    def _new_trainer(self, mode='hard', scope='sequence', quantile=0.5):
        trainer = self.trainer_cls.__new__(self.trainer_cls)
        trainer.entropy_mask_type = mode
        trainer.entropy_mask_scope = scope
        trainer.entropy_mask_quantile = quantile
        return trainer

    def test_hard_sequence_keeps_top_quantile(self):
        trainer = self._new_trainer(mode='hard', scope='sequence', quantile=1 / 3)
        entropies = torch.tensor([[0.1, 0.2, 0.9, float('nan')], [0.5, 0.1, 0.2, float('nan')]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)

        weights, metrics = trainer._build_mapo_entropy_weights(entropies, completion_mask)

        expected = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(weights, expected))
        self.assertAlmostEqual(metrics['entropy_kept_frac'].item(), 2.0 / 6.0, places=6)

    def test_soft_sequence_normalization(self):
        trainer = self._new_trainer(mode='soft', scope='sequence')
        entropies = torch.tensor([[0.1, 0.3, 0.6, float('nan')], [0.4, 0.2, 0.4, float('nan')]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)

        weights, metrics = trainer._build_mapo_entropy_weights(entropies, completion_mask)

        self.assertAlmostEqual(weights[0, :3].sum().item(), 1.0, places=6)
        self.assertAlmostEqual(weights[1, :3].sum().item(), 1.0, places=6)
        self.assertEqual(weights[0, 3].item(), 0.0)
        self.assertEqual(weights[1, 3].item(), 0.0)
        self.assertLess(weights[0, 0].item(), weights[0, 2].item())
        self.assertTrue(torch.isnan(metrics['entropy_threshold']))
        self.assertAlmostEqual(metrics['entropy_kept_frac'].item(), 1.0, places=6)

    def test_soft_global_normalization(self):
        trainer = self._new_trainer(mode='soft', scope='global')
        entropies = torch.tensor([[0.2, 0.8, float('nan')], [0.5, 0.5, float('nan')]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)

        weights, _ = trainer._build_mapo_entropy_weights(entropies, completion_mask)

        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertEqual(weights[0, 2].item(), 0.0)
        self.assertEqual(weights[1, 2].item(), 0.0)

    def test_soft_group_normalization(self):
        trainer = self._new_trainer(mode='soft', scope='group')
        entropies = torch.tensor([[0.1, 0.2], [0.7, 0.0], [0.5, 0.5], [0.2, 0.8]], dtype=torch.float32)
        completion_mask = torch.ones_like(entropies, dtype=torch.bool)
        prompt_group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        weights, _ = trainer._build_mapo_entropy_weights(entropies, completion_mask, prompt_group_ids=prompt_group_ids)

        group0 = weights[prompt_group_ids == 0].sum().item()
        group1 = weights[prompt_group_ids == 1].sum().item()
        self.assertAlmostEqual(group0, 1.0, places=6)
        self.assertAlmostEqual(group1, 1.0, places=6)

    def test_soft_nan_entropy_falls_back_uniform(self):
        trainer = self._new_trainer(mode='soft', scope='global')
        entropies = torch.tensor([[float('nan'), float('nan'), float('nan')]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

        weights, _ = trainer._build_mapo_entropy_weights(entropies, completion_mask)

        expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.float32)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
