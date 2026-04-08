import unittest
from unittest.mock import patch

import torch


class _DummyTokenizer:

    def __init__(self):
        self._mapping = {
            1: ' dog',
            2: ' runs',
            3: ' quickly',
        }

    def convert_ids_to_tokens(self, token_ids):
        return [self._mapping.get(int(token_id), ' token') for token_id in token_ids]


class TestMAPOPhase2Weights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.megatron.trainers.mapo_trainer import MegatronMAPOTrainer
            from swift.megatron.trainers import mapo_pos_utils
            cls.trainer_cls = MegatronMAPOTrainer
            cls.mapo_pos_utils = mapo_pos_utils
            cls.available = True
        except Exception as e:  # noqa: BLE001
            print(f'Warning: MAPO trainer import unavailable: {e}')
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest('MegatronMAPOTrainer unavailable in this environment')

    def _new_trainer(self, mask_temperature=1.0, temporal_kappa=1.0, loss_type='grpo'):
        trainer = self.trainer_cls.__new__(self.trainer_cls)
        trainer.mapo_mask_temperature = mask_temperature
        trainer.mapo_temporal_kappa = temporal_kappa
        trainer.loss_type = loss_type
        return trainer

    def test_omega_and_nu_normalization(self):
        trainer = self._new_trainer(mask_temperature=0.7)
        delta_h_abs = torch.tensor([[0.1, 0.2, 0.8, 0.0], [0.4, 0.6, 0.0, 0.0]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

        omega_tilde, nu_tilde, _ = trainer._build_mapo_relevance_weights(delta_h_abs, completion_mask)

        token_count = completion_mask.float().sum(-1)
        omega_sum = (omega_tilde * completion_mask.float()).sum(-1)
        nu_sum = (nu_tilde * completion_mask.float()).sum(-1)
        self.assertTrue(torch.allclose(omega_sum, token_count, atol=1e-6))
        self.assertTrue(torch.allclose(nu_sum, token_count, atol=1e-6))

    def test_high_delta_h_gets_larger_omega_and_smaller_nu(self):
        trainer = self._new_trainer(mask_temperature=0.5)
        delta_h_abs = torch.tensor([[0.1, 2.0, 0.2, 0.0]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        omega_tilde, nu_tilde, _ = trainer._build_mapo_relevance_weights(delta_h_abs, completion_mask)

        self.assertGreater(omega_tilde[0, 1].item(), omega_tilde[0, 0].item())
        self.assertGreater(omega_tilde[0, 1].item(), omega_tilde[0, 2].item())
        self.assertLess(nu_tilde[0, 1].item(), nu_tilde[0, 0].item())
        self.assertLess(nu_tilde[0, 1].item(), nu_tilde[0, 2].item())

    def test_task_failed_zero_disables_attention_loss(self):
        trainer = self._new_trainer(loss_type='grpo')
        completion_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        nu_tilde = torch.ones((1, 3), dtype=torch.float32)
        temporal = trainer._build_temporal_weights(completion_mask)
        pos_gate = torch.ones((1, 3), dtype=torch.float32)
        audio_mass = torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float32)
        task_failed = torch.zeros((1, ), dtype=torch.float32)
        advantages = torch.tensor([1.5], dtype=torch.float32)

        per_token_obj, attn_loss = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass,
            completion_mask=completion_mask,
            nu_tilde=nu_tilde,
            temporal_weights=temporal,
            pos_gate=pos_gate,
            task_failed=task_failed,
            advantages=advantages)

        self.assertTrue(torch.allclose(per_token_obj, torch.zeros_like(per_token_obj)))
        self.assertAlmostEqual(attn_loss.item(), 0.0, places=6)

    def test_pos_gate_projection_and_fallback(self):
        tokenizer = _DummyTokenizer()

        with patch('swift.megatron.trainers.mapo_pos_utils._pos_tag_words',
                   return_value=[('dog', 'NN'), ('runs', 'VBZ'), ('quickly', 'RB')]):
            gate, fallback = self.mapo_pos_utils.build_pos_token_gate(
                token_ids=[1, 2, 3], tokenizer=tokenizer, target_pos_tags=['NOUN', 'VERB'])

        self.assertFalse(fallback)
        self.assertEqual(gate, [1.0, 1.0, 0.0])

        with patch('swift.megatron.trainers.mapo_pos_utils._pos_tag_words', side_effect=RuntimeError('tagger failed')):
            gate, fallback = self.mapo_pos_utils.build_pos_token_gate(
                token_ids=[1, 2, 3], tokenizer=tokenizer, target_pos_tags=['NOUN', 'VERB'])

        self.assertTrue(fallback)
        self.assertEqual(gate, [1.0, 1.0, 1.0])

    def test_attention_loss_sign(self):
        trainer = self._new_trainer(loss_type='grpo')
        completion_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        nu_tilde = torch.ones((1, 3), dtype=torch.float32)
        temporal = trainer._build_temporal_weights(completion_mask)
        pos_gate = torch.ones((1, 3), dtype=torch.float32)
        task_failed = torch.ones((1, ), dtype=torch.float32)
        advantages = torch.tensor([1.0], dtype=torch.float32)

        _, low_mass_loss = trainer._compute_mapo_attention_objective(
            audio_mass=torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32),
            completion_mask=completion_mask,
            nu_tilde=nu_tilde,
            temporal_weights=temporal,
            pos_gate=pos_gate,
            task_failed=task_failed,
            advantages=advantages)
        _, high_mass_loss = trainer._compute_mapo_attention_objective(
            audio_mass=torch.tensor([[0.8, 0.8, 0.8]], dtype=torch.float32),
            completion_mask=completion_mask,
            nu_tilde=nu_tilde,
            temporal_weights=temporal,
            pos_gate=pos_gate,
            task_failed=task_failed,
            advantages=advantages)

        self.assertLess(high_mass_loss.item(), low_mass_loss.item())

    def test_reduce_token_mean_applies_pos_gate_with_grpo_normalization(self):
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        token_values = torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.8, 0.0, 0.0]], dtype=torch.float32)
        pos_gate = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        reduced = self.trainer_cls._reduce_token_mean(
            token_values, completion_mask, loss_type='grpo', gate_weights=pos_gate)

        # GRPO-style reduction averages per-sequence gated means: ((0.1 + 0.3) / 2 + 0.8) / 2.
        self.assertAlmostEqual(reduced.item(), 0.5, places=6)

    def test_reduce_token_mean_applies_pos_gate_with_bnpo_normalization(self):
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        token_values = torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.8, 0.0, 0.0]], dtype=torch.float32)
        pos_gate = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        reduced = self.trainer_cls._reduce_token_mean(
            token_values, completion_mask, loss_type='bnpo', gate_weights=pos_gate)

        # BNPO-style reduction uses a global token mean over the gated tokens.
        self.assertAlmostEqual(reduced.item(), 0.4, places=6)


if __name__ == '__main__':
    unittest.main()
