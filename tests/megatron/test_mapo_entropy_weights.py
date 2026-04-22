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

    def _new_trainer(self, mask_temperature=1.0, temporal_kappa=1.0, loss_type='grpo', mask_clip=6.0):
        trainer = self.trainer_cls.__new__(self.trainer_cls)
        trainer.mapo_mask_temperature = mask_temperature
        trainer.mapo_mask_clip = mask_clip
        trainer.mapo_temporal_kappa = temporal_kappa
        trainer.loss_type = loss_type
        return trainer

    def test_omega_and_nu_normalization(self):
        trainer = self._new_trainer(mask_temperature=0.7, mask_clip=0.0)  # no clipping → invariant holds
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

    def test_mask_clip_bounds_weights(self):
        """Clip=2.0 with low temperature should produce weights capped at 2.0."""
        trainer = self._new_trainer(mask_temperature=0.01, mask_clip=2.0)
        # Extreme delta_h_abs to force concentration on index 3.
        delta_h_abs = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool)

        omega_tilde, nu_tilde, _ = trainer._build_mapo_relevance_weights(delta_h_abs, completion_mask)

        self.assertLessEqual(omega_tilde.max().item(), 2.0 + 1e-5)
        self.assertLessEqual(nu_tilde.max().item(), 2.0 + 1e-5)

    def test_mask_clip_disabled_when_zero(self):
        """Clip=0 should not modify weights (sum-to-token_count invariant holds)."""
        trainer = self._new_trainer(mask_temperature=0.7, mask_clip=0.0)
        delta_h_abs = torch.tensor([[0.1, 0.2, 0.8, 0.0], [0.4, 0.6, 0.0, 0.0]], dtype=torch.float32)
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

        omega_tilde, nu_tilde, _ = trainer._build_mapo_relevance_weights(delta_h_abs, completion_mask)

        token_count = completion_mask.float().sum(-1)
        omega_sum = (omega_tilde * completion_mask.float()).sum(-1)
        nu_sum = (nu_tilde * completion_mask.float()).sum(-1)
        self.assertTrue(torch.allclose(omega_sum, token_count, atol=1e-6))
        self.assertTrue(torch.allclose(nu_sum, token_count, atol=1e-6))

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

    # ------------------------------------------------------------------
    # Regression tests for _compute_mapo_attention_objective fix
    # (v72 regression: sum-reduction + missing pos_gate → length-dependent
    #  loss that caused training collapse).
    # ------------------------------------------------------------------

    def _make_simple_trainer(self, loss_type='grpo', task_fail_floor=0.0, prefactor_clip=0.0):
        """Build a minimal trainer instance with only the attributes required by
        _compute_mapo_attention_objective and _build_temporal_weights."""
        trainer = self.trainer_cls.__new__(self.trainer_cls)
        trainer.loss_type = loss_type
        trainer.mapo_task_fail_gate_floor = task_fail_floor
        trainer.mapo_attn_prefactor_clip = prefactor_clip
        trainer.mapo_mask_temperature = 1.0
        trainer.mapo_mask_clip = 6.0
        trainer.mapo_temporal_kappa = 1.0
        return trainer

    def test_attn_loss_equals_pos_gated_mean_grpo(self):
        """With pos_gate=all-ones and unit weights, attn_loss must equal the
        GRPO per-sequence mean of (gate * temporal * nu * -log(audio_mass))."""
        trainer = self._make_simple_trainer(loss_type='grpo')
        B, T = 2, 4
        # Lengths: seq0 has 3 active tokens, seq1 has 2.
        completion_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        pos_gate = torch.ones(B, T)
        temporal = trainer._build_temporal_weights(completion_mask)
        nu_tilde = torch.ones(B, T)
        audio_mass = torch.tensor([[0.3, 0.5, 0.8, 0.0], [0.2, 0.6, 0.0, 0.0]])
        task_failed = torch.ones(B)
        advantages_abs = torch.tensor([1.0, 1.0])

        per_token_obj, attn_loss = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass,
            completion_mask=completion_mask,
            nu_tilde=nu_tilde,
            task_failed=task_failed,
            temporal_weights=temporal,
            pos_gate=pos_gate,
            advantages_abs=advantages_abs)

        # Reference: compute expected GRPO mean manually.
        log_penalty = -torch.log((audio_mass + 1e-6).clamp(min=1e-6))
        expected_per_token = temporal * nu_tilde * pos_gate * log_penalty
        mask_f = completion_mask.float()
        expected_attn_loss = ((expected_per_token * mask_f).sum(-1) /
                              (mask_f.sum(-1).clamp(min=1.0))).mean()

        self.assertAlmostEqual(attn_loss.item(), expected_attn_loss.item(), places=5)

    def test_attn_loss_length_invariant_under_padding(self):
        """attn_loss must not change when extra masked-out padding tokens are appended.

        This is the key property that v72's .sum(-1).mean() violated:
        a longer completion of all-zeros mask tokens must give the same loss.
        """
        trainer = self._make_simple_trainer(loss_type='grpo')
        completion_mask_short = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        completion_mask_long = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        pos_gate_short = torch.ones(1, 3)
        pos_gate_long = torch.ones(1, 6)
        # Audio mass: same values, padded with zeros (zeros outside mask are irrelevant).
        audio_mass_short = torch.tensor([[0.4, 0.6, 0.2]])
        audio_mass_long = torch.tensor([[0.4, 0.6, 0.2, 0.0, 0.0, 0.0]])
        nu_short = torch.ones(1, 3)
        nu_long = torch.ones(1, 6)
        task_failed = torch.ones(1)
        advantages_abs = torch.tensor([1.0])

        temporal_short = trainer._build_temporal_weights(completion_mask_short)
        temporal_long = trainer._build_temporal_weights(completion_mask_long)

        _, loss_short = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass_short,
            completion_mask=completion_mask_short,
            nu_tilde=nu_short,
            task_failed=task_failed,
            temporal_weights=temporal_short,
            pos_gate=pos_gate_short,
            advantages_abs=advantages_abs)

        _, loss_long = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass_long,
            completion_mask=completion_mask_long,
            nu_tilde=nu_long,
            task_failed=task_failed,
            temporal_weights=temporal_long,
            pos_gate=pos_gate_long,
            advantages_abs=advantages_abs)

        self.assertAlmostEqual(loss_short.item(), loss_long.item(), places=5)

    def test_attn_loss_pos_gate_excludes_masked_tokens_from_denom(self):
        """Tokens with pos_gate=0 must be excluded from both numerator and denominator.

        Concretely: a 3-token completion where the middle token has pos_gate=0
        must give the same attn_loss as a 2-token completion with both tokens
        pos-gated on and identical audio_mass / temporal values.
        """
        trainer = self._make_simple_trainer(loss_type='grpo')
        task_failed = torch.ones(1)
        advantages_abs = torch.tensor([1.0])

        # 3-token completion; middle token masked out by pos_gate.
        completion_mask_3 = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        pos_gate_3 = torch.tensor([[1.0, 0.0, 1.0]])
        audio_mass_3 = torch.tensor([[0.3, 0.9, 0.5]])  # index 1 irrelevant
        nu_3 = torch.ones(1, 3)
        temporal_3 = trainer._build_temporal_weights(completion_mask_3)

        # 2-token completion with only the two pos-active positions.
        # Temporal weights differ because T changes, so we need to craft them manually
        # to match what the 3-token completion produces for positions 0 and 2.
        completion_mask_2 = torch.tensor([[1, 1]], dtype=torch.bool)
        pos_gate_2 = torch.ones(1, 2)
        audio_mass_2 = torch.tensor([[0.3, 0.5]])
        nu_2 = torch.ones(1, 2)
        # Mirror temporal weights: for seq of T=3, pos 0 gives (1/3)^1, pos 2 gives (3/3)^1.
        temporal_2 = torch.tensor([[1.0 / 3.0, 1.0]])

        _, loss_3 = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass_3,
            completion_mask=completion_mask_3,
            nu_tilde=nu_3,
            task_failed=task_failed,
            temporal_weights=temporal_3,
            pos_gate=pos_gate_3,
            advantages_abs=advantages_abs)

        _, loss_2 = trainer._compute_mapo_attention_objective(
            audio_mass=audio_mass_2,
            completion_mask=completion_mask_2,
            nu_tilde=nu_2,
            task_failed=task_failed,
            temporal_weights=temporal_2,
            pos_gate=pos_gate_2,
            advantages_abs=advantages_abs)

        self.assertAlmostEqual(loss_3.item(), loss_2.item(), places=5)


if __name__ == '__main__':
    unittest.main()
