"""Tests for MAPOAttentionCollector head_reduce / layer_reduce behaviour."""
import unittest

import torch

from swift.megatron.trainers.mapo_attention_collector import (
    MAPOAttentionCollector,
    parse_attention_layer_spec,
)


class TestParseAttentionLayerSpec(unittest.TestCase):

    def test_single_layer(self):
        self.assertEqual(parse_attention_layer_spec('47'), {47})

    def test_multi_layer_comma_separated(self):
        self.assertEqual(parse_attention_layer_spec('45,46,47'), {45, 46, 47})

    def test_all_keyword(self):
        self.assertIsNone(parse_attention_layer_spec('all'))
        self.assertIsNone(parse_attention_layer_spec('*'))

    def test_empty_spec(self):
        self.assertEqual(parse_attention_layer_spec(''), set())

    def test_whitespace_handling(self):
        self.assertEqual(parse_attention_layer_spec(' 42 , 45 , 47 '), {42, 45, 47})

    def test_invalid_token_raises(self):
        with self.assertRaises(ValueError):
            parse_attention_layer_spec('abc')


class TestCollectorHeadReduce(unittest.TestCase):
    """Test inter-head aggregation strategies."""

    def _make_collector_with_data(self, head_reduce='max', layer_reduce='mean'):
        """Create a collector and inject a synthetic attention tensor."""
        collector = MAPOAttentionCollector(
            attention_layers={0}, head_reduce=head_reduce, layer_reduce=layer_reduce)
        # Synthetic attn: [batch=2, heads=4, query=3, key=6]
        attn = torch.zeros(2, 4, 3, 6, dtype=torch.float32)
        # Head 0: strong audio attention on key positions 0,1
        attn[:, 0, :, 0:2] = 0.8
        # Head 1: weak audio attention
        attn[:, 1, :, 0:2] = 0.1
        # Head 2: medium audio attention
        attn[:, 2, :, 0:2] = 0.4
        # Head 3: near-zero
        attn[:, 3, :, 0:2] = 0.01
        collector._layer_attentions[0] = attn
        collector._num_hook_modules = 1
        collector._num_tensor_updates = 1
        return collector

    def test_head_max_selects_strongest_head(self):
        collector = self._make_collector_with_data(head_reduce='max')
        audio_mask = torch.zeros(2, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True  # first 2 positions are audio
        completion_mask = torch.ones(2, 3, dtype=torch.bool)

        audio_mass, diag = collector.compute_audio_attention_mass(
            audio_mask, completion_mask, tp_world_size=1)

        self.assertIsNotNone(audio_mass)
        # Max head has mass 0.8 * 2 = 1.6 per query token
        expected_max = 0.8 * 2  # sum of audio key positions for strongest head
        self.assertAlmostEqual(audio_mass[0, 0].item(), expected_max, places=4)

    def test_head_mean_averages_all_heads(self):
        collector = self._make_collector_with_data(head_reduce='mean')
        audio_mask = torch.zeros(2, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(2, 3, dtype=torch.bool)

        audio_mass, diag = collector.compute_audio_attention_mass(
            audio_mask, completion_mask, tp_world_size=1)

        self.assertIsNotNone(audio_mass)
        # Mean across 4 heads: (0.8*2 + 0.1*2 + 0.4*2 + 0.01*2) / 4 = (1.6+0.2+0.8+0.02)/4
        expected_mean = (0.8 + 0.1 + 0.4 + 0.01) * 2 / 4
        self.assertAlmostEqual(audio_mass[0, 0].item(), expected_mean, places=4)

    def test_head_max_greater_than_mean(self):
        collector_max = self._make_collector_with_data(head_reduce='max')
        collector_mean = self._make_collector_with_data(head_reduce='mean')
        audio_mask = torch.zeros(2, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(2, 3, dtype=torch.bool)

        mass_max, _ = collector_max.compute_audio_attention_mass(audio_mask, completion_mask)
        mass_mean, _ = collector_mean.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertTrue((mass_max >= mass_mean).all(),
                        'Max head reduce should always >= mean head reduce')


class TestCollectorLayerReduce(unittest.TestCase):
    """Test inter-layer aggregation strategies."""

    def _make_collector_with_multi_layer_data(self, layer_reduce='mean'):
        collector = MAPOAttentionCollector(
            attention_layers={0, 1, 2}, head_reduce='mean', layer_reduce=layer_reduce)
        # 3 layers with different attention patterns
        audio_mask_indices = slice(0, 2)
        for layer_idx, audio_strength in enumerate([0.2, 0.5, 0.9]):
            attn = torch.zeros(1, 2, 4, 6, dtype=torch.float32)
            attn[:, :, :, audio_mask_indices] = audio_strength
            collector._layer_attentions[layer_idx] = attn
        collector._num_hook_modules = 3
        collector._num_tensor_updates = 3
        return collector

    def test_layer_mean(self):
        collector = self._make_collector_with_multi_layer_data(layer_reduce='mean')
        audio_mask = torch.zeros(1, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(1, 4, dtype=torch.bool)

        mass, diag = collector.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertEqual(diag['mapo_attn_layers_used'], 3.0)
        # Mean of per-layer masses: (0.2*2 + 0.5*2 + 0.9*2) / (3 * num_heads)
        # With mean head reduce and tp=1: per_layer_mass = strength*2 / 2 (mean over 2 heads)
        # Layer masses: 0.4/2=0.2, 1.0/2=0.5, 1.8/2=0.9 -> mean = (0.2+0.5+0.9)/3
        expected = (0.2 * 2 + 0.5 * 2 + 0.9 * 2) / (2 * 3)  # mean heads, mean layers
        # Actually: each layer mass = strength * 2 / 2 = strength (mean over heads gives /2)
        # Then mean over 3 layers: (0.2 + 0.5 + 0.9) / 3
        expected = (0.2 + 0.5 + 0.9) / 3
        self.assertAlmostEqual(mass[0, 0].item(), expected, places=4)

    def test_layer_max(self):
        collector = self._make_collector_with_multi_layer_data(layer_reduce='max')
        audio_mask = torch.zeros(1, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(1, 4, dtype=torch.bool)

        mass, diag = collector.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertEqual(diag['mapo_attn_layers_used'], 3.0)
        # Max of per-layer masses: max(0.2, 0.5, 0.9) = 0.9
        expected = 0.9  # strongest layer's mean-head mass
        self.assertAlmostEqual(mass[0, 0].item(), expected, places=4)

    def test_layer_max_greater_than_mean(self):
        collector_max = self._make_collector_with_multi_layer_data(layer_reduce='max')
        collector_mean = self._make_collector_with_multi_layer_data(layer_reduce='mean')
        audio_mask = torch.zeros(1, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(1, 4, dtype=torch.bool)

        mass_max, _ = collector_max.compute_audio_attention_mass(audio_mask, completion_mask)
        mass_mean, _ = collector_mean.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertTrue((mass_max >= mass_mean).all())


class TestCollectorMissingLayers(unittest.TestCase):
    """Test behavior when some requested layers don't produce data."""

    def test_partial_layer_capture(self):
        """Request layers {0, 1, 2} but only layers 0 and 2 produce attention data."""
        collector = MAPOAttentionCollector(
            attention_layers={0, 1, 2}, head_reduce='max', layer_reduce='mean')
        # Only inject layers 0 and 2, skip layer 1
        for layer_idx in [0, 2]:
            attn = torch.full((1, 2, 4, 6), 0.5)
            collector._layer_attentions[layer_idx] = attn
        collector._num_hook_modules = 3
        collector._num_tensor_updates = 2

        audio_mask = torch.zeros(1, 6, dtype=torch.bool)
        audio_mask[:, 0:2] = True
        completion_mask = torch.ones(1, 4, dtype=torch.bool)

        mass, diag = collector.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertIsNotNone(mass)
        self.assertEqual(diag['mapo_attn_layers_used'], 2.0)  # not 3
        self.assertEqual(diag['mapo_attn_hook_modules'], 3.0)

    def test_non_contiguous_layer_indices(self):
        """Layers {2, 5, 7} should work fine (non-contiguous)."""
        collector = MAPOAttentionCollector(
            attention_layers={2, 5, 7}, head_reduce='mean', layer_reduce='mean')
        for layer_idx in [2, 5, 7]:
            attn = torch.full((1, 2, 4, 6), 0.3)
            collector._layer_attentions[layer_idx] = attn
        collector._num_hook_modules = 3
        collector._num_tensor_updates = 3

        audio_mask = torch.zeros(1, 6, dtype=torch.bool)
        audio_mask[:, 0:3] = True
        completion_mask = torch.ones(1, 4, dtype=torch.bool)

        mass, diag = collector.compute_audio_attention_mass(audio_mask, completion_mask)

        self.assertIsNotNone(mass)
        self.assertEqual(diag['mapo_attn_layers_used'], 3.0)
        self.assertEqual(diag['mapo_attn_available'], 1.0)


if __name__ == '__main__':
    unittest.main()
