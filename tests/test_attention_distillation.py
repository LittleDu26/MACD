import unittest

import torch

from macd.controller_distillation import (
    DISTILL_LOSS_TYPES,
    _column_importance,
    _shared_attention_kl,
    _shared_branch_loss,
    _shared_feature_loss,
    _validate_distillation_config,
)
from macd.transformer.config import transformerconfig
from macd.transformer.daab_model import DAABTransformerEncoder
from macd.transformer.transformer import (
    TransformerEncoder,
    TransformerEncoderLayerResidual,
)


class AttentionDistillationLossTest(unittest.TestCase):
    def test_default_config_uses_detailed_objective(self):
        config = transformerconfig()
        self.assertEqual(config.attention_distill_loss, "shared_attention_feature")
        self.assertEqual(config.attention_distill_lambda_a, 1.0)
        self.assertEqual(config.attention_distill_lambda_h, 1.0)

    def test_config_validation_rejects_unknown_mode_and_negative_weights(self):
        with self.assertRaises(ValueError):
            _validate_distillation_config("unknown", 1.0, 1.0)
        with self.assertRaises(ValueError):
            _validate_distillation_config("shared_attention_feature", -1.0, 1.0)

    def test_validate_accepts_all_four_loss_modes(self):
        for loss_type in DISTILL_LOSS_TYPES:
            _validate_distillation_config(loss_type, 1.0, 1.0)

    def test_distill_loss_types_constant(self):
        self.assertEqual(
            set(DISTILL_LOSS_TYPES),
            {
                "column_kl",
                "shared_attention",
                "shared_feature",
                "shared_attention_feature",
            },
        )

    def test_shared_branch_component_splitting(self):
        parent_attn = torch.tensor(
            [[[[0.8, 0.2], [0.1, 0.9]], [[0.3, 0.7], [0.6, 0.4]]]],
            dtype=torch.float32,
        )
        child_attn = torch.tensor(
            [[[[0.2, 0.8], [0.9, 0.1]], [[0.7, 0.3], [0.4, 0.6]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        parent_hidden = torch.tensor(
            [
                [[1.0, 2.0, 4.0]],
                [[3.0, 2.0, 1.0]],
                [[2.0, 5.0, 1.0]],
            ]
        )
        child_hidden = parent_hidden.clone()
        child_hidden[0, 0, 0] += 1.0
        child_hidden.requires_grad_(True)
        indices = torch.tensor([0, 1])

        attn_only = _shared_attention_kl(
            [parent_attn], [child_attn], indices, indices
        )
        feat_only = _shared_feature_loss(
            [parent_hidden], [child_hidden], indices, indices
        )
        self.assertGreater(attn_only.item(), 0.0)
        self.assertGreater(feat_only.item(), 0.0)

        combined = _shared_branch_loss(
            [parent_attn],
            [child_attn],
            [parent_hidden],
            [child_hidden],
            indices,
            indices,
            "shared_attention_feature",
            1.0,
            1.0,
        )
        attn_branch = _shared_branch_loss(
            [parent_attn],
            [child_attn],
            [parent_hidden],
            [child_hidden],
            indices,
            indices,
            "shared_attention",
            1.0,
            0.0,
        )
        feat_branch = _shared_branch_loss(
            [parent_attn],
            [child_attn],
            [parent_hidden],
            [child_hidden],
            indices,
            indices,
            "shared_feature",
            0.0,
            1.0,
        )

        self.assertAlmostEqual(attn_branch.item(), attn_only.item(), places=5)
        self.assertAlmostEqual(feat_branch.item(), feat_only.item(), places=5)
        self.assertAlmostEqual(
            combined.item(), attn_only.item() + feat_only.item(), places=5
        )

        scaled_attn = _shared_branch_loss(
            [parent_attn],
            [child_attn],
            [parent_hidden],
            [child_hidden],
            indices,
            indices,
            "shared_attention",
            0.5,
            2.0,
        )
        self.assertAlmostEqual(
            scaled_attn.item(), 0.5 * attn_only.item(), places=5
        )

    def test_shared_attention_kl_is_per_head_and_per_query(self):
        parent = torch.tensor(
            [[[[0.8, 0.2], [0.1, 0.9]], [[0.3, 0.7], [0.6, 0.4]]]],
            dtype=torch.float32,
        )
        child = parent.clone().requires_grad_(True)
        indices = torch.tensor([0, 1])
        equal_loss = _shared_attention_kl([parent], [child], indices, indices)
        self.assertAlmostEqual(equal_loss.item(), 0.0, places=6)

        changed = torch.tensor(
            [[[[0.2, 0.8], [0.9, 0.1]], [[0.7, 0.3], [0.4, 0.6]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss = _shared_attention_kl([parent], [changed], indices, indices)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(changed.grad)

    def test_shared_feature_loss_maps_parent_and_child_token_orders(self):
        parent = torch.tensor(
            [
                [[1.0, 2.0, 4.0]],
                [[3.0, 2.0, 1.0]],
                [[2.0, 5.0, 1.0]],
            ]
        )
        child = parent.index_select(0, torch.tensor([2, 0, 1])).clone()
        parent_indices = torch.tensor([0, 2])
        child_indices = torch.tensor([1, 0])
        child.requires_grad_(True)
        equal_loss = _shared_feature_loss(
            [parent], [child], parent_indices, child_indices
        )
        self.assertAlmostEqual(equal_loss.item(), 0.0, places=6)

        changed = child.detach().clone()
        changed[0, 0, 0] += 1.0
        changed.requires_grad_(True)
        loss = _shared_feature_loss(
            [parent], [changed], parent_indices, child_indices
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(changed.grad)

    def test_column_importance_still_accepts_per_head_attention(self):
        attention = [torch.rand(2, 3, 4, 4)]
        scores = _column_importance(attention, torch.tensor([0, 2]))
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.allclose(scores.sum(dim=-1), torch.ones(2)))


class DistillationForwardOutputTest(unittest.TestCase):
    def test_original_encoder_returns_each_layers_attention_and_hidden_state(self):
        embedding_dim = 8
        layer = TransformerEncoderLayerResidual(embedding_dim, 2, 16, 0.0)
        encoder = TransformerEncoder(layer, 2)
        output, attention, hidden = encoder(
            torch.randn(4, 3, embedding_dim),
            return_attn=True,
            return_hidden_states=True,
        )
        self.assertEqual(tuple(output.shape), (4, 3, embedding_dim))
        self.assertEqual(len(attention), 2)
        self.assertEqual(tuple(attention[0].shape), (3, 2, 4, 4))
        self.assertEqual(len(hidden), 2)

    def test_daab_encoder_preserves_attention_heads_and_hidden_states(self):
        encoder = DAABTransformerEncoder(
            embedding_dim=8, heads=2, hidden_size=16, layers=2, dropout=0.0
        )
        source = torch.randn(4, 3, 8)
        bias = torch.zeros(3, 2, 4, 4)
        normal_output, normal_attention = encoder(
            source, bias, return_attention=True
        )
        self.assertEqual(tuple(normal_output.shape), (4, 3, 8))
        self.assertEqual(len(normal_attention), 2)

        output, attention, hidden = encoder(
            source,
            bias,
            return_attention=True,
            return_hidden_states=True,
        )
        self.assertEqual(tuple(output.shape), (4, 3, 8))
        self.assertEqual(len(attention), 2)
        self.assertEqual(tuple(attention[0].shape), (3, 2, 4, 4))
        self.assertEqual(len(hidden), 2)


if __name__ == "__main__":
    unittest.main()
