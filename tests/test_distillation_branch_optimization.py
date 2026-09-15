"""Tests for independent actor/critic distillation warm-up updates."""

import copy
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

import macd.controller_distillation as distillation


class _AttentionParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)


class _Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttentionParameters()
        self.norm = nn.LayerNorm(2)
        self.frozen_projection = nn.Linear(2, 2)


class _ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_net = _Branch()
        self.v_net = _Branch()


class _Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.ac = _ActorCritic()


def _distillation_parameter_ids(module):
    return {
        id(param)
        for name, param in module.named_parameters()
        if "self_attn.q_proj." in name
        or "self_attn.k_proj." in name
        or "norm" in name
    }


def _parameter_loss(module, multiplier=1.0):
    terms = [
        param.pow(2).sum()
        for param in module.parameters()
        if param.requires_grad
    ]
    return multiplier * torch.stack(terms).sum()


class DistillationBranchOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.parent = _Controller()
        self.child = _Controller()
        self.parent_robot = np.ones((2, 2), dtype=int)
        self.child_robot = np.ones((2, 2), dtype=int)
        self.indices = torch.arange(4, dtype=torch.long)

    def _run_warmup(self, loss_side_effect, clip_side_effect=None,
                    optimizer_side_effect=None):
        patches = [
            mock.patch.object(distillation, "_stack_obs", return_value={}),
            mock.patch.object(
                distillation,
                "_build_distill_views",
                return_value=({}, {}, self.indices, self.indices),
            ),
            mock.patch.object(
                distillation,
                "_distill_branch_loss",
                side_effect=loss_side_effect,
            ),
        ]
        if clip_side_effect is not None:
            patches.append(
                mock.patch.object(
                    distillation.nn.utils,
                    "clip_grad_norm_",
                    side_effect=clip_side_effect,
                )
            )
        if optimizer_side_effect is not None:
            patches.append(
                mock.patch.object(
                    distillation.optim,
                    "Adam",
                    side_effect=optimizer_side_effect,
                )
            )

        with patches[0], patches[1], patches[2]:
            if len(patches) == 3:
                distillation.attention_distill_warmup(
                    self.parent,
                    self.child,
                    [{}],
                    self.parent_robot,
                    self.child_robot,
                    torch.device("cpu"),
                    batch_size=1,
                    warmup_epochs=1,
                )
            elif len(patches) == 4:
                with patches[3]:
                    distillation.attention_distill_warmup(
                        self.parent,
                        self.child,
                        [{}],
                        self.parent_robot,
                        self.child_robot,
                        torch.device("cpu"),
                        batch_size=1,
                        warmup_epochs=1,
                    )
            else:
                with patches[3], patches[4]:
                    distillation.attention_distill_warmup(
                        self.parent,
                        self.child,
                        [{}],
                        self.parent_robot,
                        self.child_robot,
                        torch.device("cpu"),
                        batch_size=1,
                        warmup_epochs=1,
                    )

    def test_actor_and_critic_use_disjoint_optimizers_and_clipping(self):
        actor_ids = _distillation_parameter_ids(self.child.ac.mu_net)
        critic_ids = _distillation_parameter_ids(self.child.ac.v_net)
        optimizer_groups = []
        clipped_groups = []
        real_adam = torch.optim.Adam
        real_clip = torch.nn.utils.clip_grad_norm_

        def make_optimizer(params, lr):
            params = list(params)
            optimizer_groups.append({id(param) for param in params})
            return real_adam(params, lr=lr)

        def clip_branch(params, max_norm):
            params = list(params)
            clipped_groups.append({id(param) for param in params})
            return real_clip(params, max_norm)

        def branch_loss(_parent, child, _parent_batch, _child_batch,
                        _parent_indices, _child_indices, net_name, *_args):
            module = child.ac.mu_net if net_name == "mu" else child.ac.v_net
            return _parameter_loss(module)

        frozen_before = {
            "mu": copy.deepcopy(self.child.ac.mu_net.frozen_projection.state_dict()),
            "v": copy.deepcopy(self.child.ac.v_net.frozen_projection.state_dict()),
        }
        trainable_before = {
            "mu": copy.deepcopy(self.child.ac.mu_net.state_dict()),
            "v": copy.deepcopy(self.child.ac.v_net.state_dict()),
        }
        self._run_warmup(branch_loss, clip_branch, make_optimizer)

        self.assertEqual(optimizer_groups, [actor_ids, critic_ids])
        self.assertEqual(clipped_groups, [actor_ids, critic_ids])
        self.assertFalse(actor_ids & critic_ids)
        for branch_name, module in (
            ("mu", self.child.ac.mu_net),
            ("v", self.child.ac.v_net),
        ):
            self.assertTrue(
                any(
                    not torch.equal(value, trainable_before[branch_name][name])
                    for name, value in module.state_dict().items()
                    if "self_attn.q_proj." in name
                    or "self_attn.k_proj." in name
                    or "norm" in name
                )
            )
        for name, value in self.child.ac.mu_net.frozen_projection.state_dict().items():
            self.assertTrue(torch.equal(value, frozen_before["mu"][name]))
        for name, value in self.child.ac.v_net.frozen_projection.state_dict().items():
            self.assertTrue(torch.equal(value, frozen_before["v"][name]))
        self.assertTrue(all(param.requires_grad for param in self.child.parameters()))

    def test_zero_actor_loss_does_not_prevent_critic_update(self):
        actor_before = copy.deepcopy(self.child.ac.mu_net.state_dict())
        critic_before = copy.deepcopy(self.child.ac.v_net.state_dict())

        def branch_loss(_parent, child, _parent_batch, _child_batch,
                        _parent_indices, _child_indices, net_name, *_args):
            module = child.ac.mu_net if net_name == "mu" else child.ac.v_net
            multiplier = 0.0 if net_name == "mu" else 1.0
            return _parameter_loss(module, multiplier)

        self._run_warmup(branch_loss)

        for name, value in self.child.ac.mu_net.state_dict().items():
            self.assertTrue(torch.equal(value, actor_before[name]))
        self.assertTrue(
            any(
                not torch.equal(value, critic_before[name])
                for name, value in self.child.ac.v_net.state_dict().items()
                if "self_attn.q_proj." in name
                or "self_attn.k_proj." in name
                or "norm" in name
            )
        )

    def test_child_gradients_are_restored_after_branch_failure(self):
        def branch_loss(*_args):
            raise RuntimeError("synthetic branch failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic branch failure"):
            self._run_warmup(branch_loss)

        self.assertTrue(all(param.requires_grad for param in self.child.parameters()))


if __name__ == "__main__":
    unittest.main()
