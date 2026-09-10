"""Fast checks for the population-level MACD ablation configuration."""

import argparse
import importlib.util
import os
import unittest

import torch

from macd.controller_distillation import _restore_distillation_parameters_from_initial
from macd.run import controller_init_mode, inherits_parent_controller


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "macd_ablation", os.path.join(ROOT_DIR, "MACD-ablation.py")
)
macd_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(macd_ablation)


class MacdAblationConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.base_args = argparse.Namespace(
            ablations=tuple(macd_ablation.ABLATIONS),
            envs=["Thrower-v0"],
            num_runs=1,
            seed=101,
            target_size=5,
            threads_num=1,
            pop_size=4,
            train_iters=7,
            promotion_k=None,
            lambda_max=0.30,
            lambda_min=0.05,
            lambda_tau=2.0,
            selection_eps=1e-8,
            output_root=os.path.join(ROOT_DIR, "result", "test-ablation"),
        )

    def test_controller_modes_have_expected_inheritance(self):
        self.assertEqual(controller_init_mode(argparse.Namespace(distill=True)), "distill")
        self.assertEqual(controller_init_mode(argparse.Namespace(distill=False)), "random_init")
        self.assertFalse(inherits_parent_controller("random_init"))
        for mode in ("distill", "non_distill_copy", "full_copy"):
            self.assertTrue(inherits_parent_controller(mode))

    def test_no_m_uses_a_full_evaluation_stage(self):
        no_m = macd_ablation.configure_run(
            self.base_args, "MACD-noM", "Thrower-v0", 0
        )
        no_i = macd_ablation.configure_run(
            self.base_args, "MACD-noI", "Thrower-v0", 0
        )
        self.assertFalse(no_m.mmse)
        self.assertEqual(no_m.controller_init, "distill")
        self.assertEqual(no_m.train_iters, no_m.total_step)
        self.assertTrue(no_i.mmse)
        self.assertEqual(no_i.controller_init, "random_init")
        self.assertEqual(no_i.train_iters, self.base_args.train_iters)

    def test_copy_ablations_select_the_right_modes(self):
        no_d = macd_ablation.configure_run(
            self.base_args, "MACD-noD", "Thrower-v0", 0
        )
        full = macd_ablation.configure_run(
            self.base_args, "MACD-FI", "Thrower-v0", 0
        )
        self.assertEqual(no_d.controller_init, "non_distill_copy")
        self.assertFalse(no_d.distill)
        self.assertEqual(full.controller_init, "full_copy")
        self.assertFalse(full.distill)

    def test_non_distill_copy_retains_only_distillation_parameters(self):
        names = {
            "encoder.layers.0.self_attn.in_proj_weight": torch.zeros(6, 2),
            "encoder.layers.0.norm1.weight": torch.zeros(2),
            "actor.weight": torch.zeros(2, 2),
        }
        copied_parent = {name: torch.full_like(value, 3) for name, value in names.items()}

        class Controller:
            def __init__(self, state):
                self.state = state

            def state_dict(self):
                return self.state

        child = Controller(copied_parent)
        _restore_distillation_parameters_from_initial(child, names)
        projection = child.state_dict()["encoder.layers.0.self_attn.in_proj_weight"]
        self.assertTrue(torch.equal(projection[:4], torch.zeros_like(projection[:4])))
        self.assertTrue(torch.equal(projection[4:], torch.full_like(projection[4:], 3)))
        self.assertTrue(torch.equal(
            child.state_dict()["encoder.layers.0.norm1.weight"], torch.zeros(2)
        ))
        self.assertTrue(torch.equal(
            child.state_dict()["actor.weight"], torch.full((2, 2), 3.0)
        ))


if __name__ == "__main__":
    unittest.main()
