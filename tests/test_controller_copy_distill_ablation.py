import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import run_controller_copy_distill_ablation_tasks as runner
import run_offspring_attention_ablation as roaa
from macd.controller_distillation import build_copied_child_controller
from macd.transformer.config import ppoconfig, transformerconfig


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFSPRING_DIR = os.path.join(
    ROOT_DIR, "result", "attention_loss_ablation_stratified_20", "offspring"
)


class ControllerFixture:
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.body = roaa.load_parent_structure(
            os.path.join(OFFSPRING_DIR, "offspring_1", "body.npz")
        )
        cls.sample_setting = roaa.get_sample_setting("Walker-v0", cls.body)
        cls.ppo_cfg = ppoconfig()
        cls.ppo_cfg.env_name = "Walker-v0"
        cls.trans = transformerconfig()

    def new_controller(self, seed):
        roaa.set_all_seeds(seed)
        return roaa.new_random_controller(
            self.sample_setting, self.ppo_cfg, self.trans, self.device
        )

    def copied_controller(self, parent, seed, mode):
        roaa.set_all_seeds(seed)
        return build_copied_child_controller(
            parent,
            self.sample_setting,
            self.ppo_cfg,
            self.trans,
            self.device,
            mode,
        )


class ControllerCopyTest(ControllerFixture, unittest.TestCase):

    def test_full_copy_matches_parent_exactly(self):
        parent = self.new_controller(11)
        child = self.copied_controller(parent, 22, "full_copy")
        audit = runner.audit_full_copy(
            runner.clone_state(parent), child.state_dict()
        )
        self.assertEqual(audit["random_parameter_entries"], [])
        self.assertEqual(
            len(audit["copied_parameter_entries"]), len(parent.state_dict())
        )

    def test_non_distill_copy_keeps_random_qk_and_norm_but_copies_v(self):
        parent = self.new_controller(31)
        random_child = self.new_controller(42)
        child = self.copied_controller(parent, 42, "non_distill_copy")
        audit = runner.audit_non_distill_copy(
            runner.clone_state(parent),
            runner.clone_state(random_child),
            child.state_dict(),
        )
        random_entries = audit["random_parameter_entries"]
        copied_entries = audit["copied_parameter_entries"]
        self.assertTrue(any(entry.endswith("[QK]") for entry in random_entries))
        self.assertTrue(any("norm" in entry for entry in random_entries))
        self.assertTrue(any(entry.endswith("[V]") for entry in copied_entries))

    def test_two_full_copy_arms_start_identically(self):
        parent = self.new_controller(51)
        first = self.copied_controller(parent, 62, "full_copy")
        second = self.copied_controller(parent, 62, "full_copy")
        self.assertTrue(roaa.state_dicts_equal(first, second))

    def test_invalid_copy_mode_is_rejected(self):
        parent = self.new_controller(71)
        with self.assertRaises(ValueError):
            self.copied_controller(parent, 82, "unknown")


class ArmPreparationTest(ControllerFixture, unittest.TestCase):
    def test_only_full_copy_distill_arm_runs_warmup(self):
        parent = self.new_controller(91)

        def controlled_warmup(_parent, child, *_args, **_kwargs):
            with torch.no_grad():
                for name, parameter in child.named_parameters():
                    if "norm" in name:
                        parameter.add_(1.0)
                        return
            raise AssertionError("Test controller exposes no LayerNorm parameter")

        with mock.patch.object(
            runner, "collect_parent_rollout_obs", return_value=[{"obs": 1}]
        ) as collect_mock, mock.patch.object(
            runner, "attention_distill_warmup", side_effect=controlled_warmup
        ) as warmup_mock:
            no_distill, _, no_distill_audit = runner.prepare_arm_controller(
                "full_copy_no_distill",
                parent,
                None,
                self.body,
                self.body,
                "Walker-v0",
                102,
                5,
                2,
                self.device,
            )
            distilled, _, distilled_audit = runner.prepare_arm_controller(
                "full_copy_distill",
                parent,
                None,
                self.body,
                self.body,
                "Walker-v0",
                102,
                5,
                2,
                self.device,
            )

        self.assertTrue(roaa.state_dicts_equal(parent, no_distill))
        self.assertFalse(roaa.state_dicts_equal(parent, distilled))
        self.assertFalse(no_distill_audit["distillation_enabled"])
        self.assertTrue(distilled_audit["distillation_enabled"])
        self.assertEqual(collect_mock.call_count, 1)
        self.assertEqual(warmup_mock.call_count, 1)
        self.assertTrue(
            all(
                runner.is_distillation_state_entry(name)
                for name in distilled_audit["warmup_changed_parameter_entries"]
            )
        )


class ControllerCopySchedulingTest(unittest.TestCase):
    def test_current_offspring_set_creates_eighty_paired_tasks(self):
        args = SimpleNamespace(
            offspring_dir=OFFSPRING_DIR,
            offspring_ids=None,
            seed_base=2000,
        )
        offspring = runner.loss_runner.discover_offspring(args)
        self.assertEqual(len(offspring), 20)
        self.assertEqual(len(offspring) * len(runner.ARM_ORDER), 80)
        for offspring_index in offspring:
            seeds = {
                args.seed_base + offspring_index for _arm in runner.ARM_ORDER
            }
            self.assertEqual(seeds, {args.seed_base + offspring_index})

    def test_forty_workers_use_one_thread_and_two_waves(self):
        parallel, threads = runner.loss_runner.resolve_resource_config(
            40, 1, logical_cpus=72
        )
        offspring_per_wave = parallel // len(runner.ARM_ORDER)
        self.assertEqual((parallel, threads), (40, 1))
        self.assertEqual(offspring_per_wave, 10)
        self.assertEqual((20 + offspring_per_wave - 1) // offspring_per_wave, 2)

    def test_complete_curve_requires_every_expected_update(self):
        with tempfile.TemporaryDirectory() as directory:
            curve_path = os.path.join(directory, "learning_curve.csv")
            records = [
                {
                    "offspring_index": 1,
                    "arm": "random_init",
                    "seed": 2001,
                    "update": update,
                    "env_steps": update * 128,
                    "return": float(update),
                    "best_return": float(update),
                }
                for update in range(0, 201, 5)
            ]
            runner.write_curve_csv(curve_path, records)
            self.assertTrue(runner.curve_is_complete(curve_path, 200, 5))
            runner.write_curve_csv(curve_path, records[:-1])
            self.assertFalse(runner.curve_is_complete(curve_path, 200, 5))


if __name__ == "__main__":
    unittest.main()
