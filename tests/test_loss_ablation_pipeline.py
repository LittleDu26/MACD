import os
import tempfile
import unittest

import numpy as np
from evogym import hashable

import prepare_loss_ablation as prepare
import run_loss_ablation_tasks as runner


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(
    ROOT_DIR, "result", "MACD(2loss)", "Walker-v0", "0"
)


class ParentSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.individuals = prepare.parse_all_individuals(
            os.path.join(RESULT_DIR, "out.txt")
        )
        cls.eligible = prepare.eligible_parent_records(
            RESULT_DIR, cls.individuals, 13
        )
        cls.bodies = {
            item["id"]: prepare.roaa.load_parent_structure(item["structure_path"])
            for item in cls.eligible
        }

    def test_current_run_has_expected_eligible_parent_population(self):
        self.assertEqual(len(self.eligible), 41)
        self.assertTrue(all(item["maturity"] >= 13 for item in self.eligible))
        self.assertEqual(len({hashable(body) for body in self.bodies.values()}), 41)

    def test_farthest_point_parent_selection_is_deterministic(self):
        first_ids, first_distances = prepare.select_diverse_parents(
            self.eligible, self.bodies, 30
        )
        second_ids, second_distances = prepare.select_diverse_parents(
            list(reversed(self.eligible)), self.bodies, 30
        )
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_distances, second_distances)
        self.assertEqual(len(first_ids), 30)
        self.assertEqual(first_ids[0], 1076)
        self.assertEqual(len(set(first_ids)), 30)

    def test_nearest_parent_reports_all_ties_in_pool_order(self):
        child = np.array([[1, 0], [0, 0]])
        bodies = {
            8: np.array([[1, 2], [0, 0]]),
            3: np.array([[1, 0], [4, 0]]),
            5: np.array([[2, 0], [0, 0]]),
        }
        parent_id, count, tied_ids = prepare.most_similar_parents(
            child, [8, 3, 5], bodies
        )
        self.assertEqual(parent_id, 8)
        self.assertEqual(count, 1)
        self.assertEqual(tied_ids, [8, 3])


class StratifiedMutationTest(unittest.TestCase):
    def test_fixed_seed_generates_five_unique_children_per_bin(self):
        individuals = prepare.parse_all_individuals(
            os.path.join(RESULT_DIR, "out.txt")
        )
        eligible = prepare.eligible_parent_records(RESULT_DIR, individuals, 13)
        eligible_bodies = {
            item["id"]: prepare.roaa.load_parent_structure(item["structure_path"])
            for item in eligible
        }
        parent_ids, _ = prepare.select_diverse_parents(
            eligible, eligible_bodies, 30
        )
        parent_bodies = {
            parent_id: eligible_bodies[parent_id] for parent_id in parent_ids
        }
        mutation_record = {
            hashable(body): [] for body in eligible_bodies.values()
        }
        selected_hashes = set(mutation_record)
        children = []
        for bin_index, distance_bin in enumerate(prepare.DEFAULT_DISTANCE_BINS):
            low, high = distance_bin
            for child_slot in range(5):
                result = prepare.generate_stratified_child(
                    parent_ids,
                    parent_bodies,
                    distance_bin,
                    bin_index,
                    child_slot,
                    20260818,
                    1000,
                    mutation_record,
                    selected_hashes,
                    30,
                    100,
                )
                child = result["child_body"]
                distance = prepare.body_hamming_distance(
                    parent_bodies[result["direct_parent_id"]], child
                )
                self.assertGreaterEqual(distance, low)
                self.assertLessEqual(distance, high)
                self.assertGreaterEqual(result["similarity_count"], 4)
                children.append(child)
        self.assertEqual(len(children), 20)
        self.assertEqual(len({hashable(child) for child in children}), 20)


class ResourceSchedulingTest(unittest.TestCase):
    def test_auto_resource_resolution_for_current_machine_shape(self):
        self.assertEqual(runner.resolve_resource_config(0, 0, 12), (10, 1))

    def test_twenty_explicit_workers_are_supported(self):
        self.assertEqual(runner.resolve_resource_config(20, 1, 32), (20, 1))

    def test_explicit_resource_values_override_auto_mode(self):
        self.assertEqual(runner.resolve_resource_config(4, 2, 12), (4, 2))

    def test_relocated_offspring_asset_uses_local_fallback(self):
        with tempfile.TemporaryDirectory() as offspring_dir:
            local_path = os.path.join(offspring_dir, "body.npz")
            with open(local_path, "wb") as output_file:
                output_file.write(b"test")
            resolved = runner.resolve_offspring_asset_path(
                offspring_dir, "/missing/old/workspace/body.npz", "body.npz"
            )
            self.assertEqual(resolved, local_path)


if __name__ == "__main__":
    unittest.main()
