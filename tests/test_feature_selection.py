"""Unit tests for feature_selection utilities (posepred ablations).

These tests are intentionally small and dependency-free.
They validate deterministic behavior and basic matching semantics.
"""

from __future__ import annotations

import tempfile
import unittest

from autobehaver.feature_selection import (
    DEFAULT_PRESET,
    FeaturePlan,
    compute_schema_hash,
    parse_csv_list_arg,
    read_columns_list_file,
    resolve_feature_plan,
)


class TestFeatureSelection(unittest.TestCase):
    def test_parse_csv_list_arg(self):
        self.assertEqual(parse_csv_list_arg(None), [])
        self.assertEqual(parse_csv_list_arg("a,b, c"), ["a", "b", "c"])
        self.assertEqual(parse_csv_list_arg(["a,b", "c"]), ["a", "b", "c"])
        self.assertEqual(parse_csv_list_arg(["  a  ", "", "b"]), ["a", "b"])

    def test_schema_hash_deterministic_and_ordered(self):
        a = ["x", "y"]
        b = ["x", "y"]
        c = ["y", "x"]
        self.assertEqual(compute_schema_hash(a), compute_schema_hash(b))
        self.assertNotEqual(compute_schema_hash(a), compute_schema_hash(c))

    def test_resolve_feature_plan_default_preset_is_no_pos(self):
        # Ensure DEFAULT_PRESET is the expected one.
        self.assertEqual(DEFAULT_PRESET, "no_pos")

    def test_resolve_feature_plan_exclude_rules(self):
        headers = [
            "centroid_x",
            "centroid_y",
            "speed_inst",
            "foo",
            "angle_sin",
            "angle_cos",
        ]
        plan = resolve_feature_plan(headers, preset="no_pos")
        self.assertIsInstance(plan, FeaturePlan)
        # no_pos should remove centroid_x/y and angle_sin/cos
        self.assertNotIn("centroid_x", plan.included_columns)
        self.assertNotIn("centroid_y", plan.included_columns)
        self.assertNotIn("angle_sin", plan.included_columns)
        self.assertNotIn("angle_cos", plan.included_columns)
        # but keep speed_inst and foo
        self.assertIn("speed_inst", plan.included_columns)
        self.assertIn("foo", plan.included_columns)

    def test_keyword_exclude_case_insensitive(self):
        headers = ["Speed_inst", "velocity_forward_3", "foo"]
        plan = resolve_feature_plan(headers, preset="none", exclude_keywords="speed_")
        self.assertNotIn("Speed_inst", plan.included_columns)
        self.assertIn("velocity_forward_3", plan.included_columns)
        self.assertIn("foo", plan.included_columns)

    def test_include_columns_file(self):
        headers = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as td:
            p = td + "/cols.txt"
            with open(p, "w", encoding="utf-8") as f:
                f.write("# comment\n")
                f.write("b\n")
                f.write("a\n")
                f.write("\n")
            included = read_columns_list_file(p)
            self.assertEqual(included, ["b", "a"])

            plan = resolve_feature_plan(headers, include_columns_file=p)
            self.assertEqual(plan.included_columns, ["b", "a"])
            self.assertEqual(plan.excluded_columns, ["c"])

    def test_duplicate_headers_error(self):
        headers = ["a", "b", "a"]
        with self.assertRaises(ValueError):
            resolve_feature_plan(headers, preset="none")


if __name__ == "__main__":
    unittest.main()
