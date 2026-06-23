import json
import unittest
from collections import Counter
from pathlib import Path

from src.minitoolagent import TOOLS
from src.preflight import choose_profile
from src.utils import project_path, read_jsonl


class ManifestTests(unittest.TestCase):
    def test_required_project_files_exist(self):
        for rel in [
            "configs/profiles.yaml",
            "configs/models.yaml",
            "configs/generation.yaml",
            "configs/mtp.yaml",
            "configs/benchmark_manifest.yaml",
            "data/speed_prompts.jsonl",
            "data/minitoolagent_v1.jsonl",
            "run.py",
        ]:
            self.assertTrue(project_path(*rel.split("/")).exists(), rel)

    def test_speed_prompts_are_jsonl(self):
        records = read_jsonl(project_path("data", "speed_prompts.jsonl"))
        self.assertGreaterEqual(len(records), 5)
        self.assertTrue(all("prompt_id" in rec for rec in records))

    def test_minitool_seed_tasks_are_jsonl(self):
        records = read_jsonl(project_path("data", "minitoolagent_v1.jsonl"))
        self.assertEqual(len(records), 60)
        self.assertEqual(Counter(rec["language"] for rec in records), {"en": 48, "ru": 12})
        self.assertEqual(
            Counter(rec["category"] for rec in records),
            {
                "single_step_tool_call": 12,
                "correct_tool_choice": 10,
                "correct_arguments": 10,
                "multi_step_workflow": 12,
                "recovery_after_tool_error": 8,
                "policy_account_restrictions": 8,
            },
        )
        self.assertEqual(len({rec["task_id"] for rec in records}), 60)

    def test_minitool_expected_tools_are_known(self):
        records = read_jsonl(project_path("data", "minitoolagent_v1.jsonl"))
        for rec in records:
            expected = rec["expected"]
            if "first_tool" in expected:
                self.assertIn(expected["first_tool"], TOOLS, rec["task_id"])
            for tool_name in expected.get("must_call", []):
                self.assertIn(tool_name, TOOLS, rec["task_id"])
            for tool_name in expected.get("tool_sequence", []):
                self.assertIn(tool_name, TOOLS, rec["task_id"])

    def test_bfcl_manifest_is_valid_json(self):
        path = project_path("data", "bfcl_subset_manifest.json")
        json.loads(Path(path).read_text(encoding="utf-8"))

    def test_profile_selection(self):
        self.assertEqual(choose_profile(48, 80, "auto")[0], "q6_core_native")
        self.assertEqual(choose_profile(35, 80, "auto")[0], "q5_core_native")
        self.assertEqual(choose_profile(20, 80, "auto")[1], "STOP")


if __name__ == "__main__":
    unittest.main()
