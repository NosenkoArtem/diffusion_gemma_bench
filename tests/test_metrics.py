import unittest

from src.metrics import agent_quality_summary, paired_success, speed_summary, wilson_interval


class MetricsTests(unittest.TestCase):
    def test_wilson_interval_bounds_rate(self):
        lo, hi = wilson_interval(8, 10)
        self.assertLessEqual(lo, 0.8)
        self.assertGreaterEqual(hi, 0.8)

    def test_speed_summary_groups_records(self):
        summary = speed_summary(
            [
                {"model_id": "DG-Native", "prompt_id": "S2", "track": "cold", "ttfc_ms": 10, "e2e_latency_ms": 100},
                {"model_id": "DG-Native", "prompt_id": "S2", "track": "cold", "ttfc_ms": 20, "e2e_latency_ms": 120},
            ]
        )
        self.assertEqual(summary["groups"][0]["n"], 2)
        self.assertEqual(summary["groups"][0]["median_ttfc_ms"], 15.0)

    def test_agent_quality_summary(self):
        summary = agent_quality_summary(
            [
                {"model_id": "DG-Native", "task_success": True, "valid_json": True},
                {"model_id": "DG-Native", "task_success": False, "valid_json": True},
            ]
        )
        self.assertEqual(summary["groups"][0]["task_success_rate"], 0.5)

    def test_paired_success(self):
        out = paired_success(
            [
                {"task_id": "A", "model_id": "DG", "task_success": True},
                {"task_id": "A", "model_id": "G26", "task_success": False},
            ],
            "DG",
            "G26",
        )
        self.assertEqual(out["DG_only_success"], 1)


if __name__ == "__main__":
    unittest.main()
