import unittest

from src.mtp_tuning import choose_mtp_depth


class MtpConfigTests(unittest.TestCase):
    def test_choose_fastest_viable_depth(self):
        chosen = choose_mtp_depth(
            [
                {
                    "num_speculative_tokens": 2,
                    "valid_json_rate": 1.0,
                    "task_success_rate": 1.0,
                    "median_e2e_latency_ms": {"S2": 100, "S3": 110, "S4": 120},
                },
                {
                    "num_speculative_tokens": 4,
                    "valid_json_rate": 1.0,
                    "task_success_rate": 1.0,
                    "median_e2e_latency_ms": {"S2": 90, "S3": 95, "S4": 100},
                },
            ],
            {"valid_json_rate": 1.0, "task_success_rate": 1.0},
        )
        self.assertEqual(chosen["selected"]["num_speculative_tokens"], 4)

    def test_rejects_quality_drop(self):
        chosen = choose_mtp_depth(
            [
                {
                    "num_speculative_tokens": 6,
                    "valid_json_rate": 0.90,
                    "task_success_rate": 1.0,
                    "median_e2e_latency_ms": {"S2": 1, "S3": 1, "S4": 1},
                }
            ],
            {"valid_json_rate": 0.99, "task_success_rate": 1.0},
        )
        self.assertIsNone(chosen["selected"])


if __name__ == "__main__":
    unittest.main()
