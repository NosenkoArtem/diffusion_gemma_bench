import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.artifact_discovery import blocking_reasons, next_step, run_artifact_discovery


class ArtifactDiscoveryTests(unittest.TestCase):
    def test_artifact_discovery_writes_review_artifacts_without_network(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_artifact_discovery(
                "q6_core_native",
                enable_search=False,
                results_dir=root / "results",
                reports_dir=root / "reports",
            )

            self.assertEqual(result["status"], "ARTIFACT_DISCOVERY_NEEDS_REVIEW")
            self.assertIn("search_disabled", result["reasons"])
            self.assertEqual({model["model_id"] for model in result["models"]}, {"DG-Native", "G26-AR", "G26-MTP"})
            self.assertTrue(all("search_errors" in model for model in result["models"]))
            self.assertTrue((root / "results" / "artifact_discovery.json").exists())
            summary_path = root / "reports" / "experiment_summary_artifact-discovery.md"
            self.assertTrue(summary_path.exists())
            self.assertIn("Experiment 5", summary_path.read_text(encoding="utf-8"))

    def test_missing_token_is_reported_before_candidate_review(self):
        reasons = blocking_reasons(
            [{"model_id": "DG-Native", "best_candidate": None, "error_type": "hf_token_missing"}],
            hf_token_present=False,
            hf_hub_available=True,
            enable_search=True,
        )

        self.assertEqual(reasons, ["hf_token_missing"])
        self.assertIn("Load HF_TOKEN", next_step("ARTIFACT_DISCOVERY_NEEDS_REVIEW", reasons))

    def test_expired_token_is_reported_before_candidate_review(self):
        reasons = blocking_reasons(
            [
                {
                    "model_id": "DG-Native",
                    "best_candidate": None,
                    "error_type": "search_failed",
                    "search_errors": [
                        {
                            "error_type": "HfHubHTTPError",
                            "error": "401 Unauthorized. User Access Token is expired.",
                        }
                    ],
                    "candidate_repos": [],
                }
            ],
            hf_token_present=True,
            hf_hub_available=True,
            enable_search=True,
        )

        self.assertEqual(reasons, ["hf_token_invalid"])
        self.assertIn("Refresh the Hugging Face token", next_step("ARTIFACT_DISCOVERY_NEEDS_REVIEW", reasons))


if __name__ == "__main__":
    unittest.main()
