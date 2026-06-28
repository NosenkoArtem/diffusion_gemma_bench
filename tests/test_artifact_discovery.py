import unittest

from src.artifact_discovery import run_artifact_discovery
from src.utils import project_path


class ArtifactDiscoveryTests(unittest.TestCase):
    def test_artifact_discovery_writes_review_artifacts_without_network(self):
        result = run_artifact_discovery("q6_core_native", enable_search=False)

        self.assertEqual(result["status"], "ARTIFACT_DISCOVERY_NEEDS_REVIEW")
        self.assertIn("search_disabled", result["reasons"])
        self.assertEqual({model["model_id"] for model in result["models"]}, {"DG-Native", "G26-AR", "G26-MTP"})
        self.assertTrue(all("search_errors" in model for model in result["models"]))
        self.assertTrue(project_path("results", "artifact_discovery.json").exists())
        summary_path = project_path("reports", "experiment_summary_artifact-discovery.md")
        self.assertTrue(summary_path.exists())
        self.assertIn("Experiment 5", summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
