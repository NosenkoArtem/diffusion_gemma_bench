import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.model_gate import expected_profile_filename, load_yaml, project_path, run_model_gate


class ModelGateTests(unittest.TestCase):
    def test_gemma_qat_filename_matches_discovered_unsloth_artifact(self):
        models = load_yaml(project_path("configs", "models.yaml")).get("models", {})

        self.assertEqual(
            expected_profile_filename(models["G26-AR"], "q6_core_native"),
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        )
        self.assertEqual(
            expected_profile_filename(models["G26-MTP"], "q6_core_native"),
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        )

    def test_model_gate_writes_json_and_summary_without_hf_token(self):
        old_hf = os.environ.pop("HF_TOKEN", None)
        old_hub = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = run_model_gate("q6_core_native", results_dir=root / "results", reports_dir=root / "reports")

                self.assertEqual(result["status"], "MODEL_GATE_NEEDS_SETUP")
                self.assertIn("hf_token_missing", result["reasons"])
                self.assertEqual({model["model_id"] for model in result["models"]}, {"DG-Native", "G26-AR", "G26-MTP"})
                self.assertTrue((root / "results" / "model_gate.json").exists())
                summary_path = root / "reports" / "experiment_summary.md"
                self.assertTrue(summary_path.exists())
                text = summary_path.read_text(encoding="utf-8")
                self.assertIn("Experiment 3", text)
                self.assertIn("Success Criteria", text)
        finally:
            if old_hf is not None:
                os.environ["HF_TOKEN"] = old_hf
            if old_hub is not None:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hub


if __name__ == "__main__":
    unittest.main()
