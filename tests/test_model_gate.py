import os
import unittest

from src.model_gate import run_model_gate
from src.utils import project_path


class ModelGateTests(unittest.TestCase):
    def test_model_gate_writes_json_and_summary_without_hf_token(self):
        old_hf = os.environ.pop("HF_TOKEN", None)
        old_hub = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            result = run_model_gate("q6_core_native")
        finally:
            if old_hf is not None:
                os.environ["HF_TOKEN"] = old_hf
            if old_hub is not None:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hub

        self.assertEqual(result["status"], "MODEL_GATE_NEEDS_SETUP")
        self.assertIn("hf_token_missing", result["reasons"])
        self.assertEqual({model["model_id"] for model in result["models"]}, {"DG-Native", "G26-AR", "G26-MTP"})
        self.assertTrue(project_path("results", "model_gate.json").exists())
        summary_path = project_path("reports", "experiment_summary.md")
        self.assertTrue(summary_path.exists())
        text = summary_path.read_text(encoding="utf-8")
        self.assertIn("Experiment 3", text)
        self.assertIn("Success Criteria", text)


if __name__ == "__main__":
    unittest.main()
