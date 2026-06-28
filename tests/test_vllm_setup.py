import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.vllm_setup import run_vllm_setup


class VllmSetupTests(unittest.TestCase):
    def test_vllm_setup_writes_json_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_vllm_setup("q6_core_native", results_dir=root / "results", reports_dir=root / "reports")

            self.assertIn(result["status"], {"VLLM_SETUP_PASSED", "VLLM_SETUP_NEEDS_SETUP"})
            self.assertIn("vllm_import", result)
            self.assertTrue((root / "results" / "vllm_setup.json").exists())
            summary_path = root / "reports" / "experiment_summary.md"
            self.assertTrue(summary_path.exists())
            text = summary_path.read_text(encoding="utf-8")
            self.assertIn("Experiment 4", text)
            self.assertIn("vLLM", text)


if __name__ == "__main__":
    unittest.main()
