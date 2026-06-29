import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.llama_load_smoke import classify_llama_error, run_llama_load_smoke


class LlamaLoadSmokeTests(unittest.TestCase):
    def test_llama_load_smoke_dry_run_writes_artifacts_for_both_models(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_llama_load_smoke(
                "q6_core_native",
                targets=("G26-AR", "DG-Native"),
                download_enabled=False,
                load_enabled=False,
                results_dir=root / "results",
                reports_dir=root / "reports",
            )

            self.assertEqual(result["phase"], "llama-load-smoke")
            self.assertEqual(result["targets"], ["G26-AR", "DG-Native"])
            self.assertIn("download_disabled", result["reasons"])
            self.assertIn("load_disabled", result["reasons"])
            self.assertEqual(result["models"][0]["filename"], "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
            self.assertEqual(result["models"][1]["filename"], "diffusiongemma-26B-A4B-it-Q6_K.gguf")
            self.assertTrue((root / "results" / "llama_load_smoke.json").exists())
            self.assertTrue((root / "reports" / "experiment_summary_llama-load-smoke.md").exists())

    def test_llama_error_classifier_marks_unsupported_model(self):
        self.assertEqual(classify_llama_error("error: unsupported architecture"), "unsupported_model_or_gguf")


if __name__ == "__main__":
    unittest.main()
