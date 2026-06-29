import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.model_load_smoke import classify_load_error, run_model_load_smoke


class ModelLoadSmokeTests(unittest.TestCase):
    def test_model_load_smoke_dry_run_writes_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_model_load_smoke(
                "q6_core_native",
                targets=("G26-AR",),
                download_enabled=False,
                load_enabled=False,
                results_dir=root / "results",
                reports_dir=root / "reports",
            )

            self.assertEqual(result["phase"], "model-load-smoke")
            self.assertEqual(result["targets"], ["G26-AR"])
            self.assertIn("cuda_runtime", result)
            self.assertIn("imports", result)
            self.assertIn("vllm", result["imports"])
            self.assertIn("libcudart_so_13_candidates", result["cuda_runtime"])
            self.assertIn("download_disabled", result["reasons"])
            self.assertIn("load_disabled", result["reasons"])
            self.assertEqual(result["models"][0]["filename"], "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
            self.assertTrue((root / "results" / "model_load_smoke.json").exists())
            self.assertTrue((root / "reports" / "experiment_summary_model-load-smoke.md").exists())

    def test_vllm_layerwise_kv_heads_error_is_classified(self):
        error = TypeError("Field 'num_key_value_heads' expected int, got list (value: [8, 2])")

        self.assertEqual(classify_load_error(error), "vllm_model_config_incompatible")


if __name__ == "__main__":
    unittest.main()
