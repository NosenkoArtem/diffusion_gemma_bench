import os
import tempfile
import unittest
from pathlib import Path

from src.env_config import get_experiment_env, load_env_file, sanitized_env_summary
from src.utils import project_path


class EnvConfigTests(unittest.TestCase):
    def test_example_env_exists(self):
        self.assertTrue(project_path("configs", "experiment.env.example").exists())

    def test_load_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.env"
            path.write_text("PROFILE=q5_core_native\nHF_TOKEN=\n", encoding="utf-8")
            self.assertEqual(load_env_file(path)["PROFILE"], "q5_core_native")

    def test_sanitized_summary_does_not_include_secret_value(self):
        old = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "hf_secret_value"
        try:
            summary = sanitized_env_summary(get_experiment_env())
            self.assertTrue(summary["secret_presence"]["HF_TOKEN"])
            self.assertNotIn("hf_secret_value", str(summary))
        finally:
            if old is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old


if __name__ == "__main__":
    unittest.main()
