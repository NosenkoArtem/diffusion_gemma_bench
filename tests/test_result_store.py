import tempfile
import unittest
from pathlib import Path

from src.result_store import package_results, validate_result_tree


class ResultStoreTests(unittest.TestCase):
    def test_package_results_copies_small_allowed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            reports = root / "reports"
            results.mkdir()
            reports.mkdir()
            (results / "preflight.json").write_text('{"ok": true}\n', encoding="utf-8")
            (reports / "final_report.md").write_text("# Report\n", encoding="utf-8")

            manifest = package_results(
                run_id="run-1",
                profile="q6_core_native",
                phase="smoke",
                source_root=root,
                results_dir=results,
                reports_dir=reports,
            )

            run_dir = results / "runs" / "run-1"
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "final_report.md").exists())
            self.assertTrue((run_dir / "result_manifest.json").exists())
            self.assertEqual(len(manifest["copied_files"]), 2)

    def test_validate_result_tree_rejects_secret_like_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "preflight.json").write_text('{"token": "ghp_123456789012345678901234"}', encoding="utf-8")
            report = validate_result_tree(run_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(any("possible_secret" in error for error in report["errors"]))

    def test_validate_result_tree_rejects_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "model.gguf").write_bytes(b"not really a model")
            report = validate_result_tree(run_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(any("denied_pattern" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
