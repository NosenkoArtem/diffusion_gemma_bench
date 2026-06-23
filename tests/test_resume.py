import unittest

from src.bootstrap import start_phase


class ResumeTests(unittest.TestCase):
    def test_run_id_can_be_reused(self):
        manifest = start_phase("preflight", "auto", run_id="fixed-run")
        self.assertEqual(manifest["run_id"], "fixed-run")
        self.assertEqual(manifest["phase"], "preflight")


if __name__ == "__main__":
    unittest.main()
