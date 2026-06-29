import unittest

from run import main


class CliTests(unittest.TestCase):
    def test_model_load_smoke_requires_explicit_real_or_dry_run_mode(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--profile", "q6_core_native", "--phase", "model-load-smoke", "--confirm-go"])

        self.assertIn("--download/--load", str(ctx.exception))

    def test_llama_load_smoke_requires_explicit_real_or_dry_run_mode(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--profile", "q6_core_native", "--phase", "llama-load-smoke", "--confirm-go"])

        self.assertIn("--download/--load", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
