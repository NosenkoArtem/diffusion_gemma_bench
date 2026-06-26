import socket
import unittest

from src.backend_check import check_tcp_port_free, next_step_from_reasons, run_backend_check
from src.utils import project_path
from scripts.push_results_to_github import authenticated_remote_url, needs_github_token, redact_secret, remote_type


class BackendCheckTests(unittest.TestCase):
    def test_port_free_detection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        try:
            status = check_tcp_port_free(host, port)
            self.assertFalse(status["free"])
        finally:
            sock.close()

    def test_next_step_mentions_vllm(self):
        self.assertIn("vLLM", next_step_from_reasons(["vllm_not_importable"]))

    def test_backend_check_writes_json(self):
        result = run_backend_check("auto", port=8765)
        self.assertIn(result["status"], {"BACKEND_CHECK_PASSED", "BACKEND_CHECK_NEEDS_SETUP"})
        self.assertTrue(project_path("results", "backend_capability.json").exists())

    def test_github_token_injected_only_for_github_https(self):
        import os

        old = os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = "token-value"
        try:
            self.assertEqual(
                authenticated_remote_url("https://github.com/o/r.git"),
                "https://x-access-token:token-value@github.com/o/r.git",
            )
            self.assertEqual(
                authenticated_remote_url("git@github.com:o/r.git"),
                "git@github.com:o/r.git",
            )
        finally:
            if old is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = old

    def test_redact_secret_removes_token_from_errors(self):
        import os

        old = os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = "token-value"
        try:
            self.assertEqual(redact_secret("bad token-value error"), "bad *** error")
        finally:
            if old is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = old

    def test_remote_type_and_token_requirement(self):
        self.assertEqual(remote_type("https://github.com/o/r.git"), "github_https")
        self.assertEqual(remote_type("git@github.com:o/r.git"), "github_ssh")
        self.assertEqual(remote_type("https://example.com/o/r.git"), "other")
        self.assertTrue(needs_github_token("https://github.com/o/r.git"))
        self.assertFalse(needs_github_token("git@github.com:o/r.git"))


if __name__ == "__main__":
    unittest.main()
