import socket
import unittest

from src.backend_smoke import STRICT_TOOL_RESPONSE, SmokeServer, get_json, post_json, post_stream, run_backend_smoke
from src.minitoolagent import parse_response
from src.utils import project_path


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class BackendSmokeTests(unittest.TestCase):
    def test_smoke_server_health_and_completion(self):
        port = free_port()
        server = SmokeServer("127.0.0.1", port)
        server.start()
        try:
            self.assertTrue(get_json(f"http://127.0.0.1:{port}/health")["ok"])
            response = post_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "json"}], "stream": False},
            )
            text = response["choices"][0]["message"]["content"]
            self.assertEqual(text, STRICT_TOOL_RESPONSE)
            self.assertEqual(parse_response(text)["name"], "get_order")
        finally:
            server.stop()

    def test_smoke_server_streaming(self):
        port = free_port()
        server = SmokeServer("127.0.0.1", port)
        server.start()
        try:
            result = post_stream(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "json"}], "stream": True},
            )
            self.assertEqual(result["text"], STRICT_TOOL_RESPONSE)
            self.assertGreaterEqual(result["stream_chunk_count"], 2)
            self.assertGreaterEqual(result["ttfc_ms"], 0)
        finally:
            server.stop()

    def test_run_backend_smoke_writes_result(self):
        port = free_port()
        result = run_backend_smoke("q6_core_native", port=port)
        self.assertEqual(result["status"], "BACKEND_SMOKE_PASSED")
        self.assertEqual(result["server_bound_host"], "127.0.0.1")
        self.assertTrue(project_path("results", "backend_server_smoke.json").exists())


if __name__ == "__main__":
    unittest.main()
