"""Controlled OpenAI-compatible backend smoke test.

This phase verifies the local server path before any 26B model or vLLM process
is involved. It starts a tiny HTTP server bound to 127.0.0.1, checks health,
non-streaming chat completions, streaming SSE chunks, TTFC/E2E timing, and the
strict MiniToolAgent JSON parser.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backend_check import check_tcp_port_free
from .minitoolagent import AgentProtocolError, parse_response
from .utils import RESULTS_DIR, append_jsonl, git_revision, utc_now_iso, write_json


STRICT_TOOL_RESPONSE = json.dumps(
    {"type": "tool_call", "name": "get_order", "arguments": {"order_id": "O-1001"}},
    separators=(",", ":"),
)


def run_backend_smoke(profile: str = "auto", host: str = "127.0.0.1", port: int = 8765) -> dict[str, Any]:
    """Run a local OpenAI-compatible smoke server and persist results."""

    started_at = utc_now_iso()
    events_path = RESULTS_DIR / "backend_events.jsonl"
    port_status = check_tcp_port_free(host, port)
    if not port_status["free"]:
        result = {
            "timestamp": started_at,
            "profile": profile,
            "status": "BACKEND_SMOKE_FAILED",
            "server_bound_host": host,
            "server_port": port,
            "port_free": False,
            "error_type": "localhost_port_busy",
            "git": git_revision(),
        }
        write_json(RESULTS_DIR / "backend_server_smoke.json", result)
        append_jsonl(events_path, {"event": "backend_smoke_failed", **result})
        return result

    server = SmokeServer(host, port)
    server.start()
    base_url = f"http://{host}:{port}"
    result: dict[str, Any] = {
        "timestamp": started_at,
        "profile": profile,
        "server_bound_host": host,
        "server_port": port,
        "git": git_revision(),
    }
    try:
        health = get_json(f"{base_url}/health")
        result["health_ok"] = bool(health.get("ok"))

        non_stream = post_json(
            f"{base_url}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Return strict tool JSON."}], "stream": False},
        )
        non_stream_text = non_stream["choices"][0]["message"]["content"]
        result["non_streaming_ok"] = non_stream_text == STRICT_TOOL_RESPONSE
        result["non_streaming_text_length"] = len(non_stream_text)

        stream_result = post_stream(
            f"{base_url}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Return strict tool JSON."}], "stream": True},
        )
        result.update(stream_result)
        result["streaming_ok"] = stream_result["text"] == STRICT_TOOL_RESPONSE

        try:
            parsed = parse_response(non_stream_text)
            result["strict_json_ok"] = parsed["type"] == "tool_call" and parsed["name"] == "get_order"
            result["strict_json_error"] = None
        except AgentProtocolError as exc:
            result["strict_json_ok"] = False
            result["strict_json_error"] = str(exc)

        checks = ("health_ok", "non_streaming_ok", "streaming_ok", "strict_json_ok")
        result["status"] = "BACKEND_SMOKE_PASSED" if all(result.get(name) for name in checks) else "BACKEND_SMOKE_FAILED"
        append_jsonl(events_path, {"event": "backend_smoke_completed", "status": result["status"], "timestamp": utc_now_iso()})
    except Exception as exc:
        result.update(
            {
                "status": "BACKEND_SMOKE_FAILED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        append_jsonl(events_path, {"event": "backend_smoke_exception", "error_type": type(exc).__name__, "timestamp": utc_now_iso()})
    finally:
        server.stop()

    write_json(RESULTS_DIR / "backend_server_smoke.json", result)
    return result


class SmokeServer:
    """Small thread-managed HTTP server for backend path checks."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), SmokeHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                health = get_json(f"http://{self.host}:{self.port}/health", timeout=0.5)
                if health.get("ok"):
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("smoke server did not become healthy")

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class SmokeHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing the tiny OpenAI-compatible subset."""

    server_version = "BackendSmoke/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"ok": True, "service": "backend-smoke"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if payload.get("stream"):
            self.send_stream(STRICT_TOOL_RESPONSE)
        else:
            self.send_json(
                {
                    "id": "backend-smoke",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": STRICT_TOOL_RESPONSE}, "finish_reason": "stop"}],
                }
            )

    def send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_stream(self, text: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        midpoint = max(1, len(text) // 2)
        for chunk in (text[:midpoint], text[midpoint:]):
            payload = {"choices": [{"delta": {"content": chunk}}]}
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def get_json(url: str, timeout: float = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 5) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream(url: str, payload: dict[str, Any], timeout: float = 5) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content: float | None = None
    chunks: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            content = json.loads(data)["choices"][0]["delta"].get("content", "")
            if content and first_content is None:
                first_content = time.perf_counter()
            chunks.append(content)
    ended = time.perf_counter()
    return {
        "text": "".join(chunks),
        "ttfc_ms": ((first_content or ended) - started) * 1000,
        "e2e_latency_ms": (ended - started) * 1000,
        "stream_chunk_count": len(chunks),
    }
