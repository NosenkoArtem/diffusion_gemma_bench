"""OpenAI-compatible vLLM server adapter.

The adapter is intentionally thin. It records the command, starts one local
server at a time, measures streaming TTFC, and exposes telemetry placeholders.
Model-specific command flags are assembled from config files in Colab after
preflight decides which profile is feasible.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator


class VllmAdapter:
    """Manage one vLLM OpenAI-compatible server process."""

    def __init__(
        self,
        command: list[str],
        base_url: str = "http://127.0.0.1:8000",
        log_path: Path | None = None,
    ) -> None:
        self.command = command
        self.base_url = base_url.rstrip("/")
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_fh = None

    def start(self) -> None:
        """Start the server and keep stdout/stderr for reproducibility."""

        if self.process:
            raise RuntimeError("vLLM server already started")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_fh or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def health_check(self) -> dict[str, Any]:
        """Check the local health endpoint."""

        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                return {"ok": response.status == 200, "status": response.status}
        except urllib.error.URLError as exc:
            return {"ok": False, "error": str(exc)}

    def complete(self, messages: list[dict[str, str]], generation_config: dict[str, Any]) -> dict[str, Any]:
        """Make one non-streaming chat completion request."""

        payload = {"messages": messages, "stream": False, **generation_config}
        data = _post_json(f"{self.base_url}/v1/chat/completions", payload)
        text = data["choices"][0]["message"].get("content", "")
        return {"text": text, "raw": data}

    def stream_complete(self, messages: list[dict[str, str]], generation_config: dict[str, Any]) -> dict[str, Any]:
        """Make one streaming request and measure first visible content chunk."""

        start = time.perf_counter()
        chunks: list[str] = []
        first_content: float | None = None
        for chunk in _stream_sse(f"{self.base_url}/v1/chat/completions", {"messages": messages, "stream": True, **generation_config}):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            if content and first_content is None:
                first_content = time.perf_counter()
            chunks.append(content)
        end = time.perf_counter()
        return {
            "text": "".join(chunks),
            "ttfc_ms": ((first_content or end) - start) * 1000,
            "e2e_latency_ms": (end - start) * 1000,
        }

    def telemetry(self) -> dict[str, Any]:
        """Return backend telemetry when available.

        vLLM exposes different metrics across versions; Colab implementation can
        extend this method after the installed version is known.
        """

        return {"available": False}

    def unload(self) -> None:
        """Stop the server process and close logs."""

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_sse(url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            yield json.loads(data)
