"""Offline contract tests: no model, GPU, network service or credentials required."""
import http.server
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/long_test.py"

class Handler(http.server.BaseHTTPRequestHandler):
    status = 200
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.received = request
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.status == 200:
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "4827 7931 1648"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 259293, "completion_tokens": 512}, "timings": {"predicted_per_second": 100}}).encode())
        else:
            self.wfile.write(b'{"error":"private body"}')
    def log_message(self, *args):
        pass

class TestRequest(unittest.TestCase):
    def run_case(self, status):
        Handler.status = status
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            env = {**os.environ, "STRATA_API_KEY": "test-secret", "STRATA_BASE_URL": f"http://127.0.0.1:{server.server_port}", "STRATA_MODEL": "test-model", "STRATA_CONFIG_LABEL": "test-config"}
            result = subprocess.run([sys.executable, str(SCRIPT), "test-run"], env=env, text=True, capture_output=True, timeout=30)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("test-secret", result.stdout + result.stderr)
            self.assertNotIn("private body", result.stdout + result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["configuration_label"], "test-config")
            self.assertEqual(len(data["prompt_sha256"]), 64)
            self.assertEqual(server.received["max_tokens"], 512)
            return result, data
        finally:
            server.server_close()

    def test_success(self):
        result, data = self.run_case(200)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(data["status"], "done")
        self.assertTrue(all(data["markers"].values()))
        self.assertEqual(data["finish"], "length")

    def test_http_error(self):
        result, data = self.run_case(502)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["http_status"], 502)

if __name__ == "__main__":
    unittest.main()
