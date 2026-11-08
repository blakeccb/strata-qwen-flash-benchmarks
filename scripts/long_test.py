#!/usr/bin/env python3
"""Near-256K synthetic long-context request; endpoint and key come from env."""
import json
import os
import sys
import time
import urllib.request

base = os.environ.get("STRATA_BASE_URL", "http://127.0.0.1:8095").rstrip("/")
key = os.environ["STRATA_API_KEY"]
model = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-strata-ud-q4_k_xl")
tag = sys.argv[1] if len(sys.argv) > 1 else str(time.time_ns())
unit = "\nRecord alpha: the routine monitoring system reports nominal status and no pending incidents.\n"
n = 16200
a, b = n // 3, 2 * n // 3
prompt = ("Evaluation " + tag + ". Memorize the three secret values and answer the final question.\n"
          + unit * a + "\nFirst secret: amber 4827.\n"
          + unit * (b - a) + "\nSecond secret: cobalt 7931.\n"
          + unit * (n - b) + "\nThird secret: jade 1648.\n"
          + "List all three secrets, then explain a careful method for validating memory across a long document. Give a detailed numbered answer with examples.")
payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": 512, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
start = time.monotonic()
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)
choice = result["choices"][0]
content = choice["message"].get("content") or ""
print(json.dumps({"status": "done", "tag": tag, "elapsed": round(time.monotonic() - start, 2),
                  "usage": result["usage"], "timings": result.get("timings"),
                  "finish": choice["finish_reason"],
                  "markers": {v: v in content for v in ("4827", "7931", "1648")}}, ensure_ascii=False))
