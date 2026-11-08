#!/usr/bin/env python3
"""Validate published benchmark records and summarize long-context groups."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    "window4-default": ["baseline", "baselineagain", "baselinefinal"],
    "window4-p0.3": ["spec4p03", "spec4p03repeat", "spec4p03final"],
    "window5-p0.3": ["spec5p03b", "spec5p03c", "spec5p03d"],
    "window5-p0.5": ["spec5p05a", "spec5p05b", "spec5p05c", "spec5p05d"],
    "window6-default": ["spec6"],
}

def main():
    records = [json.loads(line) for line in (ROOT / "data/mtp-long.jsonl").read_text().splitlines() if line.strip()]
    by_tag = {}
    for record in records:
        tag = record["tag"]
        if tag in by_tag:
            raise ValueError(f"duplicate tag: {tag}")
        by_tag[tag] = record
        if record["status"] == "done":
            assert record["usage"]["completion_tokens"] == 512
            assert record["timings"]["predicted_per_second"] > 0
            assert set(record["markers"]) == {"4827", "7931", "1648"}
            assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        elif record["status"] != "error":
            raise ValueError(f"unknown status: {record['status']}")
    for label, tags in GROUPS.items():
        selected = [by_tag[tag] for tag in tags]
        assert all(row["status"] == "done" for row in selected)
        speeds = [row["timings"]["predicted_per_second"] for row in selected]
        markers = sum(all(row["markers"].values()) for row in selected)
        print(f"{label}: n={len(speeds)} mean={statistics.mean(speeds):.2f} median={statistics.median(speeds):.2f} range={min(speeds):.1f}–{max(speeds):.1f} markers={markers}/{len(speeds)}")
    short = ROOT / "data/short-concurrency.jsonl"
    short_rows = [json.loads(line) for line in short.read_text().splitlines() if line.strip()]
    assert all(row["phase"] in ("solo", "pair") for row in short_rows)
    print(f"validated {len(records)} long records including {sum(r['status'] == 'error' for r in records)} error(s); {len(short_rows)} short records")

if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
