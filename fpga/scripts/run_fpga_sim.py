#!/usr/bin/env python3
"""Bridge script for the Rust -> RTL simulation scaffold.

This is intentionally file-based and conservative. Today it only acknowledges
the request and emits placeholder feature metadata. Later it can compile and run
iverilog/vvp, then package the resulting vectors back to Rust.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FPGA simulator bridge.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text())

    response = {
        "produced_stage": "feature-extraction",
        "sample_rate_hz": 16000,
        "frame_count": 3000,
        "bin_count": 80,
        "notes": [
            "Skeleton simulator bridge only. No RTL execution yet.",
            f"Received audio path: {request['audio_path']}",
            f"Requested stage: {request['requested_stage']}",
            "Next step: compile feature_stage_stub_tb.v with iverilog and replace this placeholder response with parsed simulator outputs.",
        ],
    }

    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
