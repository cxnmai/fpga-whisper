#!/usr/bin/env python3
"""Skeleton worker for the future CTranslate2 baseline path.

This file is intentionally dependency-free for now. The Rust frontend can
eventually send JSON requests here and receive JSON transcripts back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder CTranslate2 worker.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = {
        "backend": "ct2-python",
        "model": args.model,
        "notes": [
            "Stub worker only. Install CTranslate2 and load the Whisper model here.",
            f"Audio input: {args.audio}",
        ],
        "segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "text": "[skeleton] worker is not wired to CTranslate2 yet.",
            }
        ],
    }
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
