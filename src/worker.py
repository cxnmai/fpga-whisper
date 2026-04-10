from __future__ import annotations

import os
import sys
from typing import Any


def build_ct2_worker_command(config: Any, request: Any) -> list[str]:
    command = [
        sys.executable,
        str(config.worker_script_path),
        "--audio",
        str(request.audio_path),
    ]
    if request.initial_prompt:
        command.extend(["--initial-prompt", request.initial_prompt])
    return command


def build_worker_env(config: Any) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(config.uv_cache_dir))
    return env
