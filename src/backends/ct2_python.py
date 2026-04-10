from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from ..config import MODEL_CT2_ALIAS, MODEL_HF_REPO, AppConfig
from ..types import (
    BackendDescriptor,
    BackendKind,
    Transcript,
    TranscriptionRequest,
)
from ..worker import build_ct2_worker_command, build_worker_env


@dataclass(slots=True)
class Ct2PythonBackend:
    config: AppConfig

    def descriptor(self) -> BackendDescriptor:
        from . import describe_backend

        return describe_backend(BackendKind.CT2_PYTHON)

    def build_worker_command(self, request: TranscriptionRequest) -> list[str]:
        return build_ct2_worker_command(self.config, request)

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        command = self.build_worker_command(request)
        env = build_worker_env(self.config)

        completed = subprocess.run(
            command,
            cwd=self.config.project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Python worker exited with a non-zero status.\n"
                f"status: {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "failed to parse worker JSON output.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ) from exc

        transcript = Transcript.from_dict(payload)
        transcript.model = MODEL_HF_REPO
        if all("Model alias:" not in note for note in transcript.notes):
            insert_at = 1 if transcript.notes else 0
            transcript.notes.insert(insert_at, f"Model alias: {MODEL_CT2_ALIAS}")
        return transcript
