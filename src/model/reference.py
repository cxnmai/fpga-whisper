from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ReferenceActivationExport:
    model_repo: str
    audio_path: str
    layer_name: str
    sequence_length: int
    exported_positions: int
    hidden_size: int
    activations: list[NDArray[np.float32]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceActivationExport":
        return cls(
            model_repo=str(data["model_repo"]),
            audio_path=str(data["audio_path"]),
            layer_name=str(data["layer_name"]),
            sequence_length=int(data["sequence_length"]),
            exported_positions=int(data["exported_positions"]),
            hidden_size=int(data["hidden_size"]),
            activations=[
                np.asarray(row, dtype=np.float32)
                for row in data.get("activations", [])
            ],
        )


def ensure_reference_activation_export(
    config: Any,
    audio_path: str | Path,
    refresh: bool,
) -> Path:
    audio_path = Path(audio_path)
    output_path = Path(config.reference_activation_cache_path(audio_path))
    if output_path.exists() and not refresh:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(getattr(config, "project_root", "."))
    module_name = getattr(
        config,
        "reference_exporter_module",
        "src.scripts.export_reference_activation",
    )
    positions = int(getattr(config, "reference_export_positions", 4))

    command = [
        sys.executable,
        "-m",
        module_name,
        "--audio",
        str(audio_path),
        "--positions",
        str(positions),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "activation exporter exited with a non-zero status.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return output_path


def load_reference_activation(path: str | Path) -> ReferenceActivationExport:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse {path}") from exc

    return ReferenceActivationExport.from_dict(data)
