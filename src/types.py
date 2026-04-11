from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

MODEL_HF_REPO = "distil-whisper/distil-small.en"
MODEL_CT2_ALIAS = "distil-small.en"
MODEL_CT2_CACHE_REPO_DIR = "models--Systran--faster-distil-whisper-small.en"


class BackendKind(str, Enum):
    CT2_PYTHON = "ct2-python"
    FPGA_SIM = "fpga-sim"
    FPGA_HW = "fpga-hw"
    FPGA_HYBRID = "fpga-hybrid"

    @property
    def display_name(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, value: str) -> "BackendKind":
        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"unsupported backend: {value}")


class PipelineStage(str, Enum):
    AUDIO_DECODE = "audio decode"
    FEATURE_EXTRACTION = "feature extraction"
    ENCODER = "encoder"
    DECODER_MATH = "decoder math"
    DECODE_POLICY = "decode policy"
    POST_PROCESS = "post-process"

    @property
    def label(self) -> str:
        return self.value


class PartitionPreset(str, Enum):
    CPU_ONLY = "cpu-only"
    FRONTEND = "frontend"
    ENCODER = "encoder"
    HYBRID = "hybrid"

    @property
    def summary(self) -> str:
        return {
            PartitionPreset.CPU_ONLY: "Reference path. All compute remains on the host.",
            PartitionPreset.FRONTEND: "Offload STFT/log-mel to the FPGA and keep the model on the host.",
            PartitionPreset.ENCODER: "Offload front-end plus encoder. Host keeps decoder generation and text logic.",
            PartitionPreset.HYBRID: "Target architecture. FPGA owns dense math blocks while host keeps decode control.",
        }[self]

    @property
    def stages_on_fpga(self) -> tuple[PipelineStage, ...]:
        return {
            PartitionPreset.CPU_ONLY: (),
            PartitionPreset.FRONTEND: (PipelineStage.FEATURE_EXTRACTION,),
            PartitionPreset.ENCODER: (
                PipelineStage.FEATURE_EXTRACTION,
                PipelineStage.ENCODER,
            ),
            PartitionPreset.HYBRID: (
                PipelineStage.FEATURE_EXTRACTION,
                PipelineStage.ENCODER,
                PipelineStage.DECODER_MATH,
            ),
        }[self]

    @classmethod
    def from_value(cls, value: str) -> "PartitionPreset":
        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"unsupported partition preset: {value}")


@dataclass(slots=True)
class TranscriptionRequest:
    audio_path: Path
    backend: BackendKind
    partition: PartitionPreset
    initial_prompt: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionRequest":
        return cls(
            audio_path=Path(data["audio_path"]),
            backend=BackendKind.from_value(str(data["backend"])),
            partition=PartitionPreset.from_value(str(data["partition"])),
            initial_prompt=(
                None
                if data.get("initial_prompt") is None
                else str(data["initial_prompt"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": str(self.audio_path),
            "backend": self.backend.value,
            "partition": self.partition.value,
            "initial_prompt": self.initial_prompt,
        }


@dataclass(slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start_seconds=float(data["start_seconds"]),
            end_seconds=float(data["end_seconds"]),
            text=str(data["text"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "text": self.text,
        }


@dataclass(slots=True)
class Transcript:
    backend: str
    model: str
    audio_duration_seconds: float
    notes: list[str] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            backend=str(data["backend"]),
            model=str(data["model"]),
            audio_duration_seconds=float(data["audio_duration_seconds"]),
            notes=[str(note) for note in data.get("notes", [])],
            segments=[
                TranscriptSegment.from_dict(item) for item in data.get("segments", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "audio_duration_seconds": self.audio_duration_seconds,
            "notes": list(self.notes),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(slots=True)
class BenchmarkRun:
    iteration: int
    elapsed_seconds: float
    realtime_factor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "realtime_factor": self.realtime_factor,
        }


@dataclass(slots=True)
class BackendDescriptor:
    id: BackendKind
    summary: str
    partition: PartitionPreset
    host_stages: list[PipelineStage]
    fpga_stages: list[PipelineStage]

    def format_lines(self) -> list[str]:
        host = ", ".join(stage.label for stage in self.host_stages) or "(none)"
        fpga = ", ".join(stage.label for stage in self.fpga_stages) or "(none)"
        return [
            f"{self.id.display_name}:",
            f"  summary: {self.summary}",
            f"  partition: {self.partition.value}",
            f"  host stages: {host}",
            f"  fpga stages: {fpga}",
        ]

    def __str__(self) -> str:
        return "\n".join(self.format_lines())
