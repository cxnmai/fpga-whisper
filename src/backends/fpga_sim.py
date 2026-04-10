from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from ..fpga.kernels.dot import simulate_dot_product
from ..fpga.sim import IverilogSimExecutor
from ..types import (
    BackendDescriptor,
    BackendKind,
    PartitionPreset,
    PipelineStage,
    Transcript,
    TranscriptionRequest,
    TranscriptSegment,
)

MODEL_HF_REPO = "distil-whisper/distil-small.en"


def describe_backend() -> BackendDescriptor:
    return BackendDescriptor(
        id=BackendKind.FPGA_SIM,
        summary="Host-side integration path for simulated RTL via file-based vector exchange.",
        partition=PartitionPreset.FRONTEND,
        host_stages=[
            PipelineStage.AUDIO_DECODE,
            PipelineStage.ENCODER,
            PipelineStage.DECODER_MATH,
            PipelineStage.DECODE_POLICY,
            PipelineStage.POST_PROCESS,
        ],
        fpga_stages=[PipelineStage.FEATURE_EXTRACTION],
    )


@dataclass(slots=True)
class FpgaSimBackend:
    project_root: Path
    io_dir: Path
    executor: IverilogSimExecutor

    @classmethod
    def from_config(cls, config: AppConfig) -> "FpgaSimBackend":
        return cls(
            project_root=config.project_root,
            io_dir=config.resolved_fpga_sim_io_dir,
            executor=IverilogSimExecutor(project_root=config.project_root),
        )

    def descriptor(self) -> BackendDescriptor:
        return describe_backend()

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        fpga_stages = (
            ", ".join(stage.label for stage in request.partition.stages_on_fpga)
            or "(none)"
        )

        vector_a = [3, -2, 7, 4, -1, 5, 2, -3]
        vector_b = [6, 8, -4, 1, 9, -2, 3, 5]

        response = simulate_dot_product(
            executor=self.executor,
            output_dir=self.io_dir,
            audio_path=str(request.audio_path),
            lhs=vector_a,
            rhs=vector_b,
        )

        notes = [
            "Python host is talking directly to the simulated RTL boundary.",
            f"Executor: {self.executor.name()}",
            f"Planned FPGA stages: {fpga_stages}",
            "First real RTL primitive: signed 8-lane int16 dot product.",
            f"Input vector A: {vector_a!r}",
            f"Input vector B: {vector_b!r}",
            f"Expected software result: {response.expected_result}",
            f"RTL result: {response.rtl_result}",
            f"Matched: {response.matched}",
            *response.notes,
        ]

        return Transcript(
            backend="fpga-sim",
            model=MODEL_HF_REPO,
            audio_duration_seconds=0.0,
            notes=notes,
            segments=[
                TranscriptSegment(
                    start_seconds=0.0,
                    end_seconds=0.0,
                    text=f"[fpga-sim] dot-product smoke test result = {response.rtl_result}",
                )
            ],
        )
