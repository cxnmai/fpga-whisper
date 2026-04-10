from __future__ import annotations

from dataclasses import dataclass

from ..types import (
    BackendDescriptor,
    BackendKind,
    PartitionPreset,
    PipelineStage,
    Transcript,
    TranscriptionRequest,
    TranscriptSegment,
)


@dataclass(slots=True)
class FpgaHybridBackend:
    """Hybrid backend scaffold.

    This keeps the host-side control flow in Python while documenting the
    intended FPGA ownership boundary for future integration.
    """

    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            id=BackendKind.FPGA_HYBRID,
            summary=(
                "Hybrid path. Host keeps control flow while FPGA absorbs dense "
                "math stages."
            ),
            partition=PartitionPreset.HYBRID,
            host_stages=[
                PipelineStage.AUDIO_DECODE,
                PipelineStage.DECODE_POLICY,
                PipelineStage.POST_PROCESS,
            ],
            fpga_stages=[
                PipelineStage.FEATURE_EXTRACTION,
                PipelineStage.ENCODER,
                PipelineStage.DECODER_MATH,
            ],
        )

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        fpga_stages = ", ".join(
            stage.label for stage in request.partition.stages_on_fpga
        )

        return Transcript(
            backend=BackendKind.FPGA_HYBRID.value,
            model="distil-whisper/distil-small.en",
            audio_duration_seconds=0.0,
            notes=[
                request.partition.summary,
                f"Planned FPGA stages: {fpga_stages or '(none)'}",
                (
                    "Next implementation step: connect the host runtime to an FPGA "
                    "transport and keep weights resident on-board during chunk "
                    "execution."
                ),
            ],
            segments=[
                TranscriptSegment(
                    start_seconds=0.0,
                    end_seconds=0.0,
                    text="[skeleton] FPGA hybrid runtime is not wired yet.",
                )
            ],
        )
