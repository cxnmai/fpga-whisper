from __future__ import annotations

from ..config import AppConfig
from ..types import BackendDescriptor, BackendKind, PartitionPreset, PipelineStage
from .base import TranscriptionBackend
from .ct2_python import Ct2PythonBackend
from .fpga_hybrid import FpgaHybridBackend
from .fpga_sim import FpgaSimBackend


def build_backend(kind: BackendKind, config: AppConfig) -> TranscriptionBackend:
    if kind is BackendKind.CT2_PYTHON:
        return Ct2PythonBackend(config)
    if kind is BackendKind.FPGA_SIM:
        return FpgaSimBackend.from_config(config)
    if kind is BackendKind.FPGA_HYBRID:
        return FpgaHybridBackend()
    raise ValueError(f"unsupported backend kind: {kind}")


def describe_backend(kind: BackendKind) -> BackendDescriptor:
    if kind is BackendKind.CT2_PYTHON:
        return BackendDescriptor(
            id=BackendKind.CT2_PYTHON,
            summary="Host-side CTranslate2 baseline. Use this as the correctness oracle.",
            partition=PartitionPreset.CPU_ONLY,
            host_stages=[
                PipelineStage.AUDIO_DECODE,
                PipelineStage.FEATURE_EXTRACTION,
                PipelineStage.ENCODER,
                PipelineStage.DECODER_MATH,
                PipelineStage.DECODE_POLICY,
                PipelineStage.POST_PROCESS,
            ],
            fpga_stages=[],
        )

    if kind is BackendKind.FPGA_SIM:
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

    if kind is BackendKind.FPGA_HYBRID:
        return BackendDescriptor(
            id=BackendKind.FPGA_HYBRID,
            summary="Hybrid path. Host keeps control flow while FPGA absorbs dense math stages.",
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

    raise ValueError(f"unsupported backend kind: {kind}")


__all__ = [
    "TranscriptionBackend",
    "Ct2PythonBackend",
    "FpgaSimBackend",
    "FpgaHybridBackend",
    "build_backend",
    "describe_backend",
]
