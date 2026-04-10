from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

from .backends import build_backend, describe_backend
from .config import default_config
from .profiling import profile_request, render_samples_table, render_summary_table
from .types import (
    BackendKind,
    BenchmarkRun,
    PartitionPreset,
    Transcript,
    TranscriptionRequest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpga-whisper",
        description="Python frontend for a hybrid Whisper + FPGA transcription project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Describe a backend and partition plan.")
    plan.add_argument(
        "--backend",
        choices=[item.value for item in BackendKind],
        default=BackendKind.FPGA_HYBRID.value,
    )
    plan.add_argument(
        "--partition",
        choices=[item.value for item in PartitionPreset],
        default=PartitionPreset.HYBRID.value,
    )

    transcribe = subparsers.add_parser(
        "transcribe", help="Run a transcription backend."
    )
    _add_transcription_args(transcribe)

    benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark a transcription backend."
    )
    _add_transcription_args(benchmark)
    benchmark.add_argument("--iterations", type=int, default=3)
    benchmark.add_argument("--warmup", type=int, default=1)

    profile = subparsers.add_parser(
        "profile", help="Profile backend system resource usage."
    )
    _add_transcription_args(profile)
    profile.add_argument("--sample-interval-ms", type=int, default=250)

    subparsers.add_parser("gemm-check", help="Validate the FPGA GEMM tile path.")
    subparsers.add_parser("linear-check", help="Validate the FPGA linear layer path.")
    subparsers.add_parser(
        "projection-tile-check", help="Validate one projection tile from the model."
    )
    subparsers.add_parser(
        "projection-sweep-check",
        help="Sweep multiple projection tiles from cached reference activations.",
    )
    subparsers.add_parser(
        "projection-full-check",
        help="Validate an accumulated full-width projection slice.",
    )
    subparsers.add_parser(
        "projection-full-sweep-check",
        help="Sweep multiple accumulated full-width projection slices.",
    )
    subparsers.add_parser("gelu-check", help="Validate the FPGA GELU block path.")
    subparsers.add_parser(
        "gelu-sweep-check",
        help="Sweep the FPGA GELU block across cached projection windows.",
    )

    return parser


def _add_transcription_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--backend",
        choices=[item.value for item in BackendKind],
        default=BackendKind.CT2_PYTHON.value,
    )
    parser.add_argument(
        "--partition",
        choices=[item.value for item in PartitionPreset],
        default=PartitionPreset.HYBRID.value,
    )
    parser.add_argument("--initial-prompt")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = default_config()

    command = args.command
    if command == "plan":
        return _run_plan(args)
    if command == "transcribe":
        return _run_transcribe(config, args)
    if command == "benchmark":
        return _run_benchmark(config, args)
    if command == "profile":
        return _run_profile(config, args)
    if command == "gemm-check":
        return _run_fpga_command("run_gemm_check", config)
    if command == "linear-check":
        return _run_fpga_command("run_linear_check", config)
    if command == "projection-tile-check":
        return _run_fpga_command("run_projection_tile_check", config)
    if command == "projection-sweep-check":
        return _run_fpga_command("run_projection_sweep_check", config)
    if command == "projection-full-check":
        return _run_fpga_command("run_projection_full_check", config)
    if command == "projection-full-sweep-check":
        return _run_fpga_command("run_projection_full_sweep_check", config)
    if command == "gelu-check":
        return _run_fpga_command("run_gelu_check", config)
    if command == "gelu-sweep-check":
        return _run_fpga_command("run_gelu_sweep_check", config)

    parser.error(f"unsupported command: {command}")
    return 2


def _run_plan(args: argparse.Namespace) -> int:
    backend = BackendKind(args.backend)
    partition = PartitionPreset(args.partition)
    descriptor = describe_backend(backend)
    print(descriptor)
    print()
    print(f"Selected partition preset: {partition.value}")
    print(f"Partition summary: {partition.summary}")
    return 0


def _run_transcribe(config, args: argparse.Namespace) -> int:
    request = TranscriptionRequest(
        audio_path=args.audio,
        backend=BackendKind(args.backend),
        partition=PartitionPreset(args.partition),
        initial_prompt=args.initial_prompt,
    )
    backend = build_backend(request.backend, config)
    transcript = backend.transcribe(request)
    print_transcript(transcript)
    return 0


def _run_benchmark(config, args: argparse.Namespace) -> int:
    request = TranscriptionRequest(
        audio_path=args.audio,
        backend=BackendKind(args.backend),
        partition=PartitionPreset(args.partition),
        initial_prompt=args.initial_prompt,
    )
    backend = build_backend(request.backend, config)
    report = benchmark_backend(
        backend=backend,
        request=request,
        iterations=max(1, args.iterations),
        warmup=max(0, args.warmup),
    )
    print_benchmark(report)
    return 0


def _run_profile(config, args: argparse.Namespace) -> int:
    request = TranscriptionRequest(
        audio_path=args.audio,
        backend=BackendKind(args.backend),
        partition=PartitionPreset(args.partition),
        initial_prompt=args.initial_prompt,
    )
    report = profile_request(
        config=config,
        request=request,
        sample_interval_seconds=max(args.sample_interval_ms, 1) / 1000.0,
    )
    print(render_summary_table(report))
    print()
    print(render_samples_table(report))
    print()
    print_transcript(report.transcript)
    return 0


def _run_fpga_command(function_name: str, config) -> int:
    try:
        from . import fpga_cli
    except ImportError as exc:
        raise SystemExit(
            "The FPGA command layer is not available yet in the Python port.\n"
            f"Missing import while loading fpga CLI module: {exc}"
        ) from exc

    handler = getattr(fpga_cli, function_name, None)
    if handler is None:
        raise SystemExit(
            f"The FPGA command '{function_name}' is not implemented in the Python port yet."
        )

    handler(config)
    return 0


def print_transcript(transcript: Transcript) -> None:
    print(f"backend: {transcript.backend}")
    print(f"model: {transcript.model}")
    print(f"audio_duration_seconds: {transcript.audio_duration_seconds:.3f}")
    for note in transcript.notes:
        print(f"note: {note}")
    for segment in transcript.segments:
        print(
            f"[{segment.start_seconds:.2f}..{segment.end_seconds:.2f}] {segment.text}"
        )


def benchmark_backend(
    backend,
    request: TranscriptionRequest,
    iterations: int,
    warmup: int,
):
    last_transcript: Transcript | None = None

    for _ in range(warmup):
        last_transcript = backend.transcribe(request)

    runs: list[BenchmarkRun] = []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        transcript = backend.transcribe(request)
        elapsed_seconds = time.perf_counter() - started
        realtime_factor = (
            elapsed_seconds / transcript.audio_duration_seconds
            if transcript.audio_duration_seconds > 0.0
            else None
        )
        runs.append(
            BenchmarkRun(
                iteration=iteration,
                elapsed_seconds=elapsed_seconds,
                realtime_factor=realtime_factor,
            )
        )
        last_transcript = transcript

    if last_transcript is None:
        raise RuntimeError("benchmark must have at least one run")

    return BenchmarkReport(
        warmup_runs=warmup,
        measured_runs=runs,
        transcript=last_transcript,
    )


class BenchmarkReport:
    def __init__(
        self,
        *,
        warmup_runs: int,
        measured_runs: list[BenchmarkRun],
        transcript: Transcript,
    ) -> None:
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.transcript = transcript


def print_benchmark(report: BenchmarkReport) -> None:
    print(f"backend: {report.transcript.backend}")
    print(f"model: {report.transcript.model}")
    print(f"audio_duration_seconds: {report.transcript.audio_duration_seconds:.3f}")
    print(f"warmup_runs: {report.warmup_runs}")
    print(f"measured_runs: {len(report.measured_runs)}")

    for run in report.measured_runs:
        if run.realtime_factor is not None:
            print(
                f"run {run.iteration}: {run.elapsed_seconds:.3f}s "
                f"(rtf {run.realtime_factor:.3f}x)"
            )
        else:
            print(f"run {run.iteration}: {run.elapsed_seconds:.3f}s")

    elapsed_values = [run.elapsed_seconds for run in report.measured_runs]
    avg_seconds = statistics.mean(elapsed_values)
    min_seconds = min(elapsed_values)
    max_seconds = max(elapsed_values)

    print(f"avg_seconds: {avg_seconds:.3f}")
    print(f"min_seconds: {min_seconds:.3f}")
    print(f"max_seconds: {max_seconds:.3f}")

    if report.transcript.audio_duration_seconds > 0.0:
        avg_rtf = avg_seconds / report.transcript.audio_duration_seconds
        print(f"avg_realtime_factor: {avg_rtf:.3f}x")

    print("transcript_preview:")
    for segment in report.transcript.segments:
        print(
            f"[{segment.start_seconds:.2f}..{segment.end_seconds:.2f}] {segment.text}"
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
