from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import psutil

from .config import AppConfig
from .types import BackendKind, Transcript, TranscriptionRequest


@dataclass(slots=True)
class ResourceSample:
    elapsed_seconds: float
    cpu_percent: float
    memory_mib: float
    virtual_memory_mib: float
    process_count: int


@dataclass(slots=True)
class ResourceSummary:
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mib: float
    peak_memory_mib: float
    avg_virtual_memory_mib: float
    peak_virtual_memory_mib: float
    peak_process_count: int


@dataclass(slots=True)
class ProfileReport:
    backend: str
    model: str
    elapsed_seconds: float
    audio_duration_seconds: float
    realtime_factor: float | None
    transcript: Transcript
    samples: list[ResourceSample]
    summary: ResourceSummary


def profile_request(
    config: AppConfig,
    request: TranscriptionRequest,
    sample_interval_seconds: float,
) -> ProfileReport:
    if request.backend is BackendKind.CT2_PYTHON:
        return profile_ct2_request(config, request, sample_interval_seconds)
    if request.backend is BackendKind.FPGA_SIM:
        raise RuntimeError(
            "system profiling is not implemented for the fpga-sim backend yet"
        )
    if request.backend is BackendKind.FPGA_HYBRID:
        raise RuntimeError(
            "system profiling is not implemented for the fpga-hybrid backend yet"
        )
    raise RuntimeError(f"unsupported backend for profiling: {request.backend}")


def profile_ct2_request(
    config: AppConfig,
    request: TranscriptionRequest,
    sample_interval_seconds: float,
) -> ProfileReport:
    command = build_worker_command(config, request)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(config.uv_cache_dir))

    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=config.project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stop_event = threading.Event()
    samples: list[ResourceSample] = []
    sampler = threading.Thread(
        target=_sampling_loop,
        args=(child.pid, started, sample_interval_seconds, stop_event, samples),
        daemon=True,
    )
    sampler.start()

    stdout, stderr = child.communicate()
    stop_event.set()
    sampler.join(timeout=max(sample_interval_seconds * 2.0, 0.1))

    elapsed_seconds = time.perf_counter() - started
    if child.returncode != 0:
        raise RuntimeError(
            "Python worker exited with non-zero status.\n"
            f"exit_code: {child.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse worker JSON output.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc

    transcript = Transcript.from_dict(payload)
    realtime_factor = (
        elapsed_seconds / transcript.audio_duration_seconds
        if transcript.audio_duration_seconds > 0.0
        else None
    )
    summary = summarize_samples(samples)

    return ProfileReport(
        backend=transcript.backend,
        model=transcript.model,
        elapsed_seconds=elapsed_seconds,
        audio_duration_seconds=transcript.audio_duration_seconds,
        realtime_factor=realtime_factor,
        transcript=transcript,
        samples=samples,
        summary=summary,
    )


def build_worker_command(
    config: AppConfig,
    request: TranscriptionRequest,
) -> list[str]:
    command = [
        str(config.worker_launcher),
        *config.worker_launcher_args,
        str(config.worker_script_path),
        "--audio",
        str(request.audio_path),
    ]
    if request.initial_prompt:
        command.extend(["--initial-prompt", request.initial_prompt])
    return command


def _sampling_loop(
    root_pid: int,
    started: float,
    sample_interval_seconds: float,
    stop_event: threading.Event,
    samples: list[ResourceSample],
) -> None:
    process: psutil.Process | None = None
    while process is None:
        try:
            process = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            return

    _prime_process_tree_cpu(process)

    interval = max(sample_interval_seconds, 0.05)
    while not stop_event.is_set():
        time.sleep(interval)
        sample = collect_sample(process, time.perf_counter() - started)
        if sample is not None:
            samples.append(sample)

    final_sample = collect_sample(process, time.perf_counter() - started)
    if final_sample is not None:
        samples.append(final_sample)


def collect_sample(
    root_process: psutil.Process,
    elapsed_seconds: float,
) -> ResourceSample | None:
    processes = collect_process_tree(root_process)
    if not processes:
        return None

    total_cpu = 0.0
    total_memory = 0
    total_virtual_memory = 0
    process_count = 0

    for process in processes:
        try:
            total_cpu += float(process.cpu_percent(interval=None))
            memory_info = process.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        total_memory += int(memory_info.rss)
        total_virtual_memory += int(memory_info.vms)
        process_count += 1

    if process_count == 0:
        return None

    return ResourceSample(
        elapsed_seconds=elapsed_seconds,
        cpu_percent=total_cpu,
        memory_mib=bytes_to_mib(total_memory),
        virtual_memory_mib=bytes_to_mib(total_virtual_memory),
        process_count=process_count,
    )


def collect_process_tree(root_process: psutil.Process) -> list[psutil.Process]:
    try:
        processes = [root_process]
        processes.extend(root_process.children(recursive=True))
        return processes
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _prime_process_tree_cpu(root_process: psutil.Process) -> None:
    for process in collect_process_tree(root_process):
        try:
            process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def summarize_samples(samples: list[ResourceSample]) -> ResourceSummary:
    if not samples:
        return ResourceSummary(
            avg_cpu_percent=0.0,
            peak_cpu_percent=0.0,
            avg_memory_mib=0.0,
            peak_memory_mib=0.0,
            avg_virtual_memory_mib=0.0,
            peak_virtual_memory_mib=0.0,
            peak_process_count=0,
        )

    return ResourceSummary(
        avg_cpu_percent=sum(sample.cpu_percent for sample in samples) / len(samples),
        peak_cpu_percent=max(sample.cpu_percent for sample in samples),
        avg_memory_mib=sum(sample.memory_mib for sample in samples) / len(samples),
        peak_memory_mib=max(sample.memory_mib for sample in samples),
        avg_virtual_memory_mib=(
            sum(sample.virtual_memory_mib for sample in samples) / len(samples)
        ),
        peak_virtual_memory_mib=max(sample.virtual_memory_mib for sample in samples),
        peak_process_count=max(sample.process_count for sample in samples),
    )


def bytes_to_mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def render_summary_table(report: ProfileReport) -> str:
    headers = [
        "backend",
        "audio_s",
        "elapsed_s",
        "rtf",
        "avg_cpu_%",
        "peak_cpu_%",
        "avg_ram_mib",
        "peak_ram_mib",
        "peak_vram_mib",
        "peak_procs",
    ]
    row = [
        report.backend,
        f"{report.audio_duration_seconds:.3f}",
        f"{report.elapsed_seconds:.3f}",
        f"{report.realtime_factor:.3f}x"
        if report.realtime_factor is not None
        else "n/a",
        f"{report.summary.avg_cpu_percent:.1f}",
        f"{report.summary.peak_cpu_percent:.1f}",
        f"{report.summary.avg_memory_mib:.1f}",
        f"{report.summary.peak_memory_mib:.1f}",
        f"{report.summary.peak_virtual_memory_mib:.1f}",
        str(report.summary.peak_process_count),
    ]
    return _render_table(headers, [row])


def render_samples_table(report: ProfileReport) -> str:
    headers = ["t_s", "cpu_%", "ram_mib", "virt_mib", "procs"]
    rows = [
        [
            f"{sample.elapsed_seconds:.3f}",
            f"{sample.cpu_percent:.1f}",
            f"{sample.memory_mib:.1f}",
            f"{sample.virtual_memory_mib:.1f}",
            str(sample.process_count),
        ]
        for sample in report.samples
    ]
    return _render_table(headers, rows)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in widths)

    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)
