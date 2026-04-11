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
from .backends import build_backend
from .worker import build_ct2_worker_command, build_worker_env


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
    if request.backend in {
        BackendKind.FPGA_SIM,
        BackendKind.FPGA_HW,
        BackendKind.FPGA_HYBRID,
    }:
        return profile_backend_request(config, request, sample_interval_seconds)
    raise RuntimeError(f"unsupported backend for profiling: {request.backend}")


def profile_backend_request(
    config: AppConfig,
    request: TranscriptionRequest,
    sample_interval_seconds: float,
) -> ProfileReport:
    backend = build_backend(request.backend, config)

    started = time.perf_counter()
    result: dict[str, Transcript] = {}
    failure: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            result["transcript"] = backend.transcribe(request)
        except BaseException as exc:  # pragma: no cover - surfaced in main thread
            failure["exc"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    stop_event = threading.Event()
    samples: list[ResourceSample] = []
    tracker = ProcessTreeTracker(os.getpid())
    sampler = threading.Thread(
        target=_sampling_loop,
        args=(tracker, started, sample_interval_seconds, stop_event, samples),
        daemon=True,
    )
    sampler.start()

    worker.join()
    stop_event.set()
    sampler.join(timeout=max(sample_interval_seconds * 2.0, 0.1))

    elapsed_seconds = time.perf_counter() - started
    if failure:
        raise failure["exc"]

    transcript = result["transcript"]
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


def profile_ct2_request(
    config: AppConfig,
    request: TranscriptionRequest,
    sample_interval_seconds: float,
) -> ProfileReport:
    command = build_ct2_worker_command(config, request)
    env = build_worker_env(config)

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
    tracker = ProcessTreeTracker(child.pid)
    sampler = threading.Thread(
        target=_sampling_loop,
        args=(tracker, started, sample_interval_seconds, stop_event, samples),
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


@dataclass(slots=True)
class ProcessTreeTracker:
    root_pid: int
    handles: dict[int, psutil.Process] | None = None

    def __post_init__(self) -> None:
        self.handles = {}

    def sample(self, elapsed_seconds: float) -> ResourceSample | None:
        processes = self._refresh_process_tree()
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

    def _refresh_process_tree(self) -> list[psutil.Process]:
        try:
            root_process = self.handles.get(self.root_pid) or psutil.Process(self.root_pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

        try:
            current_processes = [root_process]
            current_processes.extend(root_process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

        refreshed: dict[int, psutil.Process] = {}
        for process in current_processes:
            pid = process.pid
            handle = self.handles.get(pid, process)
            if pid not in self.handles:
                try:
                    handle.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            refreshed[pid] = handle

        self.handles = refreshed
        return list(refreshed.values())


def _sampling_loop(
    tracker: ProcessTreeTracker,
    started: float,
    sample_interval_seconds: float,
    stop_event: threading.Event,
    samples: list[ResourceSample],
) -> None:
    interval = max(sample_interval_seconds, 0.05)
    while not stop_event.is_set():
        time.sleep(interval)
        sample = tracker.sample(time.perf_counter() - started)
        if sample is not None:
            samples.append(sample)

    final_sample = tracker.sample(time.perf_counter() - started)
    if final_sample is not None:
        samples.append(final_sample)


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
