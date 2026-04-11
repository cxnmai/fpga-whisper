from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..fpga.kernels.logmel import (
    MEL_BINS,
    quantize_mel_filterbank,
    software_logmel_frame,
)
from ..fpga.transport import MelFrameBatchRequest
from ..types import (
    BackendDescriptor,
    BackendKind,
    PartitionPreset,
    PipelineStage,
    Transcript,
    TranscriptSegment,
    TranscriptionRequest,
)
from ..worker import build_ct2_worker_features_command, build_worker_env

MODEL_HF_REPO = "distil-whisper/distil-small.en"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
N_FFT = 400
HOP_LENGTH = 160
MEL_FLOOR = 1e-10
POWER_BIN_BITS = 24
POWER_BIN_QMAX = (1 << POWER_BIN_BITS) - 1


def describe_backend() -> BackendDescriptor:
    return BackendDescriptor(
        id=BackendKind.FPGA_HW,
        summary="Real Arty S7 frontend backend over UART. Uses board-side log-mel and host CT2.",
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
class FpgaHardwareBackend:
    config: AppConfig
    io_dir: Path
    executor: object | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: AppConfig) -> "FpgaHardwareBackend":
        return cls(config=config, io_dir=config.resolved_fpga_hw_io_dir)

    def descriptor(self) -> BackendDescriptor:
        return describe_backend()

    def _get_executor(self):
        if self.executor is None:
            from ..fpga.uart import FpgaUartExecutor

            self.executor = FpgaUartExecutor(
                port=self.config.fpga_uart_port,
                baud=self.config.fpga_uart_baud,
                timeout=self.config.fpga_uart_timeout_seconds,
            )
        return self.executor

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        try:
            from faster_whisper.audio import decode_audio
            from transformers.audio_utils import (
                mel_filter_bank,
                spectrogram,
                window_function,
            )
        except ImportError as exc:
            return Transcript(
                backend=BackendKind.FPGA_HW.value,
                model=MODEL_HF_REPO,
                audio_duration_seconds=0.0,
                notes=[
                    "Python dependencies are incomplete for the real FPGA frontend path.",
                    f"Import failure: {exc}",
                ],
                segments=[],
            )

        try:
            executor = self._get_executor()
        except Exception as exc:
            return Transcript(
                backend=BackendKind.FPGA_HW.value,
                model=MODEL_HF_REPO,
                audio_duration_seconds=0.0,
                notes=[
                    "Failed to open the FPGA UART transport.",
                    f"Transport error: {exc}",
                    f"Configured port: {self.config.fpga_uart_port}",
                    f"Configured baud: {self.config.fpga_uart_baud}",
                ],
                segments=[],
            )

        audio = decode_audio(str(request.audio_path), sampling_rate=SAMPLE_RATE)
        if len(audio) == 0:
            return Transcript(
                backend=BackendKind.FPGA_HW.value,
                model=MODEL_HF_REPO,
                audio_duration_seconds=0.0,
                notes=[
                    f"Decoded zero samples from {request.audio_path}.",
                    "No transcription was attempted.",
                ],
                segments=[],
            )

        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + (N_FFT // 2),
            num_mel_filters=MEL_BINS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2.0,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).astype(np.float32)
        mel_coefficients = quantize_mel_filterbank(mel_filters)
        window = window_function(N_FFT, "hann")

        segments: list[TranscriptSegment] = []
        notes = [
            "Python host is talking directly to the real Arty S7 UART frontend boundary.",
            f"Executor: {executor.name()}",
            "Feature source: host STFT/power spectrum, FPGA q8.8 log-mel, host Whisper log/clamp normalization.",
            f"Power quantization: per-frame unsigned {POWER_BIN_BITS}-bit bins before UART mel accumulation.",
            f"Requested partition: {request.partition.value}",
            f"Planned FPGA stages: {', '.join(stage.label for stage in request.partition.stages_on_fpga) or '(none)'}",
        ]
        if request.partition is not PartitionPreset.FRONTEND:
            notes.append(
                "Real hardware backend currently executes the frontend only; wider partitions remain planned.",
            )

        for chunk_index, start in enumerate(range(0, len(audio), CHUNK_SAMPLES)):
            stop = min(len(audio), start + CHUNK_SAMPLES)
            chunk = np.asarray(audio[start:stop], dtype=np.float32)
            if chunk.size == 0:
                continue

            power_spectrogram = spectrogram(
                chunk,
                window,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH,
                power=2.0,
            )
            power_spectrogram = np.asarray(power_spectrogram[:, :-1], dtype=np.float32)
            if power_spectrogram.size == 0:
                continue

            frames = power_spectrogram.shape[1]
            power_scale = (
                np.max(power_spectrogram, axis=0).astype(np.float64) / POWER_BIN_QMAX
            )
            power_scale = np.where(power_scale > 0.0, power_scale, 1.0)
            power_quantized = np.clip(
                np.rint(power_spectrogram / power_scale[np.newaxis, :]),
                0,
                POWER_BIN_QMAX,
            ).astype(np.uint32)

            expected_rows: list[np.ndarray] = []
            for frame_index in range(frames):
                _, expected_log = software_logmel_frame(
                    power_quantized[:, frame_index],
                    mel_coefficients,
                )
                expected_rows.append(expected_log.astype(np.int64, copy=False))

            response = executor.execute_mel_frame_batch(
                MelFrameBatchRequest(
                    audio_path=str(request.audio_path),
                    frame_count=frames,
                    power_frames=power_quantized.T.reshape(-1)
                    .astype(np.int64, copy=False)
                    .tolist(),
                    mel_coefficients=mel_coefficients.T.reshape(-1)
                    .astype(np.int64, copy=False)
                    .tolist(),
                    expected_output=np.concatenate(expected_rows)
                    .astype(np.int64, copy=False)
                    .tolist(),
                ),
                self.io_dir,
            )
            if not response.matched:
                raise RuntimeError(
                    f"mel frame batch mismatch for chunk {chunk_index + 1}: FPGA frontend diverged from host reference"
                )

            mel_log_q8_8 = np.asarray(response.rtl_output, dtype=np.float64).reshape(
                frames, MEL_BINS
            )
            log_spec = (mel_log_q8_8 / 256.0) * np.log10(2.0)
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            features = ((log_spec + 4.0) / 4.0).T.astype(np.float32, copy=False)
            features_path = self._chunk_features_path(chunk_index)
            features_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(features_path, features[np.newaxis, ...])

            worker_transcript = self._transcribe_features(
                features_path=features_path,
                audio_duration_seconds=(stop - start) / SAMPLE_RATE,
                initial_prompt=request.initial_prompt if chunk_index == 0 else None,
            )
            notes.extend(
                note
                for note in worker_transcript.notes
                if note not in notes and not note.startswith("Feature source:")
            )
            for segment in worker_transcript.segments:
                segments.append(
                    TranscriptSegment(
                        start_seconds=segment.start_seconds + (start / SAMPLE_RATE),
                        end_seconds=segment.end_seconds + (start / SAMPLE_RATE),
                        text=segment.text,
                    )
                )

        return Transcript(
            backend=BackendKind.FPGA_HW.value,
            model=MODEL_HF_REPO,
            audio_duration_seconds=len(audio) / SAMPLE_RATE,
            notes=notes,
            segments=segments,
        )

    def _transcribe_features(
        self,
        *,
        features_path: Path,
        audio_duration_seconds: float,
        initial_prompt: str | None,
    ) -> Transcript:
        command = build_ct2_worker_features_command(
            self.config,
            features_path=features_path,
            audio_duration_seconds=audio_duration_seconds,
            initial_prompt=initial_prompt,
        )
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
                "Python worker exited with a non-zero status while transcribing FPGA-generated features.\n"
                f"status: {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "failed to parse worker JSON output for FPGA-generated features.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ) from exc
        return Transcript.from_dict(payload)

    def _chunk_features_path(self, chunk_index: int) -> Path:
        return self.io_dir / f"hardware_features_chunk_{chunk_index:04d}.npy"
