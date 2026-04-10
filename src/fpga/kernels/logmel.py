from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..transport import FpgaExecutor, LogMelFrameRequest

FFT_BINS = 201
MEL_BINS = 80
MEL_COEFF_FRAC_BITS = 12
LOG_OUTPUT_FRAC_BITS = 8
POWER_BIN_BITS = 24


@dataclass(slots=True)
class LogMelFrameComparison:
    power_spectrum: list[int]
    expected_output: list[int]
    rtl_output: list[int]
    expected_dequantized: list[float]
    rtl_dequantized: list[float]
    matched: bool
    notes: list[str]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    n_freqs = (n_fft // 2) + 1
    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float32)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for mel_index in range(n_mels):
        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]
        if center <= left or right <= center:
            continue

        up_slope = (fft_frequencies - left) / (center - left)
        down_slope = (right - fft_frequencies) / (right - center)
        weights[mel_index] = np.maximum(0.0, np.minimum(up_slope, down_slope))

    return weights


def quantize_mel_filterbank(weights: np.ndarray) -> np.ndarray:
    scale = 1 << MEL_COEFF_FRAC_BITS
    quantized = np.rint(weights * scale)
    return np.clip(quantized, 0, 0xFFFF).astype(np.uint16)


def log2_linear_q8_8(value: int) -> int:
    if value <= 0:
        return 0

    exponent = value.bit_length() - 1
    base = 1 << exponent
    remainder = value - base
    fractional = (remainder << LOG_OUTPUT_FRAC_BITS) // base
    return (exponent << LOG_OUTPUT_FRAC_BITS) + fractional


def software_logmel_frame(
    power_spectrum: np.ndarray,
    mel_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accumulators = power_spectrum.astype(np.uint64) @ mel_coefficients.T.astype(np.uint64)
    log_output = np.fromiter(
        (log2_linear_q8_8(int(value)) for value in accumulators),
        dtype=np.uint16,
        count=MEL_BINS,
    )
    return accumulators, log_output


def simulate_logmel_frame(
    *,
    executor: FpgaExecutor,
    output_dir: Path,
    audio_path: str,
    power_spectrum: np.ndarray,
    mel_coefficients: np.ndarray,
) -> LogMelFrameComparison:
    if power_spectrum.shape != (FFT_BINS,):
        raise ValueError(
            f"power spectrum must have shape ({FFT_BINS},), got {power_spectrum.shape!r}"
        )
    if mel_coefficients.shape != (MEL_BINS, FFT_BINS):
        raise ValueError(
            "mel coefficient matrix must have shape "
            f"({MEL_BINS}, {FFT_BINS}), got {mel_coefficients.shape!r}"
        )

    _, expected_output = software_logmel_frame(power_spectrum, mel_coefficients)
    response = executor.execute_logmel_frame(
        LogMelFrameRequest(
            audio_path=audio_path,
            power_spectrum=power_spectrum.astype(np.uint32, copy=False)
            .astype(np.int64, copy=False)
            .tolist(),
            mel_coefficients=mel_coefficients.reshape(-1)
            .astype(np.uint16, copy=False)
            .astype(np.int64, copy=False)
            .tolist(),
            expected_output=expected_output.astype(np.uint16, copy=False)
            .astype(np.int64, copy=False)
            .tolist(),
        ),
        output_dir,
    )
    expected_dequantized = expected_output.astype(np.float32) / float(
        1 << LOG_OUTPUT_FRAC_BITS
    )
    rtl_array = np.asarray(response.rtl_output, dtype=np.float32)
    rtl_dequantized = rtl_array / float(1 << LOG_OUTPUT_FRAC_BITS)
    return LogMelFrameComparison(
        power_spectrum=power_spectrum.astype(np.uint32, copy=False)
        .astype(np.int64, copy=False)
        .tolist(),
        expected_output=expected_output.astype(np.uint16, copy=False)
        .astype(np.int64, copy=False)
        .tolist(),
        rtl_output=list(response.rtl_output),
        expected_dequantized=expected_dequantized.tolist(),
        rtl_dequantized=rtl_dequantized.tolist(),
        matched=response.matched,
        notes=list(response.notes),
    )
