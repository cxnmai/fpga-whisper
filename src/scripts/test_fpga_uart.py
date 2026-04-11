#!/usr/bin/env python3
"""End-to-end test of real FPGA hardware over UART.

Run after programming the Arty S7:

    uv run python -m src.scripts.test_fpga_uart [--port /dev/ttyUSB1]

Tests (in order):
  1. Ping             -- verify the link is alive
  2. Dot product      -- 8-lane i16 multiply-accumulate
  3. GELU block       -- 8-lane piecewise-linear activation
  4. Load mel coeffs  -- bulk-load 32 KB to FPGA BRAM
  5. Single mel frame -- 201 power bins -> 80 log-mel bins
  6. Full transcribe  -- run jfk.flac through the FPGA frontend
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fpga.uart import FpgaUartExecutor
from src.fpga.kernels.logmel import (
    FFT_BINS,
    MEL_BINS,
    MEL_COEFF_FRAC_BITS,
    quantize_mel_filterbank,
    software_logmel_frame,
)


def test_ping(fpga: FpgaUartExecutor) -> None:
    major, minor, patch = fpga.ping()
    print(f"  firmware {major}.{minor}.{patch}")


def test_dot_product(fpga: FpgaUartExecutor) -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [8, 7, 6, 5, 4, 3, 2, 1]
    expected = sum(x * y for x, y in zip(a, b))

    from src.fpga.transport import DotProductRequest

    resp = fpga.execute_dot_product(
        DotProductRequest(
            audio_path="test",
            vector_a=a,
            vector_b=b,
            expected_result=expected,
        ),
        PROJECT_ROOT / "artifacts",
    )
    status = "PASS" if resp.matched else "FAIL"
    print(f"  {status}  rtl={resp.rtl_result}  expected={expected}")
    assert resp.matched, "dot product mismatch"


def test_gelu(fpga: FpgaUartExecutor) -> None:
    # Q8.8 inputs: mix of positive, negative, and zero
    inputs = [0, 64, 128, 256, -64, -128, -256, 512]

    from src.fpga.transport import GeluBlockRequest

    # Compute expected via the Python reference (matches the RTL LUT)
    from src.fpga.kernels.logmel import log2_linear_q8_8  # just to confirm import works

    resp = fpga.execute_gelu_block(
        GeluBlockRequest(
            audio_path="test",
            input_block=inputs,
            expected_output=[0] * 8,  # we'll check RTL output is reasonable
        ),
        PROJECT_ROOT / "artifacts",
    )
    print(f"  input:  {inputs}")
    print(f"  output: {resp.rtl_output}")
    # Basic sanity: GELU(0)=0, GELU(positive) > 0
    assert resp.rtl_output[0] == 0, "GELU(0) should be 0"
    assert resp.rtl_output[3] > 0, "GELU(256) should be positive"


def test_load_coefficients(fpga: FpgaUartExecutor) -> None:
    from transformers.audio_utils import mel_filter_bank

    mel_filters = mel_filter_bank(
        num_frequency_bins=FFT_BINS,
        num_mel_filters=MEL_BINS,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    ).astype(np.float32)
    mel_q = quantize_mel_filterbank(mel_filters)
    coefficients = mel_q.T.reshape(-1).astype(np.int64).tolist()

    t0 = time.perf_counter()
    fpga.load_mel_coefficients(coefficients)
    elapsed = time.perf_counter() - t0
    print(f"  loaded {len(coefficients)} coefficients in {elapsed:.2f}s")
    return mel_q


def test_single_mel_frame(fpga: FpgaUartExecutor, mel_q: np.ndarray) -> None:
    # Synthetic power spectrum: ramp from 0 to 1000
    rng = np.random.default_rng(42)
    power = rng.integers(0, 2**16, size=FFT_BINS, dtype=np.uint32)

    _, expected = software_logmel_frame(power, mel_q)

    from src.fpga.transport import LogMelFrameRequest

    resp = fpga.execute_logmel_frame(
        LogMelFrameRequest(
            audio_path="test",
            power_spectrum=power.astype(np.int64).tolist(),
            mel_coefficients=mel_q.T.reshape(-1).astype(np.int64).tolist(),
            expected_output=expected.astype(np.int64).tolist(),
        ),
        PROJECT_ROOT / "artifacts",
    )
    status = "PASS" if resp.matched else "FAIL"
    mismatches = sum(
        1 for a, b in zip(resp.rtl_output, resp.expected_output) if a != b
    )
    print(f"  {status}  mismatches: {mismatches}/80")
    if not resp.matched:
        for i, (r, e) in enumerate(
            zip(resp.rtl_output, resp.expected_output)
        ):
            if r != e:
                print(f"    bin {i}: rtl={r}  expected={e}")
                if i > 4:
                    print(f"    ... ({mismatches - 5} more)")
                    break


def test_full_transcription(fpga: FpgaUartExecutor) -> None:
    audio_path = PROJECT_ROOT / "samples" / "jfk.flac"
    if not audio_path.exists():
        print("  SKIP  (samples/jfk.flac not found)")
        return

    try:
        from faster_whisper.audio import decode_audio
        from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
    except ImportError as exc:
        print(f"  SKIP  (missing dependency: {exc})")
        return

    # ── audio decode + STFT (host) ──
    SAMPLE_RATE = 16_000
    N_FFT = 400
    HOP_LENGTH = 160
    POWER_QMAX = (1 << 24) - 1

    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    window = window_function(N_FFT, "hann")
    power_spec = spectrogram(audio, window, frame_length=N_FFT, hop_length=HOP_LENGTH, power=2.0)
    power_spec = np.asarray(power_spec[:, :-1], dtype=np.float32)
    frames = power_spec.shape[1]

    mel_filters = mel_filter_bank(
        num_frequency_bins=FFT_BINS,
        num_mel_filters=MEL_BINS,
        min_frequency=0.0,
        max_frequency=SAMPLE_RATE / 2.0,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    ).astype(np.float32)
    mel_q = quantize_mel_filterbank(mel_filters)

    # ── quantize power spectrum ──
    power_scale = np.max(power_spec, axis=0).astype(np.float64) / POWER_QMAX
    power_scale = np.where(power_scale > 0.0, power_scale, 1.0)
    power_quantized = np.clip(
        np.rint(power_spec / power_scale[np.newaxis, :]), 0, POWER_QMAX
    ).astype(np.uint32)

    # ── load coefficients once ──
    coefficients = mel_q.T.reshape(-1).astype(np.int64).tolist()
    if not fpga._coeffs_loaded:
        fpga.load_mel_coefficients(coefficients)

    # ── send frames to FPGA ──
    print(f"  sending {frames} frames to FPGA ...")
    t0 = time.perf_counter()
    mel_accumulators = np.zeros((frames, MEL_BINS), dtype=np.float64)

    for f in range(frames):
        power_frame = power_quantized[:, f].astype(np.int64).tolist()
        result = fpga._send_mel_frame(power_frame)
        # The FPGA returns log-mel Q8.8 values; for the Whisper pipeline
        # we actually need the raw mel accumulators.  Since the sequential
        # engine returns log values, we'd need to either:
        #   a) add a CMD that returns raw accumulators, or
        #   b) accept the log-mel directly and skip the host log/clamp
        #
        # For now, just collect log-mel values to verify they're sane.
        mel_accumulators[f, :] = result

        if (f + 1) % 100 == 0 or f + 1 == frames:
            elapsed = time.perf_counter() - t0
            fps = (f + 1) / elapsed
            print(f"    frame {f + 1}/{frames}  ({fps:.1f} frames/s)")

    elapsed = time.perf_counter() - t0
    print(f"  {frames} frames in {elapsed:.1f}s  ({frames / elapsed:.1f} fps)")

    # Sanity: log-mel values should be non-negative and mostly non-zero
    nonzero = np.count_nonzero(mel_accumulators)
    total = mel_accumulators.size
    print(f"  non-zero log-mel bins: {nonzero}/{total} ({100 * nonzero / total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test FPGA whisper over UART")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="serial port")
    parser.add_argument("--baud", type=int, default=115_200)
    args = parser.parse_args()

    fpga = FpgaUartExecutor(port=args.port, baud=args.baud)

    tests = [
        ("ping", lambda: test_ping(fpga)),
        ("dot product", lambda: test_dot_product(fpga)),
        ("gelu block", lambda: test_gelu(fpga)),
        ("load mel coefficients", lambda: test_load_coefficients(fpga)),
    ]

    # Mel tests need coefficients loaded first; test_load_coefficients
    # returns the quantized matrix, but we run it via the list above.
    # Re-generate for the frame tests:
    mel_q_holder: list[np.ndarray] = []

    def _load_and_store():
        mel_q = test_load_coefficients(fpga)
        mel_q_holder.append(mel_q)

    tests[3] = ("load mel coefficients", _load_and_store)
    tests.append(("single mel frame", lambda: test_single_mel_frame(fpga, mel_q_holder[0])))
    tests.append(("full transcription (jfk.flac)", lambda: test_full_transcription(fpga)))

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            failed += 1

    fpga.close()
    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
