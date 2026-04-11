from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

MODEL_ALIAS = "distil-small.en"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark direct faster-whisper transcription on one sample."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=MODEL_ALIAS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--download-root", type=Path, default=Path("models"))
    return parser.parse_args()


def run_once(model, audio: Path, beam_size: int) -> tuple[str, float]:
    segments, info = model.transcribe(str(audio), beam_size=beam_size)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, float(info.duration)


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        raise SystemExit(f"audio file does not exist: {args.audio}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not available in the current environment."
        ) from exc

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(args.download_root),
    )

    warmup_runs = max(args.warmup, 0)
    measured_runs = max(args.iterations, 1)

    transcript_text = ""
    audio_duration_seconds = 0.0

    for _ in range(warmup_runs):
        transcript_text, audio_duration_seconds = run_once(
            model, args.audio, args.beam_size
        )

    times: list[float] = []
    for index in range(measured_runs):
        started = time.perf_counter()
        transcript_text, audio_duration_seconds = run_once(
            model, args.audio, args.beam_size
        )
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"run {index + 1}: {elapsed:.3f}s")

    avg_seconds = statistics.mean(times)
    min_seconds = min(times)
    max_seconds = max(times)

    print(f"backend: faster-whisper-direct")
    print(f"model: {args.model}")
    print(f"audio_duration_seconds: {audio_duration_seconds:.3f}")
    print(f"warmup_runs: {warmup_runs}")
    print(f"measured_runs: {measured_runs}")
    print(f"avg_seconds: {avg_seconds:.3f}")
    print(f"min_seconds: {min_seconds:.3f}")
    print(f"max_seconds: {max_seconds:.3f}")

    if audio_duration_seconds > 0:
        print(f"avg_realtime_factor: {avg_seconds / audio_duration_seconds:.3f}x")
    else:
        print("avg_realtime_factor: n/a")

    preview = transcript_text[:200]
    print(f"transcript_preview: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
