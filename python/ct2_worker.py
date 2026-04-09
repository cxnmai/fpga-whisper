#!/usr/bin/env python3
"""Minimal CTranslate2 worker for the Rust frontend.

This worker is intentionally conservative:
- direct CTranslate2 execution for Whisper
- fixed 30 second chunking
- no VAD, timestamps, or context carry-over yet

It returns JSON on stdout for both success and expected environment failures,
so the Rust frontend can stay usable before the Python environment is complete.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class TranscriptResponse:
    backend: str
    model: str
    notes: list[str]
    segments: list[TranscriptSegment]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTranslate2 Whisper worker.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--language")
    parser.add_argument("--initial-prompt")
    parser.add_argument(
        "--device",
        default=os.environ.get("FPGA_WHISPER_CT2_DEVICE", "cpu"),
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("FPGA_WHISPER_CT2_COMPUTE_TYPE", "int8"),
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.environ.get("FPGA_WHISPER_BEAM_SIZE", "1")),
    )
    return parser.parse_args()


def response_with_notes(model: str, *notes: str) -> TranscriptResponse:
    return TranscriptResponse(
        backend="ct2-python",
        model=model,
        notes=list(notes),
        segments=[],
    )


def format_language_token(language: str | None) -> str:
    if not language:
        return "<|en|>"
    if language.startswith("<|") and language.endswith("|>"):
        return language
    return f"<|{language}|>"


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        print(
            response_with_notes(
                args.model_repo,
                f"Audio file does not exist: {args.audio}",
                "No transcription was attempted.",
            ).to_json()
        )
        return 0

    try:
        import ctranslate2
        from faster_whisper.audio import decode_audio
        from faster_whisper.utils import download_model
        from transformers import WhisperProcessor
    except ImportError as exc:
        print(
            response_with_notes(
                args.model_repo,
                "Python dependencies are incomplete for the CTranslate2 baseline.",
                "Install the packages listed in python/requirements-ct2.txt.",
                f"Import failure: {exc}",
            ).to_json()
        )
        return 0

    try:
        model_path = download_model(args.model, cache_dir="models")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = ctranslate2.models.Whisper(
            model_path,
            device=args.device,
            compute_type=args.compute_type,
        )

        audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
        if len(audio) == 0:
            print(
                response_with_notes(
                    args.model_repo,
                    f"Decoded zero samples from {args.audio}.",
                    "No transcription was attempted.",
                ).to_json()
            )
            return 0

        notes = [
            "Direct CTranslate2 baseline with fixed 30 second chunking.",
            f"Model alias: {args.model}",
            f"Model path: {model_path}",
            f"Device: {args.device}",
            f"Compute type: {args.compute_type}",
        ]
        if args.initial_prompt:
            notes.append(
                "Initial prompt is accepted by the CLI but not threaded into generation yet."
            )

        segments: list[TranscriptSegment] = []

        for chunk_index, start in enumerate(range(0, len(audio), CHUNK_SAMPLES)):
            stop = min(len(audio), start + CHUNK_SAMPLES)
            chunk = audio[start:stop]
            if len(chunk) == 0:
                continue

            features = processor.feature_extractor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
            ).input_features

            language_token = format_language_token(args.language)
            if args.language is None:
                detected = model.detect_language(features)[0][0]
                language_token = detected[0]
                notes.append(
                    f"Chunk {chunk_index}: detected language {detected[0]} with score {detected[1]:.4f}."
                )

            prompt_tokens = processor.tokenizer.convert_tokens_to_ids(
                [
                    "<|startoftranscript|>",
                    language_token,
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ]
            )

            results = model.generate(
                features,
                [prompt_tokens],
                beam_size=args.beam_size,
            )

            token_ids = results[0].sequences_ids[0]
            text = processor.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()

            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start_seconds=start / SAMPLE_RATE,
                    end_seconds=stop / SAMPLE_RATE,
                    text=text,
                )
            )

        print(
            TranscriptResponse(
                backend="ct2-python",
                model=args.model_repo,
                notes=notes,
                segments=segments,
            ).to_json()
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive path for early bring-up
        print(
            response_with_notes(
                args.model_repo,
                "CTranslate2 worker failed before transcription completed.",
                f"Exception: {exc}",
            ).to_json()
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
