from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
FEATURE_SIZE = 80
MAX_FRAMES = 3000
MODEL_ALIAS = "distil-small.en"
MODEL_REPO = "distil-whisper/distil-small.en"
LANGUAGE_TOKEN = "<|en|>"


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class TranscriptResponse:
    backend: str
    model: str
    audio_duration_seconds: float
    notes: list[str]
    segments: list[TranscriptSegment]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTranslate2 Whisper worker.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--audio", type=Path)
    source_group.add_argument("--features-npy", type=Path)
    parser.add_argument("--audio-duration-seconds", type=float)
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


def prompt_tokens_for(processor) -> list[int]:
    return processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            LANGUAGE_TOKEN,
            "<|transcribe|>",
            "<|notimestamps|>",
        ]
    )


def compute_features_cpu(processor, chunk: np.ndarray) -> np.ndarray:
    features = processor.feature_extractor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
    ).input_features
    return np.asarray(features, dtype=np.float32)


def normalize_feature_tensor(path: Path) -> np.ndarray:
    features = np.load(path)
    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 2:
        features = features[np.newaxis, ...]
    if features.ndim != 3:
        raise ValueError(
            f"expected 2D or 3D feature tensor in {path}, got shape {features.shape!r}"
        )
    if features.shape[0] != 1:
        raise ValueError(
            f"expected batch size 1 in {path}, got shape {features.shape!r}"
        )
    if features.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"expected {FEATURE_SIZE} mel bins in {path}, got shape {features.shape!r}"
        )
    if features.shape[2] <= 0 or features.shape[2] > MAX_FRAMES:
        raise ValueError(
            f"expected frame count in [1, {MAX_FRAMES}] in {path}, got shape {features.shape!r}"
        )

    return np.ascontiguousarray(features)


def infer_audio_duration_seconds(features: np.ndarray) -> float:
    return float(features.shape[2] * HOP_LENGTH) / float(SAMPLE_RATE)


def transcribe_from_features(
    *,
    model,
    processor,
    ctranslate2_module,
    features: np.ndarray,
    beam_size: int,
) -> str:
    features_view = ctranslate2_module.StorageView.from_array(
        np.ascontiguousarray(features, dtype=np.float32)
    )
    results = model.generate(
        features_view,
        [prompt_tokens_for(processor)],
        beam_size=beam_size,
    )
    token_ids = results[0].sequences_ids[0]
    return processor.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    ).strip()


def response_with_notes(model: str, *notes: str) -> TranscriptResponse:
    return TranscriptResponse(
        backend="ct2-python",
        model=model,
        audio_duration_seconds=0.0,
        notes=list(notes),
        segments=[],
    )


def main() -> int:
    args = parse_args()

    if args.audio is not None and not args.audio.exists():
        print(
            response_with_notes(
                MODEL_REPO,
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
                MODEL_REPO,
                "Python dependencies are incomplete for the CTranslate2 baseline.",
                "Run the worker through `uv run` from the repo root so uv provides the dependencies.",
                f"Import failure: {exc}",
            ).to_json()
        )
        return 0

    try:
        model_path = download_model(MODEL_ALIAS, cache_dir="models")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = ctranslate2.models.Whisper(
            model_path,
            device=args.device,
            compute_type=args.compute_type,
        )

        notes = [
            "Direct CTranslate2 baseline with a shared features -> CT2 inference path and baked-in English mode.",
            f"Model alias: {MODEL_ALIAS}",
            f"Model path: {model_path}",
            f"Device: {args.device}",
            f"Compute type: {args.compute_type}",
        ]
        if args.initial_prompt:
            notes.append(
                "Initial prompt is accepted by the CLI but not threaded into generation yet."
            )

        segments: list[TranscriptSegment] = []
        audio_duration_seconds = 0.0

        if args.audio is not None:
            audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
            if len(audio) == 0:
                print(
                    response_with_notes(
                        MODEL_REPO,
                        f"Decoded zero samples from {args.audio}.",
                        "No transcription was attempted.",
                    ).to_json()
                )
                return 0

            audio_duration_seconds = len(audio) / SAMPLE_RATE
            notes.append(
                "Feature source: CPU WhisperFeatureExtractor frontend (30 second chunking)."
            )

            for start in range(0, len(audio), CHUNK_SAMPLES):
                stop = min(len(audio), start + CHUNK_SAMPLES)
                chunk = audio[start:stop]
                if len(chunk) == 0:
                    continue

                features = compute_features_cpu(processor, chunk)
                text = transcribe_from_features(
                    model=model,
                    processor=processor,
                    ctranslate2_module=ctranslate2,
                    features=features,
                    beam_size=args.beam_size,
                )

                if not text:
                    continue

                segments.append(
                    TranscriptSegment(
                        start_seconds=start / SAMPLE_RATE,
                        end_seconds=stop / SAMPLE_RATE,
                        text=text,
                    )
                )
        else:
            if not args.features_npy.exists():
                print(
                    response_with_notes(
                        MODEL_REPO,
                        f"Feature tensor file does not exist: {args.features_npy}",
                        "No transcription was attempted.",
                    ).to_json()
                )
                return 0

            features = normalize_feature_tensor(args.features_npy)
            audio_duration_seconds = (
                args.audio_duration_seconds
                if args.audio_duration_seconds is not None
                else infer_audio_duration_seconds(features)
            )
            notes.append(
                "Feature source: external precomputed log-mel tensor supplied to the worker."
            )
            notes.append(
                f"Feature tensor shape: {tuple(int(dim) for dim in features.shape)!r}"
            )

            text = transcribe_from_features(
                model=model,
                processor=processor,
                ctranslate2_module=ctranslate2,
                features=features,
                beam_size=args.beam_size,
            )
            if text:
                segments.append(
                    TranscriptSegment(
                        start_seconds=0.0,
                        end_seconds=audio_duration_seconds,
                        text=text,
                    )
                )

        print(
            TranscriptResponse(
                backend="ct2-python",
                model=MODEL_REPO,
                audio_duration_seconds=audio_duration_seconds,
                notes=notes,
                segments=segments,
            ).to_json()
        )
        return 0
    except Exception as exc:
        print(
            response_with_notes(
                MODEL_REPO,
                "CTranslate2 worker failed before transcription completed.",
                f"Exception: {exc}",
            ).to_json()
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
