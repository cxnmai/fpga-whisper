from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

SAMPLE_RATE = 16_000
MODEL_REPO = "distil-whisper/distil-small.en"
TARGET_LAYER = "model.encoder.layers.0.fc1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one Whisper activation slice.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decode_audio(audio_path: Path) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(process.stdout, dtype="<i2").astype(np.float32)
    return audio / 32768.0


def export_reference_activation(
    audio: Path,
    positions: int,
    output: Path,
) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()

    audio_samples = decode_audio(audio)
    processor = WhisperProcessor.from_pretrained(MODEL_REPO)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float32,
    )
    model.eval()

    activation_holder: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs):
        activation_holder["value"] = inputs[0].detach().cpu()

    hook = model.model.encoder.layers[0].fc1.register_forward_pre_hook(capture_input)

    features = processor.feature_extractor(
        audio_samples,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(next(model.parameters()).dtype)

    with torch.no_grad():
        model.model.encoder(features)

    hook.remove()

    activation = activation_holder["value"].squeeze(0)
    exported_positions = min(positions, int(activation.shape[0]))
    export = {
        "model_repo": MODEL_REPO,
        "audio_path": str(audio),
        "layer_name": TARGET_LAYER,
        "sequence_length": int(activation.shape[0]),
        "exported_positions": exported_positions,
        "hidden_size": int(activation.shape[-1]),
        "activations": activation[:exported_positions].tolist(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    return export_reference_activation(
        audio=args.audio,
        positions=args.positions,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
