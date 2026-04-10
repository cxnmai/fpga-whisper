# fpga-whisper

Hybrid Whisper project scaffold for:

- a baked-in `distil-whisper/distil-small.en` model target
- a **Python host runtime** managed with `uv`
- an FPGA/offload path that can progressively absorb parts of the Whisper pipeline
- RTL simulation and validation flows that stay aligned with the eventual real-FPGA boundary

## Why this shape

`faster-whisper` is the practical host-side reference, but the long-term hybrid runtime should be built around the same major boundaries that CTranslate2 exposes:

- feature extraction
- encoder
- decoder math
- decode policy

That gives the project a clean migration path from:

1. host-only CTranslate2 baseline
2. FPGA feature extraction
3. FPGA encoder
4. FPGA hybrid runtime with host-side decode control

The **host runtime is now Python-first**. The FPGA assets remain the point of the project.

## Repo layout

- `src/cli.py`: top-level Python command surface
- `src/backends/`: host runtime backends
- `src/fpga/`: simulator, transport, quantization, and kernel helpers
- `src/fpga_cli.py`: FPGA validation and sweep commands
- `src/model/ct2.py`: baked CTranslate2 `model.bin` reader
- `src/model/reference.py`: reference activation export/load helpers
- `src/profiling.py`: backend resource profiling helpers
- `src/scripts/ct2_worker.py`: direct CTranslate2 worker with graceful dependency fallback
- `src/scripts/export_reference_activation.py`: activation export helper for validation flows
- `pyproject.toml`: `uv`-managed Python project definition
- `uv.lock`: locked dependency graph for reproducible runs
- `docs/architecture.md`: stage split and milestones
- `fpga/README.md`: hardware-side ownership and folder intent

## Installation

Create or sync the Python environment with `uv`:

```bash
uv sync
```

You can also refresh the lockfile when dependencies change:

```bash
uv lock
uv sync
```

## Commands

The Python host runtime is exposed as a console script:

```bash
uv run fpga-whisper plan
uv run fpga-whisper transcribe samples/silence.wav --backend ct2-python
uv run fpga-whisper transcribe samples/jfk.flac --backend fpga-sim --partition frontend
uv run fpga-whisper gemm-check
uv run fpga-whisper logmel-frame-check
uv run fpga-whisper linear-check
uv run fpga-whisper projection-tile-check
uv run fpga-whisper projection-sweep-check
uv run fpga-whisper projection-full-check
uv run fpga-whisper projection-full-sweep-check
uv run fpga-whisper gelu-check
uv run fpga-whisper gelu-sweep-check
uv run fpga-whisper benchmark samples/jfk.flac --backend ct2-python --iterations 5 --warmup 1
uv run fpga-whisper profile samples/jfk.flac --backend ct2-python --sample-interval-ms 250
```

You can also run the packaged helper entrypoints directly:

```bash
uv run fpga-whisper-ct2-worker --audio samples/silence.wav
uv run fpga-whisper-ct2-worker --features-npy artifacts/tmp/jfk_features.npy --audio-duration-seconds 11.0
uv run fpga-whisper-export-reference-activation --audio samples/jfk.flac --positions 4 --output artifacts/reference/test.json
```

## Python host runtime

The default baseline backend is `ct2-python`.

It invokes the packaged worker at:

- `src/scripts/ct2_worker.py`

and expects JSON back.

Set up and smoke-test the baseline worker with:

```bash
uv sync
uv run fpga-whisper-ct2-worker --audio samples/silence.wav
```

The worker now has an explicit `features -> CT2` boundary for future FPGA frontend offload. It accepts either:

- `--audio <path>`: decode audio and compute log-mel features on the host
- `--features-npy <path>`: load a precomputed feature tensor and skip host feature extraction

The expected precomputed feature shape is:

- `(80, frames)` or `(1, 80, frames)`
- current Whisper frontend target: `(1, 80, 3000)` for a 30 second chunk

If the feature tensor was produced from padded audio, pass the real chunk duration explicitly:

```bash
uv run fpga-whisper-ct2-worker \
  --features-npy artifacts/tmp/jfk_features.npy \
  --audio-duration-seconds 11.0
```

Optional environment variables:

```bash
export FPGA_WHISPER_CT2_DEVICE=cpu
export FPGA_WHISPER_CT2_COMPUTE_TYPE=int8
export FPGA_WHISPER_BEAM_SIZE=1
```

Current limitations:

- fixed 30 second chunking
- no timestamps
- no VAD
- no prompt carry-over between chunks
- `initial_prompt` is parsed but not threaded into generation yet

The model is baked into the program, so there is no CLI model switch.
The language is baked in as English, so there is no CLI language switch.

## Backends

### `ct2-python`

Host-side CTranslate2 baseline. Use this as the correctness oracle.

### `fpga-sim`

Host-side integration path for simulated RTL via file-based vector exchange.

This path keeps the FPGA workflow intact:

- Python host orchestrates the pipeline
- `fpga-sim` writes each request into its own scratch directory under `fpga/tmp/`
- Python launches the direct `iverilog`/`vvp` simulator flow and reads simulator outputs back into the host runtime

This keeps the eventual real-FPGA interface aligned with the simulator interface.

### `fpga-hybrid`

Hybrid path scaffold. Host keeps control flow while the FPGA absorbs dense math stages.

## FPGA simulation path

Try the first hardware-facing scaffold with:

```bash
uv run fpga-whisper transcribe samples/jfk.flac --backend fpga-sim --partition frontend
```

Today that path performs a real frontend-assisted transcription:

- host decodes audio and computes the Whisper power spectrogram
- RTL performs batched `power spectrum -> mel accumulation`
- host applies Whisper `log10`, clamp, and normalization
- host feeds the resulting feature tensor into the shared `features -> CT2` worker path

On the included JFK sample, that produces a real transcript through the frontend boundary:

```text
[0.00..11.00] And so my fellow American, ask not what your country can do for you, ask what you can do for your country. Ask not what you can do for your country.
```

For comparison, the CPU baseline produces:

```text
[0.00..11.00] And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

The wording drift is expected at this stage because the frontend path is already numerically different from the host feature extractor.

Above that primitive, there is now a kernel-layer validation path:

```bash
uv run fpga-whisper gemm-check
uv run fpga-whisper logmel-frame-check
uv run fpga-whisper linear-check
```

`gemm-check` validates a tile-level matrix multiply contract.

`logmel-frame-check` validates the first frontend-oriented Verilog path on a real audio frame:

- host decodes audio and selects a high-energy frame
- host computes a power spectrum and quantized mel coefficients
- RTL computes `power spectrum -> mel -> log-mel`
- host compares the RTL output against the same fixed-point reference path

`linear-check` validates a simple linear layer on top of that tile contract, including bias addition and the current placeholder fixed-point format choice.

The GEMM path runs through a real RTL tile module in `fpga/rtl/gemm_tile_i16x8.v`, not a host-side loop of scalar simulator calls.

## Projection and activation validation

The project also includes model-aware validation commands:

```bash
uv run fpga-whisper projection-tile-check
uv run fpga-whisper projection-sweep-check
uv run fpga-whisper projection-full-check
uv run fpga-whisper projection-full-sweep-check
uv run fpga-whisper gelu-check
uv run fpga-whisper gelu-sweep-check
```

These commands validate real slices from the baked CTranslate2 `model.bin`:

- `projection-tile-check` validates one real tile cut from `encoder/layer_0/ffn/linear_0`
- `projection-sweep-check` runs multiple cached positions and multiple input/output windows
- `projection-full-check` accumulates adjacent `inner=8` RTL GEMM tiles into a wider projection slice
- `projection-full-sweep-check` widens that accumulated path and sweeps multiple output windows
- `gelu-check` requantizes a real accumulated projection window and runs it through the RTL GELU PWL block
- `gelu-sweep-check` runs that activation block across the wider accumulated projection sweep

If the reference activation cache is missing, the host runtime invokes the packaged exporter and writes JSON under:

- `artifacts/reference/`

After that, Python owns the cache loading, quantization, simulator invocation, and comparison path.

## Benchmarking

Use the built-in benchmark command to measure baseline transcription latency before moving stages onto the FPGA:

```bash
uv run fpga-whisper benchmark samples/jfk.flac --backend ct2-python --iterations 5 --warmup 1
```

This reports:

- per-run wall-clock time
- average/min/max transcription time
- average real-time factor, computed as `elapsed_seconds / audio_duration_seconds`

Keep this command stable and run it against the same sample set as FPGA ownership expands.

## System profiling

Use the profile command when you want host resource usage instead of just timing:

```bash
uv run fpga-whisper profile samples/jfk.flac --backend ct2-python --sample-interval-ms 250
```

This prints:

- a summary table with elapsed time, real-time factor, average and peak CPU usage, and average and peak RAM usage
- a per-sample table showing CPU and RAM usage across the transcription run

The CPU and RAM numbers are sampled from the backend process tree, which is the relevant baseline to compare against once the FPGA backend becomes real hardware.

## FPGA assets stay central

This project is **not** about removing the FPGA path. The host runtime moved to Python, but the hardware-side assets remain central:

- `fpga/rtl/*`
- `fpga/tb/*`
- `fpga/tmp/*`
- the simulator contract
- the model partitioning concepts
- the quantization and validation flows used to decide what moves onto the FPGA

The goal remains the same: wire in an FPGA to run meaningful parts of Whisper on it.

## Next steps

1. Thread prompt/context carry-over and timestamps into the Python CTranslate2 baseline.
2. Replace more host-side Whisper math with explicit FPGA-owned stage boundaries.
3. Tighten the quantization contract for wider accumulated output tiles.
4. Tighten the GELU approximation against float reference behavior.
5. Build the second FFN linear on top of the accumulated projection + GELU path.
6. Tighten the chunk-level frontend path until the `fpga-sim` transcription output converges toward the CPU baseline.
7. Replace the simulation-only transport with a real FPGA transport while keeping the host/runtime contracts stable.
