# fpga-whisper

Hybrid Whisper project skeleton for:

- a baked-in `distil-whisper/distil-small.en` model target
- a Rust CLI/TUI frontend
- a host runtime that can start from CTranslate2 and progressively hand stages to an FPGA

## Why this shape

`faster-whisper` is the reference implementation, but the long-term hybrid runtime should be built around the same boundaries that CTranslate2 exposes:

- feature extraction
- encoder
- decoder math
- decode policy

That gives the project a clean path from:

1. host-only CTranslate2 baseline
2. FPGA feature extraction
3. FPGA encoder
4. FPGA hybrid runtime with host-side decode control

## Repo layout

- `src/cli.rs`: top-level command surface
- `src/tui.rs`: minimal terminal UI shell
- `src/backend/ct2_python.rs`: host-side baseline backend
- `src/backend/fpga_hybrid.rs`: future FPGA handoff backend
- `python/ct2_worker.py`: direct CTranslate2 worker with graceful dependency fallback
- `pyproject.toml`: `uv`-managed Python dependencies for the host baseline
- `uv.lock`: locked Python dependency graph for reproducible runs
- `docs/architecture.md`: stage split and milestones
- `fpga/README.md`: hardware-side ownership and folder intent

## Commands

```bash
cargo run -- plan
cargo run -- transcribe samples/demo.wav --backend ct2-python
cargo run -- transcribe samples/jfk.flac --backend fpga-sim --partition frontend
cargo run -- gemm-check
cargo run -- linear-check
cargo run -- projection-tile-check
cargo run -- benchmark samples/jfk.flac --backend ct2-python --iterations 5 --warmup 1
cargo run -- profile samples/jfk.flac --backend ct2-python --sample-interval-ms 250
cargo run -- tui
```

## Python baseline

The `ct2-python` backend now invokes `uv run python/ct2_worker.py` and expects JSON back.

Set up the baseline Python environment with:

```bash
uv lock
uv run python/ct2_worker.py --audio samples/silence.wav
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
- `initial_prompt` is parsed but not used yet

The Rust frontend uses `uv run` by default, so `cargo run -- transcribe ... --backend ct2-python` will execute against the `uv`-managed Python environment.
The model is baked into the program, so there is no CLI model switch anymore.
The language is baked in as English, so there is no CLI language switch anymore.

## FPGA Simulation Path

The first hardware-facing scaffold keeps Python out of the RTL boundary:

- Rust host orchestrates the pipeline
- `fpga-sim` writes a request JSON into `fpga/tmp/`
- Rust generates vectors, runs `iverilog`/`vvp`, and writes a response JSON back
- Rust parses the response and continues the flow

This keeps the eventual real-FPGA interface aligned with the simulator interface.

Try it with:

```bash
cargo run -- transcribe samples/jfk.flac --backend fpga-sim --partition frontend
```

Today that path exercises a real RTL smoke primitive:

- signed int16 x int16 8-lane dot product
- request vectors written by Rust
- result checked against software on the host

Above that primitive, there is now a kernel-layer validation path:

```bash
cargo run -- gemm-check
cargo run -- linear-check
```

`gemm-check` validates a tile-level matrix multiply contract.
`linear-check` validates a simple linear layer on top of that tile contract, including bias addition and a placeholder fixed-point format choice.
The GEMM path now runs through a real RTL tile module in `fpga/rtl/gemm_tile_i16x8.v`, not a host-side loop of scalar simulator calls.
`projection-tile-check` validates one real tile cut from `encoder/layer_0/ffn/linear_0` in the baked CTranslate2 `model.bin`, quantizes it to the current `Q8.8` harness, and compares the RTL result to both the quantized software path and the original float reference.
If the reference activation cache is missing, Rust invokes `python/export_reference_activation.py` once with the system `python3` interpreter to export `model.encoder.layers.0.fc1` input activations from `samples/jfk.flac` into `artifacts/reference/`. After that, Rust owns the cache loading, quantization, and simulator comparison path.

That is the right next layer before trying to wire actual Whisper projections onto the FPGA path.

## Benchmarking

Use the built-in benchmark command to measure baseline transcription latency before you start moving stages onto the FPGA:

```bash
cargo run -- benchmark samples/jfk.flac --backend ct2-python --iterations 5 --warmup 1
```

This reports:

- per-run wall-clock time
- average/min/max transcription time
- average real-time factor, computed as `elapsed_seconds / audio_duration_seconds`

Keep this command stable and run it against the same sample set when the FPGA backend comes online.

## System Profiling

Use the profile command when you want host resource usage instead of just timing:

```bash
cargo run -- profile samples/jfk.flac --backend ct2-python --sample-interval-ms 250
```

This prints:

- a summary table with elapsed time, real-time factor, average and peak CPU usage, and average and peak RAM usage
- a per-sample table showing CPU and RAM usage across the full transcription run

The CPU and RAM numbers are sampled from the backend process tree, which is the relevant baseline to compare against once the FPGA backend is implemented.

## Criterion

There is also a Criterion benchmark harness at:

- `benches/transcriber_system_profile.rs`

Run it with:

```bash
cargo bench --bench transcriber_system_profile
```

Criterion handles the timing loop, while the shared profiler collects CPU and RAM samples for the same transcription path.

## Next steps

1. Add context carry-over and timestamps to the Python baseline so it more closely matches `faster-whisper`.
2. Replace the deterministic input vector in `projection-tile-check` with a real intermediate activation slice from the host baseline.
3. Generalize the current `i16 x i16, inner=8` simulator tile into the quantized matrix engine you want to carry onto hardware.
