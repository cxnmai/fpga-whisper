# fpga-whisper

Hybrid Whisper project skeleton for:

- `distil-whisper/distil-small.en`
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
- `python/requirements-ct2.txt`: Python packages for the host baseline
- `docs/architecture.md`: stage split and milestones
- `fpga/README.md`: hardware-side ownership and folder intent

## Commands

```bash
cargo run -- plan
cargo run -- transcribe samples/demo.wav --backend ct2-python
cargo run -- tui
```

## Python baseline

The `ct2-python` backend now invokes `python/ct2_worker.py` and expects JSON back.

Install the baseline Python dependencies with:

```bash
python3 -m pip install -r python/requirements-ct2.txt
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

## Next steps

1. Add context carry-over and timestamps to the Python baseline so it more closely matches `faster-whisper`.
2. Define a stable host/FPGA packet format for feature chunks, encoder output, and decoder work units.
3. Build a generic `int8` matrix engine on the FPGA before attempting per-layer specialization.
