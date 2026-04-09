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
- `python/ct2_worker.py`: placeholder worker for the CTranslate2 path
- `docs/architecture.md`: stage split and milestones
- `fpga/README.md`: hardware-side ownership and folder intent

## Commands

```bash
cargo run -- plan
cargo run -- transcribe samples/demo.wav --backend ct2-python
cargo run -- tui
```

## Next steps

1. Replace `python/ct2_worker.py` with a real CTranslate2 worker using `distil-small.en`.
2. Define a stable host/FPGA packet format for feature chunks, encoder output, and decoder work units.
3. Build a generic `int8` matrix engine on the FPGA before attempting per-layer specialization.
