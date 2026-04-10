# FPGA workspace

This directory is the hardware-side root for the project.

Suggested subdirectories:

- `rtl/`: reusable DSP, BRAM, and matrix-engine blocks
- `sim/`: test benches and waveform-oriented validation
- `build/`: synthesis and bitstream outputs
- `scripts/`: export, packing, and memory-image tooling
- `tb/`: standalone test benches for iverilog/vvp
- `vectors/`: request/response vectors for host-simulator exchange
- `tmp/`: transient simulator IO produced by the Rust scaffold, one scratch directory per invocation

The target split for the FPGA is:

- feature extraction
- encoder
- decoder math

The host should continue to own:

- audio decoding
- beam search / greedy decode control
- tokenization
- post-processing

Current simulator scaffold:

- Rust backend: `fpga-sim`
- transport contract: JSON request/response files in per-run scratch directories under `fpga/tmp/`
- simulator invocation: direct Rust -> `iverilog`/`vvp`
- first real RTL primitive: `fpga/rtl/dot_product_i16x8.v`
- first real testbench: `fpga/tb/dot_product_i16x8_tb.v`
- first reusable tile primitive: `fpga/rtl/gemm_tile_i16x8.v`
- accumulator-aware tile primitive: `fpga/rtl/gemm_tile_accum_i16x8.v`
- first tile testbench: `fpga/tb/gemm_tile_i16x8_tb.v`
