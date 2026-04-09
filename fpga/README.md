# FPGA workspace

This directory is the hardware-side root for the project.

Suggested subdirectories:

- `rtl/`: reusable DSP, BRAM, and matrix-engine blocks
- `sim/`: test benches and waveform-oriented validation
- `build/`: synthesis and bitstream outputs
- `scripts/`: export, packing, and memory-image tooling
- `tb/`: standalone test benches for iverilog/vvp
- `vectors/`: request/response vectors for host-simulator exchange
- `tmp/`: transient simulator IO produced by the Rust scaffold

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
- transport contract: JSON request/response files in `fpga/tmp/`
- bridge script: `fpga/scripts/run_fpga_sim.py`
- first placeholder RTL: `fpga/rtl/feature_stage_stub.v`
