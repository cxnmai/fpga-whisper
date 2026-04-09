# FPGA workspace

This directory is the hardware-side root for the project.

Suggested subdirectories:

- `rtl/`: reusable DSP, BRAM, and matrix-engine blocks
- `sim/`: test benches and waveform-oriented validation
- `build/`: synthesis and bitstream outputs
- `scripts/`: export, packing, and memory-image tooling

The target split for the FPGA is:

- feature extraction
- encoder
- decoder math

The host should continue to own:

- audio decoding
- beam search / greedy decode control
- tokenization
- post-processing
